# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import os
import json
import time
import datetime
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import wandb

from utils.dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from utils.logger import MetricLogger, SmoothedValue
from utils.utils import get_dataloader, prepare_sample
from optims import get_optimizer, LinearWarmupCosineLRScheduler
from utils.metrics import compute_wer, compute_spider

class Runner:
    def __init__(self, cfg, model, datasets, job_id, dryrun):
        super().__init__()

        self.config = cfg

        # dryrun (test with dummy model)
        self.dryrun = dryrun

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        self.device = torch.device(self.config.config.run.device)
        self.use_distributed = self.config.config.run.use_distributed
        self.start_epoch = 0
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.evaluate_only = self.config.config.run.evaluate
        self.cuda_enabled = (self.device.type == "cuda")

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r") as f:
                    self.test_prompt_dict = json.load(f)
            except:
                print("Failed to load test prompt! Try to use utf-8 encoding.")
                with open(test_prompt_path, "r", encoding="utf-8") as f:
                    self.test_prompt_dict = json.load(f)
            for k in self.test_prompt_dict.keys():
                self.test_prompt_dict[k] = self.prompt_template.format(self.test_prompt_dict[k])

        else:
            self.test_prompt_dict = None

        # model
        self._model = model
        self._model.to(self.device)
        if self.use_distributed:
            self.model = DDP(
                self._model, device_ids=[self.config.config.run.gpu]
            )
        else:
            self.model = self._model

        # dataloaders
        self.train_loader = get_dataloader(datasets["train"], self.config.config.run, is_train=True, use_distributed=self.use_distributed)
        self.valid_loader = get_dataloader(datasets["valid"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)
        # self.test_loader = get_dataloader(datasets["test"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)

        # scaler
        self.use_amp = self.config.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # optimizer & scheduler
        self.iters_per_epoch = len(self.train_loader) if self.config.config.run.epoch_based else self.config.config.run.iters_per_epoch
        self.optimizer = get_optimizer(self.model, self.config.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.config.run.optims.min_lr,
            init_lr=self.config.config.run.optims.init_lr,
            warmup_steps=self.config.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.config.run.optims.get("warmup_start_lr", -1),
        )

        self.log_config()

        self.use_svf = self.config.config.model.get("use_svf", False)
        self.svf_rank = self.config.config.model.get("svf_rank", 8)
        
        if self.use_svf:
            z_params = []
            for name, param in self.model.named_parameters():
                if "z_vector" in name:
                    param.requires_grad = True
                    z_params.append(param)
            self.svf_optimizer = torch.optim.Adam(z_params, lr=2e-3)
        else:
            self.svf_optimizer = None
        
    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)

        def model_forward(samples, no_grad=False):
            if no_grad:
                with torch.no_grad():
                    outputs = self.model(samples, verbose=True)
            else:
                outputs = self.model(samples, verbose=True)

            return outputs
        
        def preprocess_texts(texts):
            processed_texts = []
            for text in texts:
                tokens = text.split()
                unique_tokens = list(dict.fromkeys(tokens))
                unique_tokens = unique_tokens[:50] 
                if len(unique_tokens) == 0:
                    processed_text = "<empty>"
                
                processed_text = " ".join(unique_tokens)
                processed_texts.append(processed_text)
            return processed_texts
        
        def compute_rewards(samples, outputs):
            pred_texts = preprocess_texts(outputs["pred_texts"])
            ref_texts = preprocess_texts(samples["text"])

            # `pred_texts`가 `list[str]`인지 확인 후 변환
            if isinstance(pred_texts, str):
                pred_texts = [pred_texts]  # 단일 문자열이면 리스트로 변환
            elif not isinstance(pred_texts, list) or not all(isinstance(p, str) for p in pred_texts):
                raise ValueError(f"Unexpected pred_texts type: {type(pred_texts)}")
            
            # `ref_texts`가 `list[list[str]]`인지 확인 후 변환
            if isinstance(ref_texts, str):
                ref_texts = [[ref_texts]]
            elif isinstance(ref_texts, list) and all(isinstance(r, str) for r in ref_texts):
                ref_texts = [[r] for r in ref_texts]  
            elif not isinstance(ref_texts, list) or not all(isinstance(r, list) for r in ref_texts):
                raise ValueError(f"Unexpected ref_texts format: {ref_texts}")
            
            task_type = samples["task"][0] # task_type은 첫 번째 task로 통일
            print("  task_type:", task_type) # 디버그 출력: task_type 확인
            
            if task_type in ["asr"]:
                wer_value = compute_wer(pred_texts, ref_texts)
                print("  wer_value:", wer_value) # 디버그 출력: wer_value 확인
                if wer_value is None:
                    print("  Warning: WER calculation return None. Setting wer_value to 1.0.")
                    wer_value = 1.0
                reward = max(0, 1.0 - wer_value)
                print("  reward_wer:", reward) # 디버그 출력: reward 확인
                
            elif task_type in ["QA", "audiocaption", "audiocaption_v2", "gender_recognition", "phone_recognition"]:
                spider_value = 0.0
                
                try:
                    spider_result = spider(candidates=pred_texts, mult_references=ref_texts, java_path="/usr/bin/java")
                    spider_value = round(float(spider_result[0]['spider']),4)   
                except Exception as e:
                    print("  Warning: SPIDER calculation failed. Setting reward to 0.0.")
                
                reward = spider_value / 5.5
            
            else:
                reward = 0.0

            return torch.tensor(reward, dtype=torch.float, device=outputs["log_prob"].device)

        def compute_kl_divergence(ref_outputs, new_outputs):
            # KL 발산 계산. ref_outputs : 학습 전 policy, new_outputs : 학습 후 policy
            tau = 0.7
            ref_logits = ref_outputs["log_prob"].float().detach() / tau
            new_logits = new_outputs["log_prob"].float() / tau
            
            if ref_logits.size() != new_logits.size():
                min_len = min(ref_logits.size(1), new_logits.size(1))
                ref_logits = ref_logits[:, :min_len, :]
                new_logits = new_logits[:, :min_len, :]

            ref_probs = F.softmax(ref_logits, dim=-1)
            new_log_prob = F.log_softmax(new_logits, dim=-1)

            kl_div = F.kl_div(new_log_prob, ref_probs, reduction='batchmean', log_target=True)

            return torch.clamp(kl_div, min=0.0, max=20.0) # 최대값 20.0으로 제한
     
        for i in metric_logger.log_every(range(self.iters_per_epoch), self.config.config.run.log_freq, header=header, logger=self.log_writter, start_step=epoch*self.iters_per_epoch):
            if i >= self.iters_per_epoch:
                break
            
            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            if not self.dryrun:
                self.scheduler.step(cur_epoch=epoch, cur_step=i)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # loss = self.model(samples)["loss"]

                    # supervised learning
                    base_outputs = self.model(samples)
                    base_loss = base_outputs["loss"]

                    # RL
                    if self.use_svf:
                        # first pass
                        old_outputs = model_forward(samples, no_grad=True)
                        # second pass
                        new_outputs = model_forward(samples, no_grad=False)

                        rewards = compute_rewards(samples, new_outputs)
                        kl_div = compute_kl_divergence(old_outputs, new_outputs)
                        lambda_kl = 0.01
                        # RL loss
                        rl_loss = (-(new_outputs["log_prob"] * rewards).mean() + lambda_kl * kl_div).float()
                        # combined loss
                        loss = (base_loss + rl_loss).float()
                    else:
                        loss = base_loss

                if self.use_amp:
                    torch.autograd.set_detect_anomaly(True)
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % self.config.config.run.accum_grad_iters == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        if self.svf_optimizer:
                            self.scaler.unscale_(self.svf_optimizer)
                        self.scaler.step(self.optimizer)
                        if self.svf_optimizer:
                            self.scaler.step(self.svf_optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        if self.svf_optimizer:
                            self.svf_optimizer.step()

                    self.optimizer.zero_grad()
                    if self.svf_optimizer:
                        self.svf_optimizer.zero_grad() 

                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
                
                global_rank = int(os.environ["RANK"])
                if global_rank == 0:
                    wandb.log({"train/iteration": i, "train/loss": loss.item(), "train/lr": self.optimizer.param_groups[0]["lr"]})
            else: # dryrun, no model availble
                metric_logger.update(loss=0.0)
                metric_logger.update(lr=0.0)
                global_rank = int(os.environ["RANK"])
                if global_rank == 0:
                    wandb.log({"train/iteration": i, "train/loss": 0.0, "train/lr": 0.0})

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def valid_epoch(self, epoch, split, decode=False, save_json=False):
        if not self.dryrun:
            model = self.unwrap_dist_model(self.model)
            model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval: data epoch: [{}]".format(epoch)

        results = []
        for samples in metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            if not self.dryrun:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    forward_result = model(samples, verbose=True)
                loss = forward_result.get("loss", 0)
                correct = forward_result.get("correct", 0)
                total = forward_result.get("total", 1)
                res = {
                    "id": samples["id"],
                    "ground_truth": samples["text"],
                    "loss": loss.item(),
                    "acc": (correct / total).item(),
                    "total": total,
                }
            else:
                res = {
                    "id": samples["id"],
                    "ground_truth": samples["text"],
                    "loss": 0.0,
                    "acc": 0.0,
                    "total": 1,
                }

            if decode:
                if model.prompt_dict:
                    if self.test_prompt_dict is None:
                        prompts = None
                    else:
                        prompts = [self.test_prompt_dict[s] for s in samples["task"]]
                        if "Q" in samples:
                            prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, samples["Q"])]
                else:
                    prompts = None

                text = model.generate(samples, self.config.config.run, prompts=prompts)
                res["text"] = text
                res["prompt"] = prompts
                res["task"] = samples["task"]

            results.append(res)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            self.save_result(results, self.output_dir, "eval_{}_epoch_{}".format(split, epoch))

        res = {
            "loss": torch.tensor(0).float().cuda(),
            "n_sample": torch.tensor(0).float().cuda(),
            "correct": torch.tensor(0).float().cuda(),
            "n_token": torch.tensor(0).float().cuda(),
        }
        
        for item in results:
            item_loss = item["loss"]
            item_n_sample = len(item["id"])
            item_correct = item["acc"] * item["total"]
            item_n_token = item["total"]
            res["loss"] += item_loss * item_n_sample
            res["n_sample"] += item_n_sample
            res["correct"] += item_correct
            res["n_token"] += item_n_token

        if is_dist_avail_and_initialized():
            dist.all_reduce(res["loss"])
            dist.all_reduce(res["n_sample"])
            dist.all_reduce(res["correct"])
            dist.all_reduce(res["n_token"])

        ret = {"loss": 0, "agg_metrics": 0}
        ret["loss"] = (res["loss"] / res["n_sample"]).item()
        ret["agg_metrics"] = (res["correct"] / res["n_token"]).item()

        return ret

    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)

            print("result file saved to %s" % final_result_file)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid", decode=False, save_json=False)
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = valid_log["agg_metrics"]
                    if agg_metrics > best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch

                        self.save_checkpoint(cur_epoch, is_best=True)                    

                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")
                    wandb.log({"valid/epoch": cur_epoch, "valid/agg_metrics": agg_metrics})

            self.save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()

        # testing phase
        if self.evaluate_only:
            test_log = self.valid_epoch("best", "test", decode=True, save_json=True)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
