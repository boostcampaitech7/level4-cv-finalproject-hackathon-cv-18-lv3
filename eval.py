import argparse
import json
import random
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import subprocess
from transformers import DynamicCache
from tqdm import tqdm
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import WhisperFeatureExtractor

# Custom modules
from utils.salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils.utils import get_dataloader, prepare_sample
from utils.metrics import compute_wer, compute_spider
from dataset import SALMONNDataset
from models.salmonn import SALMONN

from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", 
        type=str, 
        help='path to configuration file', 
        default='configs/salmonn_eval_config.yaml'
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # --- Deprecated options ---

    parser.add_argument("--skip_scoring", action='store_false', default=True, 
                    help="(Deprecate) If True, skip scoring after inference. Use --mode instead. This option will be removed in a future version.")
    # --- Deprecated options end ---
    # --- New options ---
    parser.add_argument("--mode", type=str, default="submission", 
                    help="Mode to evaluate. Supports submission and validation modes for ASR and AAC tasks.", 
                    choices=['submission','valid']) 
    
    parser.add_argument('--tasks', nargs='+', help='arc aac latency')
    # --- New options end ---
    
    parser.add_argument("--num_it", type=int, default=100)
    parser.add_argument("--num_warmup", type=int, default=10)

    args = parser.parse_args()

    if args.tasks is None:
        raise ValueError("--task must be provided")

    # --- Override Previous Version Args ---
    args.make_submission = args.mode

    return args

def get_dataset(dataset_cfg, run_cfg, task, submission):
    testset = SALMONNTestDataset(
        dataset_cfg.prefix, dataset_cfg.test_ann_path, dataset_cfg.whisper_path, task, submission
    )

    test_loader = get_dataloader(testset, run_cfg, is_train=False, use_distributed=False)
    return test_loader

def replace_test_ann_path(cfg):
    if "test_ann_path" not in cfg.config.datasets.keys():
        if args.task == "asr":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_asr
        elif args.task == "aac":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_aac
    return cfg


def load_model(salmonn_preprocessor):
    model = salmonn_preprocessor.llama_model
    tokenizer = salmonn_preprocessor.llama_tokenizer
    return model, tokenizer


def load_preprocessor(cfg):
    salmonn_preprocessor = SALMONN.from_config(cfg.config.model)
    salmonn_preprocessor.to(cfg.config.run.device)
    salmonn_preprocessor.eval()
    return salmonn_preprocessor


class MockDataset(SALMONNDataset):
    def __init__(self, cfg, sr, audio_length, dataset_length):
        self.sr = sr
        self.audio_length = audio_length
        self.dataset_length = dataset_length
        self.prefix = cfg.config.datasets.prefix
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(
            cfg.config.datasets.whisper_path
        )
        self.random_sample = np.random.randn(self.sr * self.audio_length)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        audio = self.random_sample.copy()
        spectrogram = self.wav_processor(
            audio, sampling_rate=self.sr, return_tensors="pt"
        )["input_features"].squeeze()
        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": "test",
            "task": "asr",
            "Q": "",
            "id": idx,
        }

    @staticmethod
    def make_mock_dataloader(cfg, sr, audio_length, dataset_length=100):
        dataset = MockDataset(cfg, sr, audio_length, dataset_length)
        return get_dataloader(
            dataset, cfg.config.run, is_train=False, use_distributed=False
        )


def get_gpu_memory_usage():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    gpu_memory = int(result.strip().split("\n")[0])
    return gpu_memory


def model_inference(cfg, samples, test_prompt, salmonn):
    # TTFT
    start_time = time.time()
    llm = salmonn.llama_model

    batch_size = samples["spectrogram"].shape[0]
    spectrogram = samples["spectrogram"]
    raw_wav = samples.get("raw_wav", None)
    audio_padding_mask = samples.get("padding_mask", None)
    speech_embeds, speech_atts = salmonn.encode_speech(
        spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
    )

    prompts = [test_prompt[task] for task in samples["task"]]
    templated_prompts = [
        cfg.config.model.prompt_template.format(prompt) for prompt in prompts
    ]

    speech_embeds, speech_atts = salmonn.prompt_wrap(
        speech_embeds, speech_atts, templated_prompts, multi_prompt=True
    )

    bos = (
        torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        )
        * salmonn.llama_tokenizer.bos_token_id
    )
    bos_embeds = (
        llm.model.embed_tokens(bos)
        if not salmonn.lora
        else llm.model.model.embed_tokens(bos)
    )
    atts_bos = speech_atts[:, :1]

    speech_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
    speech_atts = torch.cat([atts_bos, speech_atts], dim=1)

    outputs = llm.model(
        inputs_embeds=speech_embeds,
        attention_mask=speech_atts,
    )
    end_time = time.time()
    ttft = end_time - start_time

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
    past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

    # TPOT
    start_time = time.time()
    with torch.no_grad():
        _ = llm.model(next_token, past_key_values=past_key_values, use_cache=True)
    end_time = time.time()
    tpot = end_time - start_time

    inference_time = ttft + tpot
    return inference_time, ttft, tpot


def main(args):
    # 기존 입력
    # python evaluate_salmonn.py --mode submission_asr
    # submission_asr, submission_aac, valid_asr, valid_aac
    # 변경
    # pythone eval.py --mode submission --tasks asr aac latency

    cfg = Config(args)
    # cfg = replace_test_ann_path(cfg) # asr, aac에 따라 .yaml에 설정되어 있는 경로를
    # cfg.config.datasets.test_ann_path을 설정함

    assert cfg.config.model.token in ('', "", "<hf_token>"), "Please remove the hf_token from the .yaml file. You must replace it with '' or <hf_token> and create .env file and write 'HF_TOKEN=<your token>' in it to safetly preceed"
    assert load_dotenv(".env"), "Please create .env file and write 'HF_TOKEN=<your token>'"
    cfg.config.model.token = os.getenv("HF_TOKEN")

    # # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    # Load data 
    # 설정한 .yaml에 따라 cfg.config.datasets.test_ann_path을 바탕으로 데이터셋을 받아옴
    # 이때 submission이면 submission 생성할 수 있도록 하고, 그 외의 경우 ref를 받아옴
    # dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task, args.make_submission)

    # test에 사용되는 prompt
    with open("/data/yh/level4-cv-finalproject-hackathon-cv-18-lv3/data/prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)

    for task in args.tasks:
        args.task = task
        print(f"{task} evaluation start")
        if task in ('asr', 'aac'):
            cfg = replace_test_ann_path(cfg)
            
            dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task, args.make_submission)
            # Evaluation
            testset_ids, hyps, refs = [], [], []
            for samples in tqdm(dataloader):
                testset_id = samples["testset_id"]
                testset_ids.extend(testset_id)

                # Preprocess
                samples = prepare_sample(samples, cuda_enabled=torch.cuda.is_available())
                batch_size = samples["spectrogram"].shape[0]
                spectrogram = samples["spectrogram"]
                raw_wav = samples.get("raw_wav", None)
                audio_padding_mask = samples.get("padding_mask", None)
                speech_embeds, speech_atts = salmonn_preprocessor.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

                # Add prompt embeds + audio embed 
                prompts = [test_prompt[task] for task in samples['task']]
                templated_prompts = [cfg.config.model.prompt_template.format(prompt) for prompt in prompts]

                speech_embeds, speech_atts = salmonn_preprocessor.prompt_wrap(speech_embeds, speech_atts, templated_prompts, multi_prompt=True)
                bos = torch.ones(
                    [batch_size, 1],
                    dtype=torch.int32,
                    device=speech_embeds.device,
                ) * tokenizer.bos_token_id

                bos_embeds = llama_model.model.model.embed_tokens(bos)
                atts_bos = speech_atts[:, :1]

                embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
                attns = torch.cat([atts_bos, speech_atts], dim=1)

                generate_cfg = cfg.config.generate

                # Generation
                outputs = llama_model.model.generate(
                    inputs_embeds=embeds,
                    pad_token_id=llama_model.config.eos_token_id[0],
                    max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                    num_beams=generate_cfg.get("num_beams", 4),
                    do_sample=generate_cfg.get("do_sample", False),
                    min_length=generate_cfg.get("min_length", 1),
                    temperature=generate_cfg.get("temperature", 1.0),
                    top_p=generate_cfg.get("top_p", 0.9),
                    repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                    length_penalty=generate_cfg.get("length_penalty", 1.0),
                    attention_mask=attns,
                )

                results = tokenizer.batch_decode(outputs)
                hyp = [result.split(generate_cfg.end_sym)[0].lower() for result in results]
                hyps.extend(hyp)

                if not args.make_submission:
                    ref = samples["text"]
                    refs.extend(ref)

            if args.make_submission:
                os.makedirs("submission_results", exist_ok=True)
                file_name = f"submission_results/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.mode}.csv"
            else:
                if args.task == 'asr':
                    compute_wer(hyps, refs)
                    
                elif args.task == 'aac':
                    compute_spider(hyps, refs)
                os.makedirs("valid_results", exist_ok=True)
                file_name = f"valid_results/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.mode}.csv"


            result_df = pd.DataFrame({"testset_id": testset_ids, "text": hyps})
            result_df.to_csv(file_name, index=False)
            
        elif task == "latency":
            dataloader = MockDataset.make_mock_dataloader(cfg, sr=16000, audio_length=10)
            sample_batch = next(iter(dataloader))
            sample_batch = prepare_sample(sample_batch, cuda_enabled=torch.cuda.is_available())
            
            # Measure memory and latency
            memory_usages = []
            inference_times = []
            ttfts = []
            tpots = []            
            
            for it in tqdm(range(args.num_it + args.num_warmup)):
                torch.cuda.synchronize()
                with torch.no_grad():
                    inference_time, ttft, tpot = model_inference(
                        cfg,
                        sample_batch,
                        test_prompt,
                        salmonn_preprocessor,
                    )
                torch.cuda.synchronize()
                after_memory_allocated = torch.cuda.max_memory_allocated()

                torch.cuda.empty_cache()  # Clear the cache to get more accurate measurements
                gc.collect()

                if it >= args.num_warmup:
                    memory_usages.append(after_memory_allocated)
                    inference_times.append(inference_time)
                    ttfts.append(ttft)
                    tpots.append(tpot)


            average_memory_usage = np.mean(memory_usages)
            average_inference_time = np.mean(inference_times)
            average_ttft = np.mean(ttfts)
            average_tpot = np.mean(tpots)

            print(
                f"Average memory used during inference: {average_memory_usage/1024**3:.4f} GB"
            )
            print(f"Average inference time: {average_inference_time:.4f} seconds")
            print(f"Average TTFT: {average_ttft:.4f} seconds")
            print(f"Average TPOT: {average_tpot:.4f} seconds")
                    
        print(f"{task} evaluation end")


if __name__ == '__main__':
    args = parse_args()

    random.seed(42)
    main(args)
