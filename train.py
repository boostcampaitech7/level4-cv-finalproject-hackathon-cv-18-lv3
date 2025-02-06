# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

from utils.utils import *
from config import Config
from utils.dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from utils.runner import Runner
from utils.runner_distillation import KD_Runner


from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--dryrun", action='store_true', help='if True, use dummy model and skip forward/backward')
    parser.add_argument("--pruned", action='store_true', help='check if llm pruned')
    parser.add_argument("--kd", action='store_true', help="if True, run Knowledge Distillation mode" )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    args = parse_args()
    cfg = Config(args)

    assert cfg.config.model.token in ('', "", "<hf_token>"), "Please remove the hf_token from the .yaml file. You must replace it with '' or <hf_token> and create .env file and write 'HF_TOKEN=<your token>' in it to safetly preceed"
    assert load_dotenv(".env"), "Please create .env file and write 'HF_TOKEN=<your token>'"
    cfg.config.model.token = os.getenv("HF_TOKEN")
    
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # Wandb logger
    global_rank = int(os.environ["RANK"])
    if global_rank == 0:
        wandb.login()
        wandb.init(project="audio_lm", name=run_config.exp_name)

    # print config
    cfg.pretty_print()

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        # "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }

    # build model
    if not args.dryrun:
        if args.kd :
            
            model_config.ckpt = model_config.ckpt_student
            student_model = load_model(model_config)

            model_config.llama_path = model_config.teacher_llama_path
            model_config.whisper_path = model_config.teacher_whisper_path
            model_config.ckpt = model_config.ckpt_teacher
            teacher_model = load_model(model_config)

            # llm 출력 층이 다르면 projection 층 


        if args.pruned:
            model_config['pruned'] = True
        model = load_model(model_config)
    else: # load small dummy language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)

    # build runner
    if args.kd :
        runner = KD_Runner(cfg, teacher_model, student_model, datasets, job_id, args.dryrun)
        print(f"Training KD mode: Alpha={model_config.alpha}, Temperature={model_config.temperature}")
    else :
        runner = Runner(cfg, model, datasets, job_id, args.dryrun)

    # train
    runner.train()


if __name__ == "__main__":
    main()