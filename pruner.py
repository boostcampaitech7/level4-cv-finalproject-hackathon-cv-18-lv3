
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

from LLMPruner.pruner import hf_llama_pruner
import LLMPruner.torch_pruning as tp
from LLMPruner.torch_pruning import dependency
import LLMPruner.torch_pruning as tp 
from transformers.models.llama.modeling_llama import LlamaRMSNorm

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
    cfg.config.model.token = os.getenv("HF_TOKEN")
    
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets
    
    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # print config
    cfg.pretty_print()

    # build datasets
    #datasets = {
    #    "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
    #    "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        # "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    #}
    #calib_samples = [datasets["train"][i] for i in range(10)]
    # build model
    if not args.dryrun:
        model = load_model(model_config)
    else: # load small dummy language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)
    print(model)
    
    text = "dummy text for forward pass"
    
    inputs = model.llama_tokenizer(text, return_tensors="pt")
    example_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    
    llama_module = model.model
    
    kwargs = {
        "importance": hf_llama_pruner.TaylorImportance(group_reduction="sum", taylor="param_first"),
        "global_pruning": True,
        "iterative_steps": 1,
        "ch_sparsity": 0.2,

        "consecutive_groups": {
            layer.self_attn.k_proj: layer.self_attn.head_dim
            for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: hf_llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": [llama_module.model.layers[i].self_attn.k_proj for i in range(2, 16)] +
                              [llama_module.model.layers[i].mlp.gate_proj for i in range(2, 16)]
    }
    pruner = tp.pruner.MetaPruner(
        model=llama_module,         
        example_inputs=example_inputs,
        **kwargs
    )                                                                                                                                     
    llama_module.train()  
    outputs = llama_module(**example_inputs, labels=example_inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    
    pruner.step()
    
    # build runner
    #runner = Runner(cfg, model, datasets, job_id, args.dryrun)
    remaining_params = sum(p.numel() for p in llama_module.parameters() if p.requires_grad)
    print(f"Remaining params after pruning: {remaining_params}")
    # train
    #runner.train()


if __name__ == "__main__":
    main()