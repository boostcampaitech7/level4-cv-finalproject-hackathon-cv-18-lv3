
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
import torch.distributed as dist
from LLMPruner.pruner import hf_llama_pruner
import LLMPruner.torch_pruning as tp
from LLMPruner.torch_pruning import dependency

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
    parser.add_argument("--not_prune", action='store_false')
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
    
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # print config
    #cfg.pretty_print()

    # build model
    if not args.dryrun:
        model = load_model(model_config)
    else: # load small dummy language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)
    
    batch_size = 8
    #text_list = [f"dummy text for forward pass {i}" for i in range(batch_size)]
    text = "dummy text for forward pass"
    inputs = model.llama_tokenizer(
        text,
        padding=True,              
        truncation=True,            
        return_tensors="pt"         
    )

    example_inputs = {
        "input_ids": inputs["input_ids"].to(f"cuda:{get_rank()}"),             # shape: (batch_size, seq_len)
        "attention_mask": inputs["attention_mask"].to(f"cuda:{get_rank()}"),   # shape: (batch_size, seq_len)
    }
    
    print(f"[Rank {get_rank()}] input_ids shape: {example_inputs['input_ids'].shape}")
    print(f"[Rank {get_rank()}] attention_mask shape: {example_inputs['attention_mask'].shape}")
    
    print(f"[Rank {get_rank()}] input_ids[0, :10]: {example_inputs['input_ids'][0, :10]}")
    print(f"[Rank {get_rank()}] attention_mask[0, :10]: {example_inputs['attention_mask'][0, :10]}")
    
    llama_module =  model.llama_model.model
    
    print(llama_module)

    params = sum(p.numel() for p in llama_module.parameters())
    print(f"params before pruning: {params}")

    llama_module.to(f"cuda:{get_rank()}")
    
    kwargs = {
        "importance": hf_llama_pruner.TaylorImportance(group_reduction="sum", taylor="param_first"),
        "global_pruning": True,
        "iterative_steps": 1,
        "ch_sparsity": 0.2,

        "consecutive_groups": {
            layer.self_attn.k_proj: layer.self_attn.head_dim
            for layer in llama_module.model.layers
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
    for param in llama_module.parameters():
        param.requires_grad = True
    outputs = llama_module(**example_inputs, labels=example_inputs["input_ids"])
    loss = outputs.loss
    print(f"loss is {loss} in rank {get_rank()}")
    loss.backward()
    
    pruner.step()

    
    remaining_params = sum(p.numel() for p in llama_module.parameters() if p.requires_grad)
    print(f"Remaining params after pruning: {remaining_params} in {get_rank()}")

    torch.save(llama_module, "./outputs_pruned/model.pt")
    print("Pruning done & saved")
    
if __name__ == "__main__":
    main()