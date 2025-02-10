import torch
import torch.nn as nn

##############################################################################
# 전역적으로 중요도를 누적할 딕셔너리
##############################################################################
global_importance_dict = {}

def get_module_key(module: nn.Module) -> str:
    """
    모듈을 고유하게 식별하기 위한 키를 반환한다.
    예: "LlamaAttention_140523146265648" 처럼 클래스 이름 + id 조합.
    """
    return f"{module.__class__.__name__}_{id(module)}"

def init_global_importance_dict(model: nn.Module):
    """
    모델 내 모든 관심 모듈(Linear, LayerNorm 등)에 대한 중요도 누적용 구조를 초기화한다.
    """
    global global_importance_dict
    global_importance_dict.clear()

    for name, mod in model.named_modules():
        # 예) nn.Linear, LlamaAttention, RMSNorm, Embedding 등 필요한 모듈만 따로 선별 가능
        if isinstance(mod, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            key = get_module_key(mod)
            # 중요도를 누적할 텐서를 저장할 공간을 None 또는 0으로 초기화
            global_importance_dict[key] = None

def accumulate_importance(module_key: str, importance_tensor: torch.Tensor):
    """
    전역 딕셔너리에 중요도를 계속 더해나가는 함수
    """
    global global_importance_dict
    if global_importance_dict[module_key] is None:
        # 아직 아무 데이터가 없는 경우
        global_importance_dict[module_key] = importance_tensor.cpu()
    else:
        # 이미 누적된 값이 있으면 덧셈
        global_importance_dict[module_key] += importance_tensor.cpu()

##############################################################################
# Hook 함수들
##############################################################################
def attn_layer_importance_acc_hook(module, ins, outs):
    """
    예: (B, T, Hidden) 형태 self-attn 출력에 대한 L2 norm 합
    """
    if isinstance(outs, tuple):
        outs = outs[0]  # tuple일 경우 첫 번째 텐서가 실제 출력
    shape = outs.shape  # (B, T, E)
    outs_flat = outs.view(-1, shape[-1])  # (B*T, E)
    importance = torch.norm(outs_flat, dim=-1).sum()  # 스칼라
    # 전역 딕셔너리에 누적
    module_key = get_module_key(module)
    accumulate_importance(module_key, importance)

def neuron_importance_acc_hook(module: nn.Linear, ins, outs):
    """
    (B, T, out_features) 형태의 출력이 있을 때, out_features별 중요도를 누적
    """
    if not isinstance(module, nn.Linear):
        return

    # outs.shape = (B, T, out_features)
    # 각 out_feature(=채널) 방향으로 합산
    importance = outs.detach().sum(dim=(0, 1))  # shape = (out_features,)
    module_key = get_module_key(module)
    accumulate_importance(module_key, importance)

def embedding_importance_acc_hook(module, ins, outs):
    """
    (B, T, E) 형태 Embedding/LayerNorm 출력에 대한 채널별 합산
    """
    importance = outs.detach().sum(dim=(0, 1))  # (E,)
    module_key = get_module_key(module)
    accumulate_importance(module_key, importance)

##############################################################################
# Hook 등록/제거 함수
##############################################################################
def remove_llama_hooks(model):
    """
    기존 forward_hook 제거 + calculated_importance 속성 제거 (기존 hooks.py 내용 참고)
    """
    for name, mod in model.named_modules():
        mod._forward_hooks.clear()
        if hasattr(mod, "calculated_importance"):
            del mod.calculated_importance

def register_llama_acc_hooks(model):
    """
    LLaMA 구조에 맞춰, 모듈에 Hook 달아서 전역 딕셔너리에 누적되도록 설정.
    """
    # 먼저 전역 딕셔너리 초기화
    init_global_importance_dict(model)

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        
        # Self-Attention
        attn = decoder_layer.self_attn
        
        # (1) attn 출력
        attn.register_forward_hook(attn_layer_importance_acc_hook)

        # (2) 각 Linear(q, k, v, o)별로 out_features 중요도
        attn.q_proj.register_forward_hook(neuron_importance_acc_hook)
        attn.k_proj.register_forward_hook(neuron_importance_acc_hook)
        attn.v_proj.register_forward_hook(neuron_importance_acc_hook)
        attn.o_proj.register_forward_hook(neuron_importance_acc_hook)

        # MLP
        mlp = decoder_layer.mlp
        mlp.gate_proj.register_forward_hook(neuron_importance_acc_hook)
        mlp.up_proj.register_forward_hook(neuron_importance_acc_hook)
        mlp.down_proj.register_forward_hook(neuron_importance_acc_hook)

        # LayerNorm(혹은 RMSNorm)
        decoder_layer.input_layernorm.register_forward_hook(embedding_importance_acc_hook)
        decoder_layer.post_attention_layernorm.register_forward_hook(embedding_importance_acc_hook)