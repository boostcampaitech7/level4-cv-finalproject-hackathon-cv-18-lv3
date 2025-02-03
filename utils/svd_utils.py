import torch
import torch.nn as nn
import torch.nn.functional as F

class SVDLinear(nn.Module):

    def __init__(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vt: torch.Tensor,
        z_param: nn.Parameter,
        out_dim: int,
        in_dim: int,
        bias: torch.nn.Parameter = None,
        clamp_min: float = 1e-6,
        clamp_max: float = 1e6
    ):
        super().__init__()

        self.register_buffer("U", U.float())
        self.register_buffer("S", S.float())
        self.register_buffer("Vt", Vt.float())

        self.z_param = z_param

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.out_dim = out_dim
        self.in_dim = in_dim

        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        S_new = torch.clamp(self.S * self.z_param, min=self.clamp_min, max=self.clamp_max)

        diag_s = torch.diag(S_new)
        W_new = self.U @ diag_s @ self.Vt
        W_new = W_new.to(dtype=x.dtype)

        B, T, C = x.shape
        x_2d = x.view(B * T, C)

        out_2d = F.linear(x_2d, W_new, self.bias)
        out = out_2d.view(B, T, self.out_dim)

        return out