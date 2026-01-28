import math
import warnings
from typing import *
from dataclasses import dataclass
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    seed: int = 0
    d_model: int = 128
    seq_len: int = 3
    n_heads: int = 4
    n_layers: int = 1
    ln_fn: str = "og"
    ln_type: str = "off"
    mlp_type: str = "ff"
    mlp_r: int = 4
    mlp_act: str = "gelu"
    kernel_init: str = "kaiming"
    final_ln: str = "off"
    P: int = 113
    dtype: Any = torch.float32
    output_scaling: float = 1.0
    unembed: str = "separate"
    weight_init_scale: float = 1.0
    use_softmax_temp: bool = False


def myln(x):
    mu = x.mean(dim=-1).unsqueeze(-1)
    var = x.var(dim=-1, unbiased=False).unsqueeze(-1)
    denom = torch.sqrt(var + 1e-5)
    return (x - mu) / denom, denom 


class Unembed(nn.Module):
    def __init__(self, in_features, vocab_size):
        super().__init__()
        self.in_features = in_features
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.empty(in_features, vocab_size))
    
    def forward(self, x):
        return x @ self.weight


LN_FNS = {
    "og": nn.LayerNorm,
    "rms": nn.RMSNorm
}

def lecun_normal_(tensor):
    fan_in = tensor.size(1) if tensor.ndim == 2 else torch.nn.init._calculate_fan_in_and_fan_out(tensor)[0]
    std = 1.0 / math.sqrt(fan_in)
    return torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2*std, b=2*std)


def init_weights(model: nn.Module, cfg: TransformerConfig) -> None:

    def internal_apply(module: nn.Module):
        if isinstance(module, nn.Linear):
            if cfg.kernel_init == "lecun":
                lecun_normal_(module.weight)
            elif cfg.kernel_init == "kaiming":
                torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            std = (1.0 / cfg.d_model)**0.5
            torch.nn.init.normal_(module.weight, std=std)
        
        if isinstance(module, Unembed):
            std = (1.0 / cfg.d_model)**0.5
            torch.nn.init.normal_(module.weight, std=std)

    model.apply(internal_apply)
    # for experiments varying weight initialization scale
    # expected behaviour: higher weight initialization scale => more grokking
    # less weight initialization scale => less grokking & faster generalization
    if cfg.weight_init_scale != 1.0:
        print("Scaling parameters by", cfg.weight_init_scale)
        for p in model.parameters():
            p.data = p.data * cfg.weight_init_scale


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.cfg = config
        self.fc1 = nn.Linear(self.cfg.d_model, self.cfg.d_model * self.cfg.mlp_r)
        self.fc2 = nn.Linear(self.cfg.d_model * self.cfg.mlp_r, self.cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = getattr(F, self.cfg.mlp_act)(self.fc1(x))
        return self.fc2(x)


class GatedMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.cfg = config
        self.fc1 = nn.Linear(self.cfg.d_model, self.cfg.d_model * self.cfg.mlp_r)
        self.fc2 = nn.Linear(self.cfg.d_model, self.cfg.d_model * self.cfg.mlp_r)
        self.fc3 = nn.Linear(self.cfg.d_model * self.cfg.mlp_r, self.cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        y = F.silu(self.fc1(x))
        return self.fc3(self.fc2(x) * y)


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.cfg = config

        self.attn = nn.MultiheadAttention(
            embed_dim=self.cfg.d_model,
            num_heads=self.cfg.n_heads,
            bias=False,
            batch_first=True,
            dtype=self.cfg.dtype
        )
        if self.cfg.ln_type != "off":
            self.ln1 = LN_FNS[self.cfg.ln_fn](self.cfg.d_model, dtype=self.cfg.dtype, eps=0.0)

        if self.cfg.mlp_type == "ff":
            self.mlp = MLP(config)
        elif self.cfg.mlp_type == "glu":
            self.mlp = GatedMLP(config)

        if self.cfg.ln_type != "off":
            self.ln2 = LN_FNS[self.cfg.ln_fn](self.cfg.d_model, dtype=self.cfg.dtype, eps=0.0)

        if self.cfg.ln_type == "peri":
            self.ln3 = LN_FNS[self.cfg.ln_fn](self.cfg.d_model, dtype=self.cfg.dtype)
            self.ln4 = LN_FNS[self.cfg.ln_fn](self.cfg.d_model, dtype=self.cfg.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.cfg.ln_type == "pre":
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=True)[0]
            x = x + self.mlp(self.ln2(x))
            return x

        elif self.cfg.ln_type == "attn-pre":
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=True)[0]
            x = x + self.mlp(x)
            return x

        elif self.cfg.ln_type == "attn-pre-qk":
            x = x + self.attn(self.ln1(x), self.ln1(x), x, need_weights=True)[0]
            x = x + self.mlp(x)
            return x
        
        elif self.cfg.ln_type == "attn-pre-v":
            x = x + self.attn(x, x, self.ln1(x), need_weights=True)[0]
            x = x + self.mlp(x)
            return x

        elif self.cfg.ln_type == "pre-v":
            x = x + self.attn(x, x, self.ln1(x), need_weights=True)[0]
            x = x + self.mlp(self.ln2(x))
            return x

        elif self.cfg.ln_type == "mlp-pre":
            x = x + self.attn(x, x, x, need_weights=True)[0]
            x = x + self.mlp(self.ln2(x))
            return x
        
        elif self.cfg.ln_type == "mlp-inpnorm":
            x = x + self.attn(x, x, x, need_weights=True)[0]
            x = x + self.mlp(F.normalize(x, dim=-1))
            return x

        elif self.cfg.ln_type == "post":
            x = x + self.attn(x, x, x, need_weights=True)[0]
            x = self.ln1(x)
            x = x + self.mlp(x)
            x = self.ln2(x)
            return x

        elif self.cfg.ln_type == "peri":
            x = x + self.ln2(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=True)[0])
            x = x + self.ln4(self.mlp(self.ln3(x)))
            return x

        else:
            assert self.cfg.ln_type == "off", "If you don't want LayerNorm in the model, set `cfg.ln_type == 'off'`."
            x = x + self.attn(x, x, x, need_weights=True)[0]
            x = x + self.mlp(x)
            return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.cfg = config

        self.embed = nn.Embedding(self.cfg.P+1, self.cfg.d_model, dtype=self.cfg.dtype)
        self.pos_emb = nn.Parameter(
            torch.empty(self.cfg.seq_len, self.cfg.d_model).normal_(mean=0.0, std=0.02)
        )

        self.layers = nn.ModuleList([
            Block(config) for _ in range(self.cfg.n_layers)
        ])
        if self.cfg.final_ln != "off":
            self.final_ln = LN_FNS[self.cfg.ln_fn](self.cfg.d_model, dtype=self.cfg.dtype, eps=1e-5)
            
            if self.cfg.final_ln == "center_only":
                self.final_ln.weight.data = torch.ones_like(self.final_ln.weight.data)
                self.final_ln.weight.requires_grad = False
                
                if self.cfg.ln_fn == "og":
                    self.final_ln.bias.data = torch.zeros_like(self.final_ln.bias.data)
                    self.final_ln.bias.requires_grad = False
        
        assert self.cfg.unembed in ["tied", "separate"], "Un-embed can either be weight-tied with the embedding layer or learned as a separate projection"
        if self.cfg.unembed != "tied":
            self.unembed = Unembed(self.cfg.d_model, self.cfg.P+1)
        
        self.ln_params_frozen = False
    
    def remove_ln_params_from_grad(self) -> None:
        for i in range(self.cfg.n_layers):
            for j in range(1, 5):
                if hasattr(self.layers[i], f"ln{j}"):
                    getattr(self.layers[i], f"ln{j}").weight.requires_grad = False
                    if self.cfg.ln_fn == "og":
                        getattr(self.layers[i], f"ln{j}").bias.requires_grad = False
        
        if hasattr(self, "final_ln"):
            self.final_ln.weight.requires_grad = False
            if self.cfg.ln_fn == "og":
                self.final_ln.bias.requires_grad = False
        
        self.ln_params_frozen = True

    def forward(self, x: Tensor, output_prelogits=False) -> Tensor:
        x = self.embed(x)
        x = x + self.pos_emb

        for layer in self.layers:
            x = layer(x)

        if self.cfg.final_ln != "off":
            x = self.final_ln(x)
        
        if self.cfg.unembed == "tied":
            logits = x[:, -1, :] @ self.embed.weight.T
        else:
            logits = self.unembed(x[:, -1, :])

        if not self.cfg.use_softmax_temp:
            if not output_prelogits:
                return self.cfg.output_scaling * logits
            else:
                return self.cfg.output_scaling * logits, x[:, -1, :]
        else:
            if not output_prelogits:
                return logits
            else:
                return logits, x[:, -1, :]
