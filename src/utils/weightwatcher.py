import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
import pandas as pd
from typing import *
from powerlaw import Fit
from copy import deepcopy

from src.models import Transformer


class TorchWeightWatcher():
    def __init__(self, model: Transformer, details_format: str = "df"):
        assert details_format in ["df", "dict"], "Unsupported `details_format` provided."
        self.details_format = details_format
        
        self.supported_layers = ["linear", "nondynamicallyquantizablelinear"]
        self.device = self.get_device()
        
        self.model = deepcopy(model)
        self.model = self.model.to(self.device)
        
        if hasattr(model.layers[0].attn, "in_proj_weight"):
            q, k, v = model.layers[0].attn.in_proj_weight.data.clone().chunk(3)
            
            self.model.add_module("attn_q", nn.Linear(model.cfg.d_model, model.cfg.d_model))
            self.model.attn_q.weight.data = q
            assert torch.equal(self.model.attn_q.weight.data, q) 
            
            self.model.add_module("attn_k", nn.Linear(model.cfg.d_model, model.cfg.d_model))
            self.model.attn_k.weight.data = k 
            assert torch.equal(self.model.attn_k.weight.data, k) 

            self.model.add_module("attn_v", nn.Linear(model.cfg.d_model, model.cfg.d_model))
            self.model.attn_v.weight.data = v 
            assert torch.equal(self.model.attn_v.weight.data, v) 
    
    
    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        return "cpu"


    def is_supported(self, layer: nn.Module) -> bool:
        layer_class_name = type(layer).__name__.lower()
        if layer_class_name in self.supported_layers:
            return True
        return False


    @torch.no_grad()
    def compute_esd(self, X: Tensor) -> Tensor:
        n, m = X.shape
        if n != m:
            eigenvalues = torch.linalg.svdvals(X)
        else:
            eigvals = torch.linalg.eigvalsh(X @ X.T)
            eigenvalues = torch.sqrt(torch.abs(eigvals))
        
        return torch.sort(eigenvalues, descending=True).values
   
    
    @torch.no_grad()
    def fit_powerlaw_for_matrix(self, matrix: Tensor) -> List:
        matrix = matrix.to(self.device)
        corr = matrix.T @ matrix
        evs = self.compute_esd(corr)
        eigenvalues_np = evs.cpu().numpy()

        fit = Fit(eigenvalues_np, discrete=False, verbose=False)
        eigenvalues_used = eigenvalues_np[eigenvalues_np >= fit.xmin]
        
        if fit.xmax:
            eigenvalues_used = eigenvalues_used[eigenvalues_used <= fit.xmax]
        
        alpha = getattr(fit, 'alpha')
        return [alpha, eigenvalues_used, eigenvalues_np]
    

    def analyze(self) -> Union[Dict, pd.DataFrame]:
        data = {
            "layer_index": [], 
            "layer_name": [], 
            "weight_shape": [], 
            "alpha": [], 
            "num_eigenvals_fit": [],
            "num_eigenvals": [],
            "stable_ranks": [],
            "effective_ranks": [],
            "ranks": []
        }

        for idx, (name, module) in enumerate(self.model.named_modules()):
            if self.is_supported(module): 
                data["layer_index"].append(idx)
                data["layer_name"].append("model."+name)
                
                weight = module.weight.data
                weight = weight.to(self.device)

                if len(weight.shape) > 2:
                    N = weight.shape[0]
                    weight = weight.view((N, -1))
                
                weight_shape = ",".join([str(item) for item in weight.shape])
                data["weight_shape"].append(weight_shape)
                data["ranks"].append(torch.linalg.matrix_rank(weight).item())

                stable_rank = torch.linalg.norm(weight, ord='fro') / torch.linalg.norm(weight, ord=2)
                data["stable_ranks"].append(stable_rank.item())

                [alpha, evals_eff, evals] = self.fit_powerlaw_for_matrix(weight)
                data["alpha"].append(alpha.item())
                data["num_eigenvals_fit"].append(len(evals_eff))
                data["num_eigenvals"].append(len(evals))

                ev = torch.from_numpy(evals).to(self.device)
                p = ev / torch.linalg.norm(ev, ord=1)
                entropy = -1 * (p * torch.log(p)).sum()
                effective_rank = torch.exp(entropy)
                data["effective_ranks"].append(effective_rank.item())

            
        if self.details_format == "df":
            data = pd.DataFrame.from_dict(data)
        return data


@torch.no_grad()
def compute_esd(X: Tensor) -> Tensor:
    n, m = X.shape
    if n != m:
        eigenvalues = torch.linalg.svdvals(X)
    else:
        eigvals = torch.linalg.eigvalsh(X @ X.T)
        eigenvalues = torch.sqrt(torch.abs(eigvals))
    
    return torch.sort(eigenvalues, descending=True).values

@torch.no_grad()
def esd_powerlaw(X):
    evs = compute_esd(X)
    eigenvalues_np = evs.cpu().numpy()

    fit = Fit(eigenvalues_np, discrete=False, verbose=False)
    eigenvalues_used = eigenvalues_np[eigenvalues_np >= fit.xmin]
    
    if fit.xmax:
        eigenvalues_used = eigenvalues_used[eigenvalues_used <= fit.xmax]
    
    alpha = getattr(fit, 'alpha')
    return [alpha, eigenvalues_used, eigenvalues_np]

@torch.no_grad()
def compressibility_stats(x, which=""):
    X = x.T @ x
    
    out = {}
    out["rank"+which] = torch.linalg.matrix_rank(X).item()
    out["stable_rank"+which] = (torch.linalg.norm(X, ord="fro") / torch.linalg.norm(X, ord=2)).item()
    
    [alpha, evals_used, evals] = esd_powerlaw(X)
    out["alpha"+which] = alpha.item()
     
    ev = torch.from_numpy(evals).to(x.device)
    p = ev / torch.linalg.norm(ev, ord=1)
    entropy = -1 * (p * torch.log(p)).sum()
    out["effective_rank"+which] = torch.exp(entropy).item()
    
    return out
