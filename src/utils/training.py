import warnings
warnings.simplefilter("ignore")

import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from hessian_eigenthings import compute_hessian_eigenthings

from typing import *
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    b1: float = 0.9
    b2: float = 0.98
    hessian_method: str = "power_iter"
    hessian_k: int = 3
    eps: float = 1e-8


def get_optimizer_fn(cfg: OptimizerConfig, parameters):
    ref_cfg = {
        "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "betas": (cfg.b1, cfg.b2),
        "eps": cfg.eps
    }
    return torch.optim.AdamW(params=parameters, **ref_cfg)


def compute_effective_rank(x):
    X = x.T @ x
    n, m = X.shape
    if n != m:
        eigenvalues = torch.linalg.svdvals(X)
    else:
        eigvals = torch.linalg.eigvalsh(X @ X.T)
        eigenvalues = torch.sqrt(torch.abs(eigvals))
    
    evals = torch.sort(eigenvalues, descending=True).values
    evals = evals.to(x.device)
    p = evals / torch.linalg.norm(evals, ord=1)
    entropy = -1 * (p * torch.log(p)).sum()
    return entropy.exp()


def loss_fn(
    model: nn.Module, tokens: Tensor, labels: Tensor, 
    eff_rank_loss: bool = False, eff_rank_loss_strength: float = 1e-3,
    subtract_init_pred: bool = False, init_pred: Tensor = None,
    device: str = "cuda"
) -> Tuple[Any, Any]:
    out = model(tokens, output_prelogits=eff_rank_loss)
    if not eff_rank_loss:
        logits = out
        other_loss = 0.0
    else:
        logits, prelogits = out
        other_loss = eff_rank_loss_strength * compute_effective_rank(prelogits)
        
    # loss = F.cross_entropy(logits, labels)
    if device == "cuda":
        logits = logits.to(torch.float64)
    
    if model.cfg.use_softmax_temp:
        temp = 1 / model.cfg.output_scaling
    else:
        temp = 1
    
    if subtract_init_pred and init_pred is not None:
        logits = logits - init_pred
    
    log_probs = (logits / temp).log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    loss = -correct_log_probs.mean() + other_loss
    preds = log_probs.argmax(dim=-1)
    
    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).sum() / labels.shape[0]
    return loss, accuracy


@torch.no_grad()
def get_weight_norm(model: nn.Module):
    t = sum((p**2).sum() for p in model.parameters() if p is not None)
    return math.sqrt(t)


def inspect_gradients(model: nn.Module):
    t = sum((p.grad.data ** 2).sum() for p in model.parameters() if p.grad is not None)
    return math.sqrt(t) 


def measure_update(timing: str, model: nn.Module, updates: Dict):
    update_norm = None
    if timing == "before":
        assert len(updates) == 0, "Dict to store params must be empty before `optimizer.step()`"
        for n, p in model.named_parameters():
            if p is not None:
                before = p.data.clone()
                updates[n] = -1 * before
    
    else: # timing == "after" here
        assert len(updates) != 0, "Dict collect parameters before `optimizer.step()`"
        for n, p in model.named_parameters():
            if p is not None:
                after = p.data.clone()
                updates[n] += after
                updates[n] = torch.sum(updates[n]**2)
    
        total = 0
        for v in updates.values():
            total += v.item()
        update_norm = math.sqrt(total)
    
    return updates, update_norm
        

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, tokens: Tensor, targets: Tensor, device: str = "cuda", **kwargs):
    model.train()
    optimizer.zero_grad()
    
    grad_info = {}
    grad_info["weight_norm"] = get_weight_norm(model)
    
    (loss, accuracy) = loss_fn(model, tokens, targets, device=device, **kwargs)
    loss_val = torch.tensor(loss.item())
    loss.backward()
    
    updates, _ = measure_update("before", model, {})
    grad_info["grad_norm"] = inspect_gradients(model)
    
    optimizer.step()
    updates, update_norm = measure_update("after", model, updates)
    grad_info["update_norm"] = update_norm
    return {"loss": loss_val, "accuracy": accuracy, "neg_log_loss": -torch.log(loss_val)}, grad_info


@torch.no_grad()
def spec_step(model: nn.Module, tokens: Tensor, targets: Tensor, device: str = "cuda", **kwargs):
    model.eval()
    (loss, accuracy) = loss_fn(model, tokens, targets, device=device, **kwargs)
    return {"spec_loss": loss, "spec_accuracy": accuracy, "spec_neg_log_loss": -torch.log(loss)}


@torch.no_grad()
def test_step(model: nn.Module, tokens: Tensor, targets: Tensor, device: str = "cuda", **kwargs):
    model.eval()
    (loss, accuracy) = loss_fn(model, tokens, targets, device=device, **kwargs)
    return {"loss": loss, "accuracy": accuracy, "neg_log_loss": -torch.log(loss)}


def inspect_hessian(model: nn.Module, dataset: torch.utils.data.Dataset, method: str = "lanczos" , k: int = 3) -> Dict:
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*dataset),
        batch_size=dataset[0].shape[0],
        shuffle=False
    )
    eigenvals, _ = compute_hessian_eigenthings(
        model, loader, F.cross_entropy, k, mode=method
    )
    if k > 1:
        return {
            "max_eigenval": eigenvals.max(),
            "mean_eigenval": eigenvals.mean(),
            "min_eigenval": eigenvals.min(),
        }
    else:
        return {
            "max_eigenval": eigenvals.max()
        }

 
def compute_ntk(model, x):
    N = x.shape[0]
    params = dict(model.named_parameters())
    
    def fnet_single(params, x_single):
        out = functional_call(model, params, (x_single.unsqueeze(0),))
        return out.squeeze(0)
    
    def compute_grad(x_single):
        grads = grad(lambda p: fnet_single(p, x_single).sum())(params)
        return torch.cat([g.flatten() for g in grads.values()])
    
    grads_all = vmap(compute_grad)(x)
    ntk = grads_all @ grads_all.T
    return ntk


def fnet_single(model, params, x_single):
	out = functional_call(model, params, (x_single.unsqueeze(0),))
	return out.squeeze(0)


def compute_ntk_full(model, x):
    N = x.shape[0]
    params = dict(model.named_parameters())
    out_dim = model.cfg.P+1
    full_ntk = torch.zeros((N, N)).float().to(x.device)

    for i in range(out_dim):
        def compute_grad(x_single):
            grads = grad(lambda p: fnet_single(model, p, x_single)[i].sum())(params)
            return torch.cat([g.flatten() for g in grads.values()])

        grads_all = vmap(compute_grad)(x)
        ntk = grads_all @ grads_all.T
        full_ntk += ntk

    return full_ntk

