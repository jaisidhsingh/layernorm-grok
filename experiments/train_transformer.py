import os
import warnings
warnings.simplefilter("ignore")

import tyro
import torch
import wandb
import pickle
from typing import *
from tqdm import tqdm
from pprint import pprint
from statistics import mean
from dataclasses import dataclass, asdict

from src.utils import *
from src.data import MathGameDataset, MathGameConfig
from src.models import *


@dataclass
class ExperimentConfig:
    seed: int = 0
    name: str = "test"
    project: str = "your-wandb-project"
    results_folder: str = "/somewhere/results"
    
    htsr_on: bool = False
    htsr_every: int = 200
    
    hessian_on: bool = False
    hessian_every: int = 200
    
    alpha_req_on: bool = False
    alpha_req_every: int = 200
    
    ntk_on: bool = False
    use_full_ntk: bool = False
    ntk_every: int = 200
    
    wrong_lr: bool = False
    fix_wd: bool = False

    check_mlp_mem: bool = False
    subtract_init_pred: bool = False
    
    eff_rank_loss: bool = False
    eff_rank_loss_strength: float = 1e-3
    eff_rank_entry: int = 1000
    
    fourier_on: bool = False
    fourier_every: int = 100
    
    save_ckpt: bool = False
    save_every: int = 100
    save_folder: str = "/somewhere/checkpoints"

    use_wandb: bool = False
    wandb_mode: str = "online"
    wandb_logs_folder: str = "/somewhere/wandb_logs"

    n_train_steps: int = 8000
    batch_size: int = -1
    save_plots: bool = False
    dtype: Any = torch.float32


@dataclass
class Config:
    exp: ExperimentConfig
    model: TransformerConfig
    opt: OptimizerConfig
    data: MathGameConfig


def col_energy(X):
    col_means = X.mean(axis=0, keepdims=True)
    between = np.var(col_means)
    within = np.mean((X - col_means)**2)
    return between / (within + 1e-8)


def main(cfg: Config):
    seed_everything(cfg.exp.seed)
    cfg.exp.dtype = cfg.model.dtype
    device = get_device()
    
    if cfg.exp.wrong_lr:
        tmp = cfg.opt.learning_rate
        cfg.opt.learning_rate = tmp / (cfg.model.output_scaling**2)
    
    if cfg.exp.fix_wd:
        tmp = cfg.opt.weight_decay
        cfg.opt.weight_decay = (1e-3 * tmp) / cfg.opt.learning_rate
    
    if cfg.exp.check_mlp_mem:
        tmp = deepcopy(cfg.model.ln_type)
    
    os.makedirs(cfg.exp.save_folder, exist_ok=True)
    ckpt_folder = os.path.join(cfg.exp.save_folder, cfg.exp.name)
    os.makedirs(ckpt_folder, exist_ok=True)

    print("starting experiment with name:", cfg.exp.name)
    print("=="*50)
    pprint(cfg)
    print("=="*50)

    math_game = MathGameDataset(cfg.data)
    train_dataset, test_dataset = math_game.create_problem_set()

    tokens_train, targets_train = train_dataset
    tokens_test, targets_test = test_dataset

    tokens_train, targets_train = tokens_train.to(device), targets_train.to(device)
    tokens_test, targets_test = tokens_test.to(device), targets_test.to(device)

    model = Transformer(cfg.model).to(device)
    init_weights(model, cfg.model)
    
    train_init_pred, test_init_pred = None, None 
    if cfg.exp.subtract_init_pred:
        model.eval()
        train_init_pred = model(tokens_train).detach()
        test_init_pred = model(tokens_test).detach()
        model.train()

    optimizer = get_optimizer_fn(cfg.opt, model.parameters())

    if cfg.exp.use_wandb:
        selected_mode = cast(Literal["online", "offline", "disabled"], cfg.exp.wandb_mode)
        wandb.init(
            mode=selected_mode, name=cfg.exp.name, project=cfg.exp.project,
            entity="your-wandb-entity", config=asdict(cfg), dir=cfg.exp.wandb_logs_folder
        )
    
    if cfg.exp.ntk_on:
        ntk_fn = compute_ntk if not cfg.exp.use_full_ntk else compute_ntk_full
        ntk_0 = ntk_fn(model, tokens_train)
        ntk_0_F = ntk_0.norm(p="fro")
    
    metric_logger = MetricLogger(cfg.exp)
    bar = tqdm(total=cfg.exp.n_train_steps)
    
    for step in range(1, cfg.exp.n_train_steps+1):
        if cfg.model.use_softmax_temp:
            with torch.no_grad():
                h_train, h_test = model(tokens_train), model(tokens_test)
                top2_train, _ = h_train.topk(k=2, dim=-1)
                marg_train = (top2_train[:, 0] - top2_train[:, 1]).mean()
                top2_test, _ = h_test.topk(k=2, dim=-1)
                marg_test = (top2_test[:, 0] - top2_test[:, 1]).mean()
        
        train_kwargs = {"subtract_init_pred": cfg.exp.subtract_init_pred, "init_pred": train_init_pred}
        test_kwargs = {"subtract_init_pred": cfg.exp.subtract_init_pred, "init_pred": test_init_pred}
        
        # train and test step 
        train_logs, grad_logs = train_step(model, optimizer, tokens_train, targets_train, device=device, **train_kwargs)
        test_logs = test_step(model, tokens_test, targets_test, device=device, **test_kwargs)
        
        if cfg.exp.check_mlp_mem:
            # perturb the mlp input to check norm-based memorization
            model.cfg.ln_type = "mlp-inpnorm"
            for l in range(model.cfg.n_layers):
                model.layers[l].cfg.ln_type = "mlp-inpnorm"
            
            # compute performace metrics under the perturbation
            spec_train_logs = spec_step(model, tokens_train, targets_train, device=device)
            spec_test_logs = spec_step(model, tokens_test, targets_test, device=device)
            train_logs.update(spec_train_logs)
            test_logs.update(spec_test_logs)
            
            # set it back to what was provided
            model.cfg.ln_type = tmp
            for l in range(model.cfg.n_layers):
                model.layers[l].cfg.ln_type = tmp
            assert model.cfg.ln_type == cfg.model.ln_type, "Mismatch in reset" 
        
        if cfg.model.use_softmax_temp:
            train_logs.update({"top2_margin": marg_train})
            test_logs.update({"top2_margin": marg_test})
        
        ## define logs 
        logs = {"train": train_logs, "test": test_logs, "grads": grad_logs}
        
        if cfg.exp.alpha_req_on:
            if step % cfg.exp.alpha_req_every == 0 or step == 1:
                with torch.no_grad():
                    _, prelogits_train = model(tokens_train, output_prelogits=True)
                    train_req = compressibility_stats(prelogits_train, which="_train")
                    _, prelogits_test = model(tokens_test, output_prelogits=True)
                    train_req.update(compressibility_stats(prelogits_test, which="_test"))
                    logs.update({"alpha_req": train_req})

        if cfg.exp.hessian_on:
            if step % cfg.exp.hessian_every == 0 or step == 1:
                hessian_logs = inspect_hessian(
                    model, train_dataset,
                    method=cfg.opt.hessian_method,
                    k=cfg.opt.hessian_k
                )
                logs.update({"hessian": hessian_logs})

        if cfg.exp.htsr_on:
            if step % cfg.exp.htsr_every == 0 or step == 1:
                watcher = TorchWeightWatcher(model, details_format="dict")
                details = watcher.analyze()
                logs["htsr"] = {k : mean(details[k]) for k in ["alpha", "stable_ranks", "effective_ranks", "ranks"]}
        
        if cfg.exp.ntk_on:
            if step % cfg.exp.ntk_every == 0 or step == 1:
                ntk_t = ntk_fn(model, tokens_train)
                delta = torch.norm(ntk_t - ntk_0, p="fro") / ntk_0_F
                logs["ntk"] = {"relative_distance": delta, "ntk_init_fro": ntk_0_F}
        
        if cfg.exp.fourier_on:
            if step % cfg.exp.fourier_every == 0 or step == 1:
                with torch.no_grad():
                    w_e = model.embed.weight.data.clone()[:cfg.model.P, :]
                    w_f = torch.fft.rfft(w_e, dim=0).abs().T.cpu().numpy()
                    logs["fourier"] = {"col_energy": col_energy(w_f)}
        
            
        metric_logger.step(logs, step=step)
        
        if cfg.exp.save_ckpt:
            if step % cfg.exp.save_every == 0 or step == 1:
                pass
                dump = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "logs": metric_logger.data,
                    "config": asdict(cfg)
                }
                torch.save(dump, os.path.join(ckpt_folder, f"step_{step}.pt"))

        if cfg.exp.use_wandb:
            wandb_logs = {}
            for k, v in logs.items():
                for kk in v.keys():
                    wandb_logs[f"{k}/{kk}"] = logs[k][kk]

            wandb.log(wandb_logs, step=step)

        bar.set_postfix({"train_loss": train_logs["loss"].item(), "test_loss": test_logs["loss"].item()})
        bar.update(1)

    bar.close()
    save_experiment_results(cfg, metric_logger)
    print(f"Finished experiment {cfg.exp.name}")
    print("=="*50,"\n")


if __name__ == "__main__":
    defaults = Config(
        exp=ExperimentConfig(),
        data=MathGameConfig(),
        model=TransformerConfig(),
        opt=OptimizerConfig()
    )
    cfg = tyro.cli(Config, default=defaults)
    main(cfg)
