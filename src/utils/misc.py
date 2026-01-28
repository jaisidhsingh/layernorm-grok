import os
import json
import torch
import random
import warnings
import numpy as np
from typing import *
from dataclasses import asdict
from src.utils.plotting import make_plots
warnings.simplefilter("ignore")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"


def save_experiment_results(cfg, metric_logger):
    os.makedirs(os.path.join(cfg.exp.results_folder, cfg.exp.name), exist_ok=True)
    logs_save_path = os.path.join(cfg.exp.results_folder, cfg.exp.name, f"{cfg.exp.name}_logs.json")
    metric_logger.save_data(logs_save_path)

    config_save_path = os.path.join(cfg.exp.results_folder, cfg.exp.name, f"{cfg.exp.name}_config.json")
    config_to_save = asdict(cfg)

    plot_save_path = None
    if cfg.exp.save_plots:
        plot_save_path = os.path.join(cfg.exp.results_folder, cfg.exp.name, f"{cfg.exp.name}_plots.png")
        make_plots(cfg.exp, metric_logger.data, plot_save_path)

    for k in ["exp", "model"]:
        tmp = config_to_save[k]["dtype"]
        config_to_save[k]["dtype"] = str(tmp)

    with open(config_save_path, "w") as f:
        json.dump(config_to_save, f)

    print("results saved at:")
    print(logs_save_path)
    print(config_save_path)

    if cfg.exp.save_plots:
        print(plot_save_path)

    print("\n")


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class MetricLogger():
    def __init__(self, cfg):
        metrics = ["train", "test", "grads"]
        if cfg.htsr_on:
            metrics.append("htsr")
        if cfg.hessian_on:
            metrics.append("hessian")
        if cfg.ntk_on:
            metrics.append("ntk")
        if cfg.fourier_on:
            metrics.append("fourier")
        if cfg.alpha_req_on:
            metrics.append("alpha_req")
            
        self.metrics = metrics
        self.data = {m: {} for m in metrics}
        self.step_number: int = 0

    def step(self, logs, step=None):
        self.step_number += 1
        if step is None:
            step = self.step

        for m in logs.keys():
            assert m in self.metrics, "Unsupported metric provided"
            self.data[m][step] = {k: round(float(v), 4) for k, v in logs[m].items()}

    def save_data(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.data, f)
