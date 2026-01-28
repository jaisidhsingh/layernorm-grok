import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_LOG_COMPILES'] = '0'

import logging
import warnings
from typing import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

warnings.simplefilter("ignore")
logging.getLogger('jax').setLevel(logging.ERROR)


def make_gif(frames, out_path, fps=5):
    """
    frames: list of (timestep_id, image_array)
    """
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0][1], aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
    title = ax.set_title(f"t = {frames[0][0]}")
    ax.axis("off")

    def update(i):
        tid, img = frames[i]
        # im = ax.imshow(frames[0][1], aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
        im.set_data(img)
        vmin, vmax = img.min(), img.max()
        im.set_clim(vmin=vmin, vmax=vmax)
        title.set_text(f"t = {tid}")
        return im, title

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 // fps,
        blit=False
    )

    ani.save(
        out_path,
        writer=PillowWriter(fps=fps)
    )
    plt.close(fig)


def remove_spikes_from_loss(loss_list):
    out = []
    prev = None
    for item in loss_list:
        if prev is not None:
            if item - prev > 0.5:
                out.append(prev)
            else:
                out.append(item)
        else:
            out.append(item)
        prev = item

    return out


def remove_spikes_from_acc(acc_list):
    out = []
    prev = None
    for item in acc_list:
        if prev is not None:
            if prev - item > 0.1:
                out.append(prev)
            else:
                out.append(item)
        else:
            out.append(item)
        prev = item
    return out


def make_plots(cfg, logs: dict, save_path: str):
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("Set2"))

    _, axes = plt.subplots(1, 2, figsize=(10, 6))
    x = list(logs["train"].keys())

    for i, metric in enumerate(["loss", "accuracy"]):
        train = [v[metric] for v in logs["train"].values()]
        test = [v[metric] for v in logs["test"].values()]

        test_emb_n = None
        test_emb_w = None

        axes[i].plot(x, train, label="train")
        axes[i].plot(x, test, label="test")

        axes[i].set_xlabel("steps")
        # axes[i].set_xticks([0, 500, 1000, 1500, 2000])
        axes[i].set_ylabel(metric)

        if metric == "loss":
            axes[i].set_ylim([-0.2, 15])
            pass
        else:
            axes[i].set_ylim([-0.1, 1.1])

        axes[i].legend()
        axes[i].set_title(f"{metric} curves")

    plt.suptitle(cfg.name)
    plt.savefig(save_path, dpi=300)
