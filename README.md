<div align="center">

# Explaining Grokking in Transformers through the Lens of Inductive Bias

<a href="https://arxiv.org/abs/2602.06702v1"><img src="https://img.shields.io/badge/📄%20Paper-arXiv-b31b1b.svg?style=for-the-badge" alt="Paper"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%202.0-green.svg?style=for-the-badge" alt="License"></a>

Jaisidh Singh, Diganta Misra, Antonio Orvieto
</div>


## Abstract

> We investigate grokking in transformers through the lens of inductive bias: dispositions arising from architecture or optimization that let the network prefer one solution over another. We first show that architectural choices such as the position of Layer Normalization (LN) strongly modulates grokking speed. This modulation is explained by isolating how LN on specific pathways shapes shortcut-learning and attention entropy. Subsequently, we study how different optimization settings modulate grokking, inducing distinct interpretations of previously proposed controls such as readout scale. Particularly, we find that using readout scale as a control for lazy training can be confounded by learning rate and weight decay in our setting. Accordingly, we show that features evolve continuously throughout training, suggesting grokking in transformers can be more nuanced than a lazy-to-rich transition of the learning regime. Finally, we show how generalization predictably emerges with feature compressibility in grokking, across different modulators of inductive bias.

<img src="./assets/overview.jpg"><br>

## Reproducing results

<img src="./assets/ibs.jpg"><br>

This repository releases our code to explanably modulate grokking by varying the position of layer normalization (LN) in transformers learning modular addition. Experiments given in the submission can be reproduced by running the files in the `scripts` folder. We use `wandb` to log all metrics across different seeds. Subsequently, these metrics are aggregated across seeds on the `wandb` console, from where each metric presented in the experiments is downloaded in `csv` format. Note that experiments measuring attention entropy are not included in these scripts. To measure attention score entropy, simply save checkpoints and then measure the entropy in the forward pass as follows:

```python
@torch.no_grad()
def measure(cfg, model, tokens):
    x = model.embed(tokens.to(cfg.device))
    x = x + model.pos_emb
    
    for layer in model.layers:
        if cfg.ln_type == "off" or cfg.ln_type == "mlp-pre":
            a, w = layer.attn(x, x, x, need_weights=True)
        elif cfg.ln_type == "attn-pre" or cfg.ln_type == "pre":
            a, w = layer.attn(layer.ln1(x), layer.ln1(x), layer.ln1(x), need_weights=True)
        elif cfg.ln_type == "attn-pre-qk":
            a, w = layer.attn(layer.ln1(x), layer.ln1(x), x, need_weights=True)
        elif cfg.ln_type == "attn-pre-v" or cfg.ln_type == "pre-v":
            a, w = layer.attn(x, x, layer.ln1(x), need_weights=True)
    
    X = a[:, -1, :] @ a[:, -1, :].T
    attn = w.clamp(cfg.eps)
    entropy = -1 * (attn * attn.log()).sum(dim=-1).mean()
    return entropy
```

Note that we use different specifiers in the code for the LN configurations given in our <a href="https://arxiv.org/abs/2602.06702v1">preprint</a>. The code specifiers are provided to the `ln_type` subargument of the `model` subconfig. These are defined as follows.

1. **No LN**: `off`
2. **A**$^*$: `attn-pre`
3. **M+A**$^*$: `pre`
4. **M**: `mlp-pre`
5. **A**$^{qk}$: `attn-pre-qk`
6. **A**$^{v}$: `attn-pre-v`
