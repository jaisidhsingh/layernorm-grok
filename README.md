# Explaining Grokking in Transformers through the Lens of Inductive Bias

> Jaisidh Singh, Diganta Misra, Antonio Orvieto

This repository releases our code to explanably modulate grokking by varying the position of layer normalization (LN) in transformers learning modular addition. Experiments given in the submission can be reproduced by running the files in the `scripts` folder. We use `wandb` to log all metrics across different seeds. Subsequently, these metrics are aggregated across seeds on the `wandb` console, from where each metric presented in the experiments is downloaded in `csv` format. Note that experiments measuring attention entropy are not included in these scripts. To measure attention score entropy, one simply saves checkpoints and then measures the entropy in the forward pass as follows:

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

Note that we use different specifiers in the code for the LN configurations given in our preprint (to be released). The code specifiers are provided to the `ln_type` subargument of the `model` subconfig. These are defined as follows.

1. **No LN**: `off`
2. **A**$^*$: `attn-pre`
3. **M+A**$^*$: `pre`
4. **M**: `mlp-pre`
5. **A**$^{qk}$: `attn-pre-qk
6. **A**$^{v}$: `attn-pre-v
