# Explaining Grokking in Transformers through the Lens of Inductive Bias

> Jaisidh Singh, Diganta Misra, Antonio Orvieto

This repository releases our code to explanably modulate grokking by varying the position of layer normalization (LN) in transformers learning modular addition. Experiments given in the submission can be reproduced by running the files in the `scripts` folder. We use `wandb` to log all metrics across different seeds. Subsequently, these metrics are aggregated across seeds on the `wandb` console, from where each metric presented in the experiments is downloaded in `csv` format.

Note that we use different specifiers in the code for the LN configurations given in our preprint (to be released). The code specifiers are provided to the `ln_type` subargument of the `model` subconfig. These are defined as follows.

1. **No LN**: `off`
2. **A**$^*$: `attn-pre`
3. **M+A**$^*$: `pre`
4. **M**: `mlp-pre`
5. **A**$^{qk}$: `attn-pre-qk
6. **A**$^{v}$: `attn-pre-v
