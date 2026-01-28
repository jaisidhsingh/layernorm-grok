#!/bin/bash
conda activate pt
echo "Activated my environment"

nvidia-smi

cd /full_path_to/submission_3168

# you can use any number and type of seeds you like. we just liked these.
seeds="456 789 111 222 333"
for seed in $seeds; do
    python3 -m experiments.train_transformer \
        --exp.name check-mem_trl-1l_none_seed-$seed \
        --exp.project your-wandb-project \
        --exp.seed $seed \
        --data.seed $seed \
        --model.seed $seed \
        --exp.check_mlp_mem \
        --exp.ntk_on \
        --exp.ntk_every 200 \
        --exp.fourier_on \
        --exp.fourier_every 200 \
        --exp.use_wandb \
        --exp.n_train_steps 10000 \
        --model.ln_type off \
        --model.final_ln off \
        --model.kernel_init lecun \
        --model.output_scaling 1.0 \
        --opt.eps 1e-8 ;
done


seeds="456 789 111 222 333"
layernorms="attn-pre pre"
for seed in $seeds; do
    for ln in $layernorms; do
        python3 -m experiments.train_transformer \
            --exp.name check-mem_trl-1l_mlp-pre_seed-$seed \
            --exp.project your-wandb-project \
            --exp.seed $seed \
            --data.seed $seed \
            --model.seed $seed \
            --exp.check_mlp_mem \
            --exp.ntk_on \
            --exp.ntk_every 200 \
            --exp.fourier_on \
            --exp.fourier_every 200 \
            --exp.use_wandb \
            --exp.n_train_steps 10000 \
            --model.ln_type $ln \
            --model.final_ln off \
            --model.kernel_init lecun \
            --model.output_scaling 1.0 \
            --opt.eps 1e-8 ;
    done
done
