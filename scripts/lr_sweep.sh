#!/bin/bash
conda activate pt
echo "Activated my environment"

nvidia-smi

cd /full_path_to/submission_3168

# you can use any number and type of seeds you like. we just liked these.
seeds="456 789 111 222 333"
rates="1e-3 3e-3 5e-3 1e-2"
for seed in $seeds; do
    for lr in $rates; do
        python3 -m experiments.train_transformer \
            --exp.name trf-1l_lecun_none_lr-${lr}_seed-$seed \
            --exp.project your-wandb-project \
            --exp.seed $seed \
            --data.seed $seed \
            --model.seed $seed \
            --exp.fourier_on \
            --exp.fourier_every 200 \
            --exp.ntk_on \
            --exp.ntk_every 200 \
            --exp.use_wandb \
            --exp.fix_wd \
            --exp.n_train_steps 10000 \
            --model.ln_type off \
            --model.final_ln off \
            --model.kernel_init lecun \
            --model.output_scaling 1.0 \
            --opt.learning_rate $lr \
            --opt.eps 1e-8 ;
    done
done