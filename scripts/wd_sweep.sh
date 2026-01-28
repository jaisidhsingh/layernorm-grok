#!/bin/bash
conda activate pt
echo "Activated my environment"

nvidia-smi

cd /full_path_to/submission_3168

# you can use any number and type of seeds you like. we just liked these.
seeds="456 789 111 222 333"
wds="0.1 0.2 0.5 0.75 1.0 1.5"
for seed in $seeds; do
    for wd in $wds; do
        python3 -m experiments.train_transformer \
            --exp.name trf-1l_lecun_none_wd-${wd}_seed-$seed \
            --exp.project your-wandb-project \
            --exp.seed $seed \
            --data.seed $seed \
            --model.seed $seed \
            --exp.fourier_on \
            --exp.fourier_every 200 \
            --exp.ntk_on \
            --exp.ntk_every 200 \
            --exp.alpha_req_on \
            --exp.alpha_req_every \
            --exp.use_wandb \
            --exp.n_train_steps 10000 \
            --model.ln_type off \
            --model.final_ln off \
            --model.kernel_init lecun \
            --model.output_scaling 1.0 \
            --opt.weight_decay $wd \
            --opt.eps 1e-8 ;
    done
done