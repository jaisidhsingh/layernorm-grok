#!/bin/bash
conda activate pt
echo "Activated my environment"

nvidia-smi

cd /full_path_to/submission_3168

# you can use any number and type of seeds you like. we just liked these.
seeds="456 789 111 222 333"
scalings="0.5 0.75 1.0 1.25 1.5"
for seed in $seeds; do
    for os in $scalings; do
        python3 -m experiments.train_transformer \
            --exp.name trf-1l_lecun_none_os-${os}_seed-$seed \
            --exp.project your-wandb-project \
            --exp.seed $seed \
            --data.seed $seed \
            --model.seed $seed \
            --exp.fourier_on \
            --exp.fourier_every 200 \
            --exp.ntk_on \
            --exp.ntk_every 200 \
            --exp.use_wandb \
            --exp.n_train_steps 10000 \
            --model.ln_type off \
            --model.final_ln off \
            --model.kernel_init lecun \
            --model.output_scaling $os \
            --opt.eps 1e-8 ;
    done
done

# lecun init is sensitive to eps=0.0 and causes NaNs.
# kaiming init has very similar init parameter norm but
# does not lead to NaNs with eps=0.0.
# All results of learning rate, weight decay, layernorm position, etc
# hold with kaiming initialization as well.
for seed in $seeds; do
    for os in $scalings; do
        python3 -m experiments.train_transformer \
            --exp.name eps0_trf-1l_kaiming_none_os-${os}_seed-$seed \
            --exp.project your-wandb-project \
            --exp.seed $seed \
            --data.seed $seed \
            --model.seed $seed \
            --exp.fourier_on \
            --exp.fourier_every 200 \
            --exp.ntk_on \
            --exp.ntk_every 200 \
            --exp.use_wandb \
            --exp.n_train_steps 10000 \
            --model.ln_type off \
            --model.final_ln off \
            --model.kernel_init kaiming \
            --model.output_scaling $os \
            --opt.eps 0.0 ;
    done
done


# Subtracting the output of the network at initialization
# from the trainable network (both scaled by the variable readout scale)
# yields very similar results. It is not enough to subtract the initialized
# output to make the network truly lazy. Shown for eps=0.0 below.
seeds="456 789 111 222 333"
scalings="0.5 0.75 1.0 1.25 1.5"
for seed in $seeds; do
    for os in $scalings; do
        python3 -m experiments.train_transformer \
            --exp.name trf-1l_lecun_none_os-${os}_seed-$seed \
            --exp.project your-wandb-project \
            --exp.seed $seed \
            --data.seed $seed \
            --model.seed $seed \
            --exp.fourier_on \
            --exp.fourier_every 200 \
            --exp.ntk_on \
            --exp.ntk_every 200 \
            --exp.use_wandb \
            --exp.subtract_init_pred \
            --exp.n_train_steps 10000 \
            --model.ln_type off \
            --model.final_ln off \
            --model.kernel_init kaiming \
            --model.output_scaling $os \
            --opt.eps 0.0 ;
    done
done