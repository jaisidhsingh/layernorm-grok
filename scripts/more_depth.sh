# #!/bin/bash
conda activate pt
echo "Activated my environment"

nvidia-smi

cd /full_path_to/submission_3168

# you can use any number and type of seeds you like. we just liked these.
seeds="456 789 111 222 333"
layernorms="off attn-pre pre mlp-pre"
for seed in $seeds; do
    for ln in $layernorms; do
        python3 -m experiments.train_transformer \
            --exp.name trf-3l_lecun_${ln}_seed-$seed \
            --exp.project your-wandb-project \
            --exp.seed $seed \
            --data.seed $seed \
            --model.seed $seed \
            --exp.fourier_on \
            --exp.fourier_every 200 \
            --exp.ntk_on \
            --exp.ntk_every 200 \
            --exp.alpha_req_on \
            --exp.alpha_req_every 200 \
            --exp.use_wandb \
            --exp.n_train_steps 10000 \
            --model.d_model 64 \
            --model.n_layers 3 \
            --model.ln_type $ln \
            --model.final_ln off \
            --model.kernel_init lecun \
            --model.output_scaling 1.0 \
            --opt.eps 1e-8 ;
    done
done