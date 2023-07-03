#!/bin/bash

# Setup arrays
envs=(tworoom fourroom)
agents=(psrl ucrl2 klucrl)
seeds=(0)
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"


for env in "${envs[@]}"
do
    for agent in "${agents[@]}"
    do
        for seed in "${seeds[@]}"
        do
            screen -dmS train_${env}_${agent}_${seed} python3 scripts/train_transformer.py \
                --seed $seed \
                --config configs/${env}_${agent}.yaml
        done
    done
done