#!/bin/bash

envs=(riverswim tworoom fourroom)
agents=(psrl ucrl2 klucrl)
seeds=(0 1 2)

for env in "${envs[@]}"
do
    for agent in "${agents[@]}"
    do
        for seed in "${seeds[@]}"
        do
            screen -dmS regret_${env}_${agent}_${seed} python3 scripts/generate_data.py \
                --seed $seed \
                --goal-reward 0 \
                --config configs/${env}_${agent}.yaml
        done
    done
done