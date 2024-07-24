#!/bin/bash



# Get CLI arguments
script_name=$1



# Validate arguments
if [ -z "$script_name" ]
then
    script_name="generate_data"
fi



# Setup arrays
envs=(riverswim)
# envs=(tworoom fourroom)
agents=(psrl ucrl2 klucrl)
seeds=(0 1 2 3 4 5 6 7 8 9)
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"



# Run scripts
echo "Setup: script_name=$script_name, envs=(${envs[@]}), agents=(${agents[@]}), seeds=(${seeds[@]}), GPUs=(${gpus[@]})"

gpu_i=0

for env in "${envs[@]}"
do
    for agent in "${agents[@]}"
    do
        for seed in "${seeds[@]}"
        do
            echo "Running run_script.sh on GPU='${gpus[$gpu_i]}' with script_name=$script_name, env=$env, agent=$agent, seed=$seed"

            CUDA_VISIBLE_DEVICES="${gpus[$gpu_i]}" . run_script.sh $script_name $env $agent $seed

            # increment i by one and loop back when we reach the end of the array
            ((gpu_i++))
            if [ $gpu_i -ge ${#gpus[@]} ]; then
                gpu_i=0
            fi
        done
    done
done