#!/bin/bash



# Setup arrays
sweep_ids=("7ldzge41" "0ydouz3y" "ynissmqk" "15zz7m04")
instances=(3)
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"



# Run scripts
echo "Setup: seep_ids=(${sweep_ids[@]}), ids=(${instances[@]}), GPUs=(${gpus[@]})"

gpu_i=0

for sweep_id in "${sweep_ids[@]}"
do
    for instance in "${instances[@]}"
    do
        echo "Running run_script.sh on GPU='${gpus[$gpu_i]}' with sweep_id=$sweep_id, instance=$instance"

        CUDA_VISIBLE_DEVICES="${gpus[$gpu_i]}" screen -dmS "sweep_${sweep_id}_${instance}_GPU${gpus[$gpu_i]}" \
            wandb agent cesar-salcedo/psrl/$sweep_id

        # increment i by one and loop back when we reach the end of the array
        ((gpu_i++))
        if [ $gpu_i -ge ${#gpus[@]} ]; then
            gpu_i=0
        fi
    done
done