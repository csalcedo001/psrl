#!/bin/bash



# Get CLI arguments
script_name=$1
env=$2
agent=$3
seed=$4



# Validate arguments
if [ -z "$script_name" ]
then
    script_name="generate_data"
fi

if [ -z "$env" ]
then
    env="tworoom"
fi

if [ -z "$agent" ]
then
    agent="ucrl2"
fi

if [ -z "$seed" ]
then
    seed="0"
fi



script="scripts/$script_name.py"

if [ -f "$script" ]; then
    # Run script
    
    echo "Running $script with env=$env, agent=$agent, seed=$seed"

    screen -dmS ${script_name}_${env}_${agent}_${seed} \
        python3 $script \
            --seed $seed \
            --config configs/${env}_${agent}.yaml
else
    echo "Script $script does not exist"
fi