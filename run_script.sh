#!/bin/bash

script=$1
env=$2
agent=$3
seed=$4

if [ -z "$script" ]
then
    echo "script is empty"
    exit 1
fi

if [ -z "$env" ]
then
    echo "env is empty"
    exit 1
fi

if [ -z "$agent" ]
then
    echo "agent is empty"
    exit 1
fi

if [ -z "$seed" ]
then
    echo "seed is empty"
    exit 1
fi


screen -dmS ${script}_${env}_${agent}_${seed} \
    python3 scripts/$script.py \
        --seed $seed \
        --config configs/${env}_${agent}.yaml