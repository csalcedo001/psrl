#!/bin/bash

for i in {1..20}
do
    echo "Running experiment $i"
    nohup python3 train.py --env riverswim --agent psrl --max_steps 10000 > /dev/null &
done