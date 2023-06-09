# Compare versions of UCRL2

This experiment is designed mainly to compare the efficiency of UCRL2 while maintaining its performance in both versions of code available.

## Description of the experiment

In this experiment we first train both versions of PSRL in RiverSwim environment for multiple number of states (from 5 to 150 in steps of 5). After training, we use the agent's trajectories to compute the regret, used to make plots. We also time the execution of the training loop to compare agent's performance.


## How to run

Simply run `main.py` in the usual way

```bash
python train.py
```

The script will produce three output images:

- regret_vs_states.png: regret in y-axis, number of states in x-axis.
- regret_vs_step.png: regret in y-axis, timestep in x-axis (for highest number of states).
- time_vs_states.png: execution time in y-axis, number of states in x-axis.