# Test PSRL reward heatmap

This experiment is designed to show the heatmap produced by PSRL after training in a gridworld envionment by averaging multiple reward functions sampled from the agent's distribution

## Description of the experiment

In this experiment we first train PSRL in a gridworld environment with an initial state in the upper left corner and a terminal state in the lower left corner. After training, we use the agent's learnt reward function distribution to sample reward functions. We average together multiple reward functions to observe where the agent expects to find more reward.


## How to run

The configuration of the experiment is saved in `exp_config.yaml`, which has the following parameters:

- training_steps: Number of training steps
- side: Size of the side of the square gridworld. This means there are side^2 states
- psrl_heatmap_samples: Number of reward functions to sample from the trained agent

Train the agent by running `train.py` with this command

```bash
python train.py
```

Then run `heatmap.py` to load back the weights from the agent, sample multiple reward functions, and plot a heatmap from the average value found at each cell in the gridworld:

```bash
python heatmap.py
```