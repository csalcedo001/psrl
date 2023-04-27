# Posterior Sampling for Reinforcement Learning

Implementation of PSRL from [the paper](https://proceedings.neurips.cc/paper/2013/hash/6a5889bb0190d0211a991f47bb19a777-Abstract.html):

Osband, I., Russo, D., & Van Roy, B. (2013). (More) efficient reinforcement learning via posterior sampling. Advances in Neural Information Processing Systems, 26.

## Installation

1. Create conda environment

```bash
cd psrl/
conda create --name psrl python=3.9
conda activate psrl
```

2. Install requirements

```bash
pip install -r requirements.txt
pip install -e .
```


## Running experiments

Main executables:
* scripts/plot_regret.py: plots like Figure 2 in the paper
* scripts/gridworld_video.py: video of trajectory (gridworlds only)

Arguments:
* agent: psrl, random_agent, or optimal
* env: riverswim, randommdp, tworoom, fourroom


Run the next command to get a plot of the regret of PSRL vs (so far) a random agent (soon UCRL2)

```bash
python scripts/plot_regret.py --env riverswim --max_steps 10000
```

To get a video of the trajectory of an agent through a gridworld, run the following command

```bash
python scripts/gridworld_video.py --agent psrl --env tworoom
```