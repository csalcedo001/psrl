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
* scripts/gridworld_plots.py: plots and video of trajectory (gridworlds only)

Additional:
* scripts/train.py: the previous script perform training by default, so this script is just a proof-of-concept that training is done correctly. Note: Uses W&B by default

Arguments:
* agent: psrl, random_agent, or optimal
* env: riverswim, randommdp, tworoom, or fourroom
* experiment_name: used as folder name within runs/ to save the plots. Default: date and uuid.


Run the following example command to get a plot of the regret of PSRL vs (so far) a random agent (soon UCRL2)

```bash
python scripts/plot_regret.py --experiment_name riverswim_regret --env riverswim --max_steps 10000
```

To get a video of the trajectory of an agent through a gridworld, run the next example command

```bash
python scripts/gridworld_plots.py --experiment_name psrl_tworoom --agent psrl --env tworoom
```

The script for training, train.py, can be run with the followiing command


```bash
python scripts/train.py --experiment_name train_psrl_tworoom --agent psrl --env tworoom
```

After running each of these commands, plots and data will be saved in a folder inside runs/, the default directory for results. The name can be set explicitly via --experiment_name or, in case omitted, is set to string composed of the current date and time, and a uuid.
