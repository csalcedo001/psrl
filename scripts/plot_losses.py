import os
import pickle

from plotting import save_losses_plot
from arg_utils import get_experiment_parser
from utils import load_experiment_config, set_seed, get_file_path_from_config




# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)



# Setup experiment
seed = exp_config.seed
if args.seed is not None:
    seed = args.seed
set_seed(seed)
print("*** SEED:", seed)

no_goal = exp_config.no_goal
if args.goal_reward is not None:
    no_goal = args.goal_reward == 0
    
if not no_goal:
    exp_config.save_dir = os.path.join(exp_config.save_dir, 'regret_plot')



# Get losses data
losses_path = get_file_path_from_config('losses.pkl', exp_config)
with open(losses_path, 'rb') as f:
    losses = pickle.load(f)



# Save plots
print('Saving plots...')
save_losses_plot(
    losses,
    get_file_path_from_config('training_loss.png', exp_config, root_type='plots'),
    title='Training Loss',
)