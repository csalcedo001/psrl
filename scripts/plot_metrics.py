import os
import pickle

from plotting import save_losses_plot, save_accuracy_plot
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
    exp_config.save_path = os.path.join(exp_config.save_path, 'regret_plot')
    exp_config.plots_path = os.path.join(exp_config.plots_path, 'regret_plot')



# Get losses data
metrics_path = get_file_path_from_config('metrics.pkl', exp_config)
with open(metrics_path, 'rb') as f:
    metrics = pickle.load(f)



# Save plots
print('Saving plots...')
save_losses_plot(
    metrics['loss'],
    get_file_path_from_config('training_loss.png', exp_config, mkdir=True, root_type='plots'),
    title='Training Loss',
)
save_accuracy_plot(
    metrics['raw_accuracy'],
    get_file_path_from_config('training_raw_accuracy.png', exp_config, root_type='plots'),
    title='Raw Accuracy',
)
save_accuracy_plot(
    metrics['last_action_accuracy'],
    get_file_path_from_config('training_last_action_accuracy.png', exp_config, root_type='plots'),
    title='Last Action Accuracy',
)