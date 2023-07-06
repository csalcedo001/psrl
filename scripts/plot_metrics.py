from setup_script import setup_script
from file_system import load_pickle
from plotting import save_losses_plot, save_accuracy_plot
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script(mode='plot')



# Get losses data
metrics_path = get_file_path_from_config('metrics.pkl', exp_config)
metrics = load_pickle(metrics_path)



# Save plots
print('Saving plots...')

save_losses_plot(
    metrics['train/batch_loss'],
    get_file_path_from_config('train_batch_loss.png', exp_config, mkdir=True, root_type='plots'),
    title='Training Loss',
    step_is_iteration=True,
)
save_losses_plot(
    metrics['train/loss'],
    get_file_path_from_config('train_loss.png', exp_config, mkdir=True, root_type='plots'),
    title='Training Loss',
)
save_accuracy_plot(
    metrics['val/accuracy'],
    get_file_path_from_config('val_accuracy.png', exp_config, root_type='plots'),
    title='Validation Accuracy',
)
save_accuracy_plot(
    metrics['val/last_action_accuracy'],
    get_file_path_from_config('val_last_action_accuracy.png', exp_config, root_type='plots'),
    title='Single-Action Validation Accuracy',
)