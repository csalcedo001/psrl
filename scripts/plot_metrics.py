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
    metrics['loss'],
    get_file_path_from_config('training_loss.png', exp_config, mkdir=True, root_type='plots'),
    title='Training Loss',
)
save_losses_plot(
    metrics['val_loss'],
    get_file_path_from_config('val_loss.png', exp_config, mkdir=True, root_type='plots'),
    title='Validation Loss',
    x_is_iterations=False,
)
save_accuracy_plot(
    metrics['val_accuracy'],
    get_file_path_from_config('val_accuracy.png', exp_config, root_type='plots'),
    title='Validation Accuracy',
)
save_accuracy_plot(
    metrics['val_last_action_accuracy'],
    get_file_path_from_config('val_last_action_accuracy.png', exp_config, root_type='plots'),
    title='Single-Action Validation Accuracy',
)