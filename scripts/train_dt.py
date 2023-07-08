import copy
import glob
from transformers import Trainer, TrainingArguments, DecisionTransformerConfig
import numpy as np
from datasets import Dataset, DatasetDict


from setup_script import setup_script
from file_system import load_pickle
from trajectory_dataset import trajectories_to_dt_dataset_format, DecisionTransformerGymDataCollator
from models import TrainableDT
from utils import get_file_path_from_config




# Get experiment configuration
exp_config, env, agent, accelerator = setup_script(mode='run')
max_ep_len = 1000



# Get dataset of trajectories
data_config_pattern = copy.deepcopy(exp_config)
data_config_pattern.seed = '****'
data_path_pattern = get_file_path_from_config('trajectories.pkl', data_config_pattern)
paths = glob.glob(data_path_pattern)
paths.sort()

# Load trajectories
print("Loading trajectories...")
trajectories = []
for trajectory_path in paths:
    trajectory = load_pickle(trajectory_path)
    trajectories.append(trajectory)

# Split into train and val sets
print("Splitting into train and val sets...")
train_val_split_ratio = 0.9
train_trajectories = trajectories[:max(int(len(trajectories) * train_val_split_ratio), 1)]
val_trajectories = trajectories[int(len(trajectories) * train_val_split_ratio):]

# Get dataset of trajectories
print("Setting up training set...")
train_trajectory_data = trajectories_to_dt_dataset_format(
    train_trajectories,
    env.observation_space.n,
    env.action_space.n,
    max_ep_len,
)

print("Setting up validation set...")
val_trajectory_data = trajectories_to_dt_dataset_format(
    val_trajectories,
    env.observation_space.n,
    env.action_space.n,
    max_ep_len,
)

train_trajectory_dataset = Dataset.from_dict(train_trajectory_data)
val_trajectory_dataset = Dataset.from_dict(val_trajectory_data)

trajectory_dataset = DatasetDict({
    'train': train_trajectory_dataset,
    'val': val_trajectory_dataset,
})
print("Finished setting up datasets.")

print("Dataset:", trajectory_dataset)
print("Train set length:", len(train_trajectory_dataset[0]['observations']))
print("Observations shape:", np.array(train_trajectory_dataset[0]['observations']).shape)
print("Actions shape:     ", np.array(train_trajectory_dataset[0]['actions']).shape)
print("Rewards shape:     ", np.array(train_trajectory_dataset[0]['rewards']).shape)
print("Dones shape:       ", np.array(train_trajectory_dataset[0]['dones']).shape)

data_collator = DecisionTransformerGymDataCollator(trajectory_dataset['train'])
print("Finished data processing.")




# Get model
model_config = DecisionTransformerConfig()

model_config.max_length = max_ep_len
model_config.state_dim = env.observation_space.n
model_config.act_dim = env.action_space.n
model_config.max_ep_len = exp_config.training_steps

model = TrainableDT(model_config)
print("Model:")
print(model)



# Train
training_args = TrainingArguments(
    output_dir=get_file_path_from_config('dt_checkpoints', exp_config, mkdir=True),
    remove_unused_columns=False,
    num_train_epochs=exp_config.epochs,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
    save_steps=1000,
    logging_steps=50,
    eval_steps=50,
    save_total_limit=5,
    # resume_from_checkpoint=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trajectory_dataset['train'],
    eval_dataset=trajectory_dataset['val'],
    data_collator=data_collator,
)

trainer.train()





# trajectory_dataset = DecisionTransformerDataset(
#     raw_trajectories,
#     seq_len=exp_config.seq_len
# )
# vocab_size = env.action_space.n + env.observation_space.n + 2

# data_loader = DataLoader(
#     trajectory_dataset,
#     batch_size=exp_config.batch_size,
#     shuffle=True,
#     drop_last=True,
# )



# # Training
# model.train()
# optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
# criterion = torch.nn.CrossEntropyLoss()

# model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# metrics = {
#     'loss': [],
#     'raw_accuracy': [],
#     'last_action_accuracy': [],
# }

# print("Starting training...")
# for epoch in range(exp_config.epochs):
#     pbar = tqdm(total=len(data_loader))
#     for batch in data_loader:
#         states, actions, rewards, _, timesteps = batch

#         target_actions = actions.detach().clone().to(device)

#         states = F.one_hot(states, num_classes=env.observation_space.n).float()
#         actions = F.one_hot(actions, num_classes=env.action_space.n).float()
#         # return_to_go = rewards.sum(axis=-1, keepdim=True)
#         return_to_go = rewards.unsqueeze(-1).float()
#         timesteps = torch.arange(exp_config.seq_len).unsqueeze(0).repeat(exp_config.batch_size, 1).long()
        
#         output = model(
#             states=states.to(device),
#             actions=actions.to(device),
#             # rewards=rewards,
#             returns_to_go=return_to_go.to(device),
#             timesteps=timesteps.to(device),
#         )
#         loss = criterion(output.action_preds, target_actions)

#         accelerator.backward(loss)

#         optimizer.step()
#         optimizer.zero_grad()

#         metrics['loss'].append(loss.item())

#         pbar.update(1)
#         pbar.set_description(f"[{epoch}/{exp_config.epochs}] Loss: {loss.item():.4f}")
    
#     if epoch % 10 == 0:
#         checkpoints_dir = get_file_path_from_config('decision_transformer', exp_config)
#         accelerator.save_state(checkpoints_dir)



# # Evaluation after training
# device = accelerator.device

# x, y = next(iter(data_loader))
# x = x.to(device)

# model.eval()
# output = model(input_ids=x)
# y_hat_post = output.logits.argmax(dim=-1).cpu().numpy()

# print("x:")
# print(x.cpu().detach().numpy()[-10:, -20:])
# print()

# print("y_hat:")
# print(y_hat_post[-10:, -20:])
# print()

# print("y:")
# print(y.cpu().numpy()[-10:, -20:])
# print()



# # Save results
# model_path = get_file_path_from_config('model.pt', exp_config, mkdir=True)
# torch.save(model.state_dict(), model_path)

# metrics_path = get_file_path_from_config('metrics.pkl', exp_config, mkdir=True)
# with open(metrics_path, 'wb') as f:
#     pickle.dump(metrics, f)