import copy
import glob
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, DecisionTransformerConfig, DecisionTransformerModel
import numpy as np
import random
from tqdm import tqdm
from datasets import Dataset, DatasetDict


from setup_script import setup_script
from file_system import load_pickle
from utils import get_file_path_from_config




def trajectories_to_dt_dataset_format(trajectories, num_states, num_actions, max_ep_len):
    dataset_observations = []
    dataset_actions = []
    dataset_rewards = []
    dataset_dones = []

    split_trajectories = []
    for trajectory in trajectories:
        for i in range(len(trajectory) // max_ep_len):
            split_trajectory = trajectory[i * max_ep_len: (i + 1) * max_ep_len]
            split_trajectories.append(split_trajectory)

    for trajectory in tqdm(split_trajectories):
        observations = []
        actions = []
        rewards = []
        dones = []

        for transition in trajectory:
            obs, act, rew, _ = transition

            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            dones.append(False)

        observations = F.one_hot(torch.LongTensor([observations]), num_classes=num_states).float()[0].detach().numpy()
        actions = F.one_hot(torch.LongTensor([actions]), num_classes=num_actions).float()[0].detach().numpy()
        rewards = np.array([rewards]).T
        dones = np.array([dones]).T
        
        dataset_observations.append(observations)
        dataset_actions.append(actions)
        dataset_rewards.append(rewards)
        dataset_dones.append(dones)


    trajectories_data = {
        'observations': dataset_observations,
        'actions': dataset_actions,
        'rewards': dataset_rewards,
        'dones': dataset_dones,
    }

    return trajectories_data

class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 20 #subsets of the episode we use for training
    state_dim: int = 17  # size of state space
    act_dim: int = 6  # size of action space
    max_ep_len: int = 1000 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for i in tqdm(range(len(dataset))):
            obs = dataset[i]["observations"]
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        
        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)

            # get sequences from dataset
            s.append(np.array(feature["observations"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["actions"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))

            d.append(np.array(feature["dones"][si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        data = {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }

        # print("STATE: ", s.shape)
        # print("ACTION:", a.shape)
        # print("REWARD:", r.shape)
        # print("RTG:   ", rtg.shape)
        # print("TIME:  ", timesteps.shape)
        # print("MASK:  ", mask.shape)

        return data

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)




# Get experiment configuration
exp_config, env, agent, accelerator = setup_script(mode='debug')
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
    output_dir="output/",
    remove_unused_columns=False,
    num_train_epochs=2,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=trajectory_dataset['train'],
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