import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, DecisionTransformerConfig, DecisionTransformerModel
import pickle
from tqdm import tqdm
import numpy as np
import random
from datasets import Dataset
from accelerate import Accelerator

from psrl.config import get_env_config
from psrl.utils import env_name_map

from arg_utils import get_experiment_parser, process_experiment_config
from trajectory_dataset import DecisionTransformerDataset
from metrics import compute_raw_accuracy, compute_last_action_accuracy
from utils import load_experiment_config, set_seed, get_file_path_from_config, get_experiment_path_from_config





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
        print(type(dataset))
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for obs in dataset["observations"]:
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

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }

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
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)
exp_config = process_experiment_config(args, exp_config)



# Setup experiment
set_seed(exp_config.seed)
print("*** SEED:", exp_config.seed)

data_dir = get_experiment_path_from_config(exp_config, mkdir=True, root_type='data')
accelerator = Accelerator(project_dir=data_dir)
device = accelerator.device



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = exp_config.no_goal
env = env_class(env_config)



# Get dataset of trajectories
checkpoints_path = os.path.join(os.path.dirname(__file__), exp_config.data_dir)
os.makedirs(checkpoints_path, exist_ok=True)

trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config)
with open(trajectories_path, 'rb') as f:
    raw_trajectory = pickle.load(f)

observations = []
actions = []
rewards = []
dones = []
for transition in raw_trajectory:
    obs, act, rew, _ = transition

    obs = F.one_hot(torch.LongTensor([obs]), num_classes=env.observation_space.n).float()[0].detach().numpy()
    act = F.one_hot(torch.LongTensor([act]), num_classes=env.action_space.n).float()[0].detach().numpy()

    observations.append(obs)
    actions.append(act)
    rewards.append(rew)
    dones.append(False)

processed_trajectory_data = {
    'observations': [observations],
    'actions': [actions],
    'rewards': [rewards],
    'dones': [dones],
}
ds = Dataset.from_dict(processed_trajectory_data)
# print(np.array(ds[0]['observations']).shape)
# print(np.array(ds['observations']).shape)
# processed_trajectory_data[0] = processed_trajectory_data

# collator = DecisionTransformerGymDataCollator(processed_trajectory_data)
# collator.max_len = exp_config.seq_len
# collator.max_ep_len = exp_config.training_steps

# dataset = DecisionTransformerDataset(
#     env,
#     processed_trajectory_data,
#     seq_len=exp_config.seq_len,
# )




# Get model
model_config = DecisionTransformerConfig()

model_config.state_dim = env.observation_space.n
model_config.act_dim = env.action_space.n
model_config.max_ep_len = exp_config.training_steps
# model_config.vocab_size = vocab_size
# model_config.hidden_state = vocab_size
model_config.n_positions = exp_config.seq_len
# model_config.n_ctx = exp_config.seq_len
# model_config.n_embd = exp_config.n_embd

# print(model_config)

model = TrainableDT(model_config)
model.to(device)



# Train
training_args = TrainingArguments(
    output_dir="output/",
    remove_unused_columns=False,
    num_train_epochs=120,
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
    train_dataset=ds,
    data_collator=DecisionTransformerGymDataCollator,
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