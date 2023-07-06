
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, env, raw_trajectories, seq_len=64, beta=8, stride=None):
        if seq_len % 2 != 0:
            raise ValueError("seq_len must be even")
        
        self.seq_len = seq_len
        self.beta = beta
        self.missing_token = env.observation_space.n + env.action_space.n

        if stride == None:
            stride = seq_len // 2
        self.stride = stride
        
        # We assume all trajectories have same length
        trajectories = []
        for raw_trajectory in raw_trajectories:
            trajectory = []
            for s, a, _, _ in raw_trajectory:
                trajectory.append(s + env.action_space.n)
                trajectory.append(a)

            trajectories.append(trajectory)
        
        self.trajectories = trajectories

        self.trajectory_num_idxs = ((len(self.trajectories[0]) - self.seq_len) // 2) // self.stride
        self.dataset_num_idxs = len(self.trajectories) * self.trajectory_num_idxs
    
    def get_vocab_size(self):
        return self.missing_token + 1

    def __len__(self):
        return self.dataset_num_idxs
    
    def __getitem__(self, idx):
        # Get trajectory of length seq_len
        traj_idx = idx // self.trajectory_num_idxs
        step_idx = idx % self.trajectory_num_idxs

        start_idx = 2 * step_idx * self.stride
        end_idx = start_idx + self.seq_len

        trajectory = np.array(self.trajectories[traj_idx][start_idx : end_idx])

        y = torch.LongTensor(trajectory)

        # Mask some actions
        missing_actions = 1 + min(int(np.random.exponential(self.beta)), self.seq_len // 2)
        missing_s_and_a = 2 * missing_actions - 1

        x = torch.LongTensor(trajectory.copy())
        x[-missing_s_and_a:] = self.missing_token

        return x, y



class DecisionTransformerDataset(Dataset):
    def __init__(self, env, raw_trajectories, seq_len=64):
        if seq_len % 2 != 0:
            raise ValueError("seq_len must be even")

        self.env = env
        self.seq_len = seq_len
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.timesteps = []
        t = 0
        for s, a, r, s_ in raw_trajectories:
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(s_)
            self.timesteps.append(t)

            t +=1

    def __len__(self):
        return (len(self.states) - self.seq_len) // 2
    
    def __getitem__(self, idx):
        # Get trajectory of length seq_len
        states = torch.LongTensor(self.states[idx: idx + self.seq_len])
        actions = torch.LongTensor(self.actions[idx: idx + self.seq_len])
        rewards = torch.LongTensor(self.rewards[idx: idx + self.seq_len])
        next_states = torch.LongTensor(self.next_states[idx: idx + self.seq_len])
        timesteps = torch.LongTensor(self.next_states[idx: idx + self.seq_len])

        states = F.one_hot(states, num_classes=self.env.observation_space.n).float()
        actions = F.one_hot(actions, num_classes=self.env.action_space.n).float()



        return {
            'observations': states,
            'actions': actions,
            'rewards': rewards,
        }