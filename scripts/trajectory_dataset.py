
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, env, raw_trajectories, seq_len=64, beta=4):
        if seq_len % 2 != 0:
            raise ValueError("seq_len must be even")
        
        self.seq_len = seq_len
        self.beta = beta
        self.missing_token = env.observation_space.n + env.action_space.n
        
        trajectories = []
        for s, a, _, _ in raw_trajectories:
            trajectories.append(s + env.action_space.n)
            trajectories.append(a)
        
        self.trajectories = trajectories
    
    def get_vocab_size(self):
        return self.missing_token + 1

    def __len__(self):
        return (len(self.trajectories) - self.seq_len) // 2
    
    def __getitem__(self, idx):
        # Get trajectory of length seq_len
        trajectory = np.array(self.trajectories[2 * idx: 2 * idx + self.seq_len])

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