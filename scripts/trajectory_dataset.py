
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


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


# Prepares data to be fed to data collator
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


        observations_idx = observations
        observations = np.zeros((max_ep_len, num_states))
        observations[np.arange(max_ep_len), observations_idx]

        actions_idx = actions
        actions = np.zeros((max_ep_len, num_actions))
        actions[np.arange(max_ep_len), actions_idx]

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

# For Huggingface trainer
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