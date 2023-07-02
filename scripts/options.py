from gym import spaces
import numpy as np
import torch


class OptionSet:
    def __init__(self, model, seq_len, vocab_size, num_options):
        self.model = model
        self.seq_len = seq_len
        self.missing_token = vocab_size - 1
        self.num_options = num_options

        self.pos = None
        self.sequence = None
    
    def reset(self, state):
        self.pos = 1
        self.sequence = [self.missing_token] * self.seq_len
        self.sequence[0] = state
    
    def get_actions(self, option):
        num_actions = self.seq_len // 2
        missing_actions = option / max(self.num_options - 1, 1) * (num_actions - 2) + 1
        missing_s_and_a = 2 * missing_actions - 1

        sequence = torch.LongTensor(self.sequence)
        sequence[-missing_s_and_a:] = self.missing_token

        output = self.model(sequence.view(1, -1))
        next_sequence = output.logits.argmax(dim=-1)[0].cpu().numpy()

        action_idxs = np.arange(missing_actions) * 2 + self.seq_len - missing_s_and_a
        actions = next_sequence[action_idxs]

        return actions

    def update(self, action, state):
        # seq_len - 1 instead of seq_len because we want to keep the last token
        # as a missing token

        if self.pos == self.seq_len - 1:
            self.sequence.pop()
            self.sequence += [action, state]
            self.sequence.append(self.missing_token)
            self.sequence = self.sequence[2:]
        else:
            self.sequence[self.pos] = action
            self.sequence[self.pos + 1] = state

        self.pos = max(self.pos + 2, self.seq_len - 1)


class OptionEnvWrapper:
    def __init__(self, env, model, seq_len, vocab_size, num_options):
        self.env = env
        self.state = None
        self.option_set = OptionSet(model, seq_len, vocab_size, num_options)

        self.observation_space = self.env.observation_space
        self.action_space = spaces.Discrete(num_options)

        self.reset()
    
    def reset(self):
        self.state = self.env.reset()

        self.option_set.reset(self.state)

        return self.state
    
    def step(self, option):
        option_reward = 0
        option_len = 0
        done = False

        actions = self.get_actions(option)
        next_states = []
        rewards = []

        for action in actions:
            next_state, reward, done, _ = self.env.step(action)

            self.state = next_state

            rewards.append(reward)
            next_states.append(next_state)

            option_reward += reward
            option_len += 1

            self.option_set.update(action, self.state)

            if done:
                break
        
        avg_reward = option_reward / max(option_len, 1)

        info = {
            'actions': actions,
            'next_states': next_states,
            'rewards': rewards,
        }
        
        return self.state, avg_reward, done, info