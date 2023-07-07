from datasets import load_dataset
import numpy as np

dataset = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2")

print(dataset)
print("Observations shape:", np.array(dataset['train']['observations']).shape)
print("Actions shape:     ", np.array(dataset['train']['actions']).shape)
print("Rewards shape:     ", np.array(dataset['train']['rewards']).shape)
print("Dones shape:       ", np.array(dataset['train']['dones']).shape)