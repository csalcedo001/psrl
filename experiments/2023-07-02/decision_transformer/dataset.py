from datasets import load_dataset
import numpy as np

dataset = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2")

print(dataset)
print(np.array(dataset['train']['observations']).shape)