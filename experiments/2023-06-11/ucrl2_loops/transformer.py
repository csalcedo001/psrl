import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Tokenizer

# Define the TrajectoryDataset for training the GPT-2 model
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

# Define the GPT2ModelTraining class
class GPT2ModelTraining:
    def __init__(self, trajectories, state_dim, action_dim):
        self.trajectories = trajectories
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize the GPT-2 model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def prepare_input(self, trajectory):
        inputs = []
        for transition in trajectory:
            state, action, reward, next_state = transition
            input_text = f"State: {state}, Action: {action}, Reward: {reward}, Next state: {next_state}"
            inputs.append(input_text)
        return inputs

    def preprocess(self, inputs):
        input_ids = []
        attention_masks = []

        for input_text in inputs:
            encoded_inputs = self.tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=512,
                                                        pad_to_max_length=True, return_tensors='pt')
            input_ids.append(encoded_inputs['input_ids'])
            attention_masks.append(encoded_inputs['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def train_model(self):
        dataset = TrajectoryDataset(self.trajectories)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(10):  # Replace with your desired number of training epochs
            for batch in train_loader:
                inputs = self.prepare_input(batch)
                input_ids, attention_masks = self.preprocess(inputs)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()


# Example usage
trajectories = [
    [(1, 3, 0.5, 2), (2, 1, 0.2, 3), (3, 2, 0.7, 4)],  # Example trajectory 1
    [(1, 2, 0.9, 3), (3, 1, 0.3, 2), (2, 3, 0.6, 4)],  # Example trajectory 2
    # Add more trajectories as needed
]

state_dim = 5
action_dim = 3

# Create the GPT-2 model training instance
gpt2_trainer = GPT2ModelTraining(trajectories, state_dim, action_dim)

# Train the GPT-2 model
gpt2_trainer.train_model()