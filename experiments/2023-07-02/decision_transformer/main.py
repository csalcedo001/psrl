from transformers import DecisionTransformerModel, DecisionTransformerConfig
import numpy as np
import torch
import gym

model_name = "edbeeching/decision-transformer-gym-hopper-expert"
model = DecisionTransformerModel.from_pretrained(model_name)
# model_config = DecisionTransformerConfig()
# model = DecisionTransformerModel(config=model_config)

print(model)
print(model.config)


env = gym.make("Hopper-v3")
state_dim = env.observation_space.shape[0] # state size
act_dim = env.action_space.shape[0] # action size



# Function that gets an action from the model using autoregressive prediction 
# with a window of the previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards
    
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)
    
    # The prediction is conditioned on up to 20 previous time-steps
    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    
    # pad all tokens to sequence length, this is required if we process batches
    padding = model.config.max_length - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)
    
    # perform the prediction
    state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,)
    return action_preds[0, -1]


TARGET_RETURN = 3.6 # This was normalized during training
MAX_EPISODE_LENGTH = 1000 

max_ep_len = 100
scale = 1.0

state_mean = np.array(
    [1.3490015,  -0.11208222, -0.5506444,  -0.13188992, -0.00378754,  2.6071432,
     0.02322114, -0.01626922, -0.06840388, -0.05183131,  0.04272673,])

state_std = np.array(
    [0.15980862, 0.0446214,  0.14307782, 0.17629202, 0.5912333,  0.5899924,
         1.5405099,  0.8152689,  2.0173461,  2.4107876,  5.8440027,])

state_mean = torch.from_numpy(state_mean)
state_std = torch.from_numpy(state_std)

state = env.reset()
target_return = torch.tensor(TARGET_RETURN).float().reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).float()
actions = torch.zeros((0, act_dim)).float()
rewards = torch.zeros(0).float()
timesteps = torch.tensor(0).reshape(1, 1).long()

# take steps in the environment
for t in range(max_ep_len):
    # add zeros for actions as input for the current time-step
    actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1)])

    # predicting the action to take
    action = get_action(model,
                        (states - state_mean) / state_std,
                        actions,
                        rewards,
                        target_return,
                        timesteps)
    actions[-1] = action
    action = action.detach().numpy()

    # interact with the environment based on this action
    state, reward, done, _ = env.step(action)
    
    cur_state = torch.from_numpy(state).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward
    
    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)
    
    if done:
        break