def rollout(env, agent, episodes):
    trajectories = []
    for episode in range(episodes):
        state = env.reset()

        trajectory = []
        while True:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            transition = (state, action, reward, next_state)
            trajectory.append(transition)

            if done:
                break
        
        trajectories.append(trajectory)
    
    return trajectories