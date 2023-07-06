import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


env_plot_name_map = {
    'riverswim': 'RiverSwim',
    'randommdp': 'RandomMDP',
    'tworoom': 'TwoRoom',
    'fourroom': 'FourRoom',
    'gridworld': 'GridWorld',
}
agent_plot_name_map = {
    'psrl': 'PSRL',
    'ucrl2': 'UCRL2',
    'kl_ucrl': 'KL-UCRL',
    'random_agent': 'Random',
    'optimal': 'Optimal',
}


def choose_gridworld_color(symbol):
    if symbol == ' ':
        color = 'w'
    elif symbol == '#':
        color = 'k'
    elif symbol == 'S':
        color = 'b'
    elif symbol == 'G':
        color = 'g'
    elif symbol == 'R':
        color = 'y'
    elif symbol == '.':
        color = '#7f7f7f'
    else:
        color = None
    
    return color


def init_plt_grid(ax, env):
    plt.xlim(0, env.cols)
    plt.ylim(0, env.rows)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(env.rows):
        for j in range(env.cols):
            color = choose_gridworld_color(env.grid[i][j])
            
            x = j
            y = env.rows - i - 1

            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))

def add_cell_divisions_to_grid(ax, env):
    x = np.arange(0, env.cols + 1)
    y_0 = np.zeros(env.cols + 1)
    y_1 = np.ones(env.cols + 1) * env.rows

    vertical = np.stack([x, y_0, x, y_1]).T

    y = np.arange(0, env.rows + 1)
    x_0 = np.zeros(env.rows + 1)
    x_1 = np.ones(env.rows + 1) * env.cols

    horizontal = np.stack([x_0, y, x_1, y]).T

    # Plot lines
    for x0, y0, x1, y1 in vertical:
        plt.plot([x0, x1], [y0, y1], 'k', linewidth=0.5)

    for x0, y0, x1, y1 in horizontal:
        plt.plot([x0, x1], [y0, y1], 'k', linewidth=0.5)

def save_gridworld_plot(env, env_name, file_path):
    print("Processing gridworld plot...")

    fig, ax = plt.subplots()
    plt.title(f'{env_plot_name_map[env_name]} Environment')

    init_plt_grid(ax, env)
    add_cell_divisions_to_grid(ax, env)

    plt.savefig(file_path)
    plt.close(fig)

def save_policy_plot(env, agent, file_path, title=None):
    print("Processing policy plot...")

    origins = []
    vectors = []
    for state in range(env.observation_space.n):
        action = agent.act(state)

        # up: 0, right: 1, down: 2, left: 3
        axis = action % 2
        direction = action // 2

        
        # Correction for plot
        if axis == 0:
            direction = 1 - direction


        pos = np.array(env._get_pos_from_state(state))

        next_pos = pos.copy()
        next_pos[axis] += 1 if direction == 0 else -1


        # Correction for plot
        pos = pos[::-1]
        pos[1] = env.rows - pos[1] - 1
        next_pos = next_pos[::-1]
        next_pos[1] = env.rows - next_pos[1] - 1


        dir_vec = next_pos - pos
        
        origins.append(pos + 0.5 - dir_vec * 0.4)
        vectors.append(dir_vec * 0.8)

    origins = np.array(origins).T
    vectors = np.array(vectors).T


    fig, ax = plt.subplots()
    plt.title(title)
    init_plt_grid(ax, env)

    plt.quiver(*origins, *vectors, color='#000000', scale=1, scale_units='xy', angles='xy')

    add_cell_divisions_to_grid(ax, env)
    

    plt.savefig(file_path)
    plt.close(fig)


def save_expected_reward_heatmap_plot(env, r_hat, file_path, title=None):
    print("Processing expected reward heatmap plot...")

    r = r_hat.sum(axis=1)
    r_min = r.min()
    r_max = r.max()
    

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=r_min, vmax=r_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title(title)
    init_plt_grid(ax, env)
    
    for state in range(env.observation_space.n):
        i, j = env._get_pos_from_state(state)
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(r[state])))
    
    fig.colorbar(cmap, ax=ax)

    add_cell_divisions_to_grid(ax, env)
    

    plt.savefig(file_path)
    plt.close(fig)


def save_action_value_heatmap_plot(env, q, file_path, title=None):
    print("Processing action value function heatmap plot...")

    q_min = q.min()
    q_max = q.max()

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=q_min, vmax=q_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title(title)
    init_plt_grid(ax, env)

    rotation_mat = np.array([
        [[1,  0], [0,  1]],
        [[0, -1], [1,  0]],
        [[-1, 0], [0, -1]],
        [[0,  1], [-1, 0]]
    ])

    upper_triangle = np.array([
        [0,      0],
        [-0.5, 0.5],
        [0.5,  0.5]
    ])
    
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            i, j = env._get_pos_from_state(state)
                
            x = j
            y = env.rows - i - 1

            x += 0.5
            y += 0.5

            triang_pts = upper_triangle @ rotation_mat[action] + np.array([[x, y]])
            
            ax.add_patch(plt.Polygon(triang_pts, color=cmap.to_rgba(q[state, action])))
    
    fig.colorbar(cmap, ax=ax)

    add_cell_divisions_to_grid(ax, env)
        
    
    plt.savefig(file_path)
    plt.close(fig)


def save_state_value_heatmap_plot(env, v, file_path, title=None):
    print("Processing state value function heatmap plot...")

    v_min = v.min()
    v_max = v.max()

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title(title)
    init_plt_grid(ax, env)
    
    for state in range(env.observation_space.n):
        i, j = env._get_pos_from_state(state)
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(v[state])))
    
    fig.colorbar(cmap, ax=ax)

    add_cell_divisions_to_grid(ax, env)
        
    
    plt.savefig(file_path)
    plt.close(fig)


def save_empirical_state_visitation_heatmap_plot(env, state_count, file_path, title=None):
    print("Processing empirical state visitation plot...")

    sc_min = state_count.min()
    sc_max = state_count.max()


    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=sc_min, vmax=sc_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title(title)
    init_plt_grid(ax, env)

    for state in range(env.observation_space.n):
        i, j = env._get_pos_from_state(state)
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(state_count[state])))

    fig.colorbar(cmap, ax=ax)

    add_cell_divisions_to_grid(ax, env)

    
    plt.savefig(file_path)
    plt.close(fig)


def save_reward_count_heatmap_plot(env, r_count, file_path, title=None):
    print("Processing empirical total reward plot...")

    r_emp = r_count.sum(axis=1)
    r_min = r_emp.min()
    r_max = r_emp.max()

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=r_min, vmax=r_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title(title)
    init_plt_grid(ax, env)

    for state in range(env.observation_space.n):
        i, j = env._get_pos_from_state(state)
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(r_emp[state])))

    fig.colorbar(cmap, ax=ax)

    add_cell_divisions_to_grid(ax, env)

    
    plt.savefig(file_path)
    plt.close(fig)

def save_losses_plot(losses, file_path, title=None, x_is_iterations=True):
    print("Processing losses plot...")

    fig, ax = plt.subplots()
    plt.title(title)

    if x_is_iterations:
        plt.xlabel("Iteration")
    else:
        plt.xlabel("Epochs")
    
    plt.ylabel("Loss")

    ax.plot(losses)

    plt.savefig(file_path)
    plt.close(fig)

def save_accuracy_plot(accuracies, file_path, title=None):
    print("Processing accuracy plot...")

    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    x = np.arange(len(accuracies)) * 10
    ax.plot(accuracies)

    plt.savefig(file_path)
    plt.close(fig)

def save_regret_plot(agent_regrets, file_path, title=None):
    print("Processing regret plot...")

    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Regret")

    cmap = mpl.colormaps['tab10']
    for i, agent in enumerate(agent_regrets):
        regrets = np.array(agent_regrets[agent])
        
        mean_regret = np.mean(regrets, axis=0)
        min_regret = np.min(regrets, axis=0)
        max_regret = np.max(regrets, axis=0)

        x_index = np.arange(len(mean_regret))

        c = cmap(i % 10)

        ax.fill_between(x_index, min_regret, max_regret, alpha=0.5, color=c)
        ax.plot(x_index, mean_regret, label=agent_plot_name_map[agent], color=c)

    plt.legend()

    plt.savefig(file_path)
    plt.close(fig)