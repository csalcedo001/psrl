import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def choose_gridworld_color(symbol):
    if symbol == ' ':
        color = 'w'
    elif symbol == '#':
        color = 'k'
    elif symbol == 'S':
        color = 'b'
    elif symbol == 'T':
        color = 'g'
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


def save_policy_plot(env, agent, state_to_pos, filename, path='/.', title=None):
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


        pos = np.array(state_to_pos[state])

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

    file_path = os.path.join(path, filename + '.png')
    plt.savefig(file_path)
    plt.close(fig)


def save_expected_reward_heatmap_plot(env, state_to_pos, r_hat, filename, path='./', title=None):
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
        i, j = state_to_pos[state]
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(r[state])))
    
    fig.colorbar(cmap, ax=ax)
        
    file_path = os.path.join(path, filename + '.png')
    plt.savefig(file_path)


def save_state_value_heatmap_plot(env, state_to_pos, v, filename, path='./', title=None):
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
        i, j = state_to_pos[state]
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(v[state])))
    
    fig.colorbar(cmap, ax=ax)
        
    file_path = os.path.join(path, filename + '.png')
    plt.savefig(file_path)


def save_action_value_heatmap_plot(env, state_to_pos, q, filename, path='./', title=None):
    print("Processing state value function heatmap plot...")

    action_name_map = ['up', 'right', 'down', 'left']


    fig, axes = plt.subplots(2, 2)
    plt.title(title)

    for row in range(2):
        for col in range(2):
            ax = axes[row, col]

            action = row * 2 + col

            v = q[:, action]

            v_min = v.min()
            v_max = v.max()

            cmap = mpl.colormaps['plasma']
            norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            ax.title.set_title('Action value for action={}'.format(action_name_map[action]))
            init_plt_grid(ax, env)
            
            for state in range(env.observation_space.n):
                i, j = state_to_pos[state]
                    
                x = j
                y = env.rows - i - 1

                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(v[state])))
            
            ax.colorbar(cmap, ax=ax)
                
            file_path = os.path.join(path, filename + '.png')
            plt.savefig(file_path)


def save_empirical_state_visitation_heatmap_plot(env, state_to_pos, state_count, filename, path='./', title=None):
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
        i, j = state_to_pos[state]
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(state_count[state])))

    fig.colorbar(cmap, ax=ax)

    file_path = os.path.join(path, filename + '.png')
    plt.savefig(file_path)


def save_reward_count_heatmap_plot(env, state_to_pos, r_count, p_count, filename, path='./', title=None):
    print("Processing empirical total reward plot...")

    r_emp = r_count.sum(axis=1) / np.clip(p_count.sum(axis=(1, 2)), 1, None)
    r_min = r_emp.min()
    r_max = r_emp.max()

    cmap = mpl.colormaps['plasma']
    norm = mpl.colors.Normalize(vmin=r_min, vmax=r_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots()
    plt.title(title)
    init_plt_grid(ax, env)

    for state in range(env.observation_space.n):
        i, j = state_to_pos[state]
            
        x = j
        y = env.rows - i - 1

        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=cmap.to_rgba(r_emp[state])))

    fig.colorbar(cmap, ax=ax)

    file_path = os.path.join(path, filename + '.png')
    plt.savefig(file_path)