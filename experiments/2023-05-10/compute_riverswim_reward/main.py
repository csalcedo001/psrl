import numpy as np

from psrl.envs import RiverSwimEnv
from psrl.config.utils import get_env_config

def compute_aprox_cum_p_conv(p, n):
    pc = p
    p_total = np.zeros_like(p)
    for _ in range(n):
        p_total += pc
        pc = np.matmul(pc, p)
    
    return p_total


env_config = get_env_config('riverswim')
env_config.n = 10
env = RiverSwimEnv(env_config)

print(f'Number of states: {env.observation_space.n}')

p, r = env.get_p_and_r()
r[0, 0] = 0.005
r[-1, 1] = 1

s0 = np.zeros((env.observation_space.n, 1))
s0[0, 0] = 1
n = 1000

for a in range(env.action_space.n):
    pa = p[:, a]
    ra = r[:, a].sum(axis=1, keepdims=True)
    # print(ra)

    s = s0
    expected_r = 0
    for _ in range(n):
        # print(s)
        expected_r += np.sum(s * ra)
        s = np.matmul(pa.T, s)

    expected_r += np.sum(s * ra)

    expected_r /= (n + 1)
    


    # pc = pa
    # pc_total = np.zeros_like(pa)
    # for _ in range(n):
    #     pc_total += pc
    #     pc = np.matmul(pc, pa)

    # s_distr = np.dot(pc_total, s0)
    # expected_r = np.sum(s_distr * ra)

    print(f'Expected reward for action {a}: {expected_r}')