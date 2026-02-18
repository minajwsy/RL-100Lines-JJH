import argparse
import numpy as np

# ==============================================================================
# SAC Hyperparameters & Benchmarks
# Reference:
# - [Continuous] Haarnoja et al. (2018, 2019) "Soft Actor-Critic Algorithms and Applications"
# - [Discrete] Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
# ==============================================================================

ENV_CONFIGS = {
    # --------------------------------------------------------------------------
    # 1. Gymnasium (Classic Control - Toy Environments)
    # --------------------------------------------------------------------------
    'Pendulum-v1': {
        'lr_pi': 1e-3,
        'lr_q': 1e-3,
        'lr_alpha': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'target_freq': 1,
        'batch_size': 256,
        'buffer_limit': 100_000,
        'init_alpha': 1.0,
        'target_entropy': -1.0,
        'learning_starts': 1_000,
        'max_timesteps': 100_000
    },
    'CartPole-v1': {
        'lr_pi': 1e-3,
        'lr_q': 1e-3,
        'lr_alpha': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'target_freq': 1,
        'batch_size': 64,
        'buffer_limit': 100_000,
        'target_entropy': -1.0,
        'init_alpha': 0.01,
        'learning_starts': 200,
        'max_timesteps': 100_000
    },

    # --------------------------------------------------------------------------
    # 2. MuJoCo (Paper Standard Benchmarks)
    # --------------------------------------------------------------------------
    'Hopper-v5': {
        'lr_pi': 3e-4,
        'lr_q': 3e-4,
        'lr_alpha': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'target_freq': 1,
        'batch_size': 256,
        'buffer_limit': 1_000_000,
        'init_alpha': 1.0,
        'target_entropy': -3.0,
        'learning_starts': 10_000,
        'max_timesteps': 1_000_000
    },

    'HalfCheetah-v5': {
        'lr_pi': 3e-4,
        'lr_q': 3e-4,
        'lr_alpha': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'target_freq': 1,
        'batch_size': 256,
        'buffer_limit': 1_000_000,
        'init_alpha': 1.0,
        'target_entropy': -6.0,
        'learning_starts': 10_000,
        'max_timesteps': 3_000_000
    },

    'Humanoid-v4': {
        'lr_pi': 3e-4,
        'lr_q': 3e-4,
        'lr_alpha': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'target_freq': 1,
        'batch_size': 256,
        'buffer_limit': 1_000_000,
        'init_alpha': 0.05,
        'target_entropy': -17.0,
        'learning_starts': 10_000,
        'max_timesteps': 10_000_000
    },

    # --------------------------------------------------------------------------
    # 3. DeepMind Control Suite (DMC)
    # --------------------------------------------------------------------------
    'dm_control/cheetah-run-v0': {
        'lr_pi': 3e-4,
        'lr_q': 3e-4,
        'lr_alpha': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'target_freq': 1,
        'batch_size': 256,
        'buffer_limit': 1_000_000,
        'init_alpha': 0.1,
        'target_entropy': -6.0,
        'learning_starts': 10_000,
        'max_timesteps': 3_000_000
    },

    'dm_control/walker-walk-v0': {
        'lr_pi': 3e-4,
        'lr_q': 3e-4,
        'lr_alpha': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'target_freq': 1,
        'batch_size': 256,
        'buffer_limit': 1_000_000,
        'init_alpha': 0.1,
        'target_entropy': -6.0,
        'learning_starts': 10_000,
        'max_timesteps': 2_000_000
    }
}

parser = argparse.ArgumentParser(description='SAC Hyperparameters')
parser.add_argument('--env', type=str, default="Hopper-v5", help='Environment name')
args, _ = parser.parse_known_args()
env_name = args.env

default_cfg = ENV_CONFIGS['Pendulum-v1']

if env_name in ENV_CONFIGS:
    cfg = ENV_CONFIGS[env_name]
else:
    print(f"[Warning] '{env_name}' Use default environment(Pendulum-v1).")
    cfg = default_cfg

lr_pi = cfg['lr_pi']
lr_q = cfg['lr_q']
lr_alpha = cfg['lr_alpha']
gamma = cfg['gamma']
tau = cfg['tau']
target_freq = cfg['target_freq']
batch_size = cfg['batch_size']
buffer_limit = cfg['buffer_limit']
init_alpha = cfg['init_alpha']
target_entropy = cfg['target_entropy']
max_timesteps = cfg['max_timesteps']
learning_starts = cfg['learning_starts']
