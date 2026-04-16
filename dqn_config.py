import torch as T
import argparse

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# ==============================================================================
# DQN Hyperparameters & Benchmarks
# Reference:
# - Mnih et al. (2015) "Human-level control through deep reinforcement learning"
# - RL Baselines3 Zoo tuned hyperparameters (hyperparams/dqn.yml)
# - CleanRL DQN benchmarks (docs.cleanrl.dev/rl-algorithms/dqn/)
# NOTE: DQN requires discrete action spaces.
#       Standard MuJoCo environments (continuous) are not supported.
# ==============================================================================

ENV_CONFIGS = {
    # --------------------------------------------------------------------------
    # 1. Gymnasium (Classic Control)
    # --------------------------------------------------------------------------
    # Expected Return: 500 ± 0 (perfect score) at ~50k timesteps
    'CartPole-v1': {
        'lr_q': 2.3e-3,
        'gamma': 0.99,
        'buffer_limit': 100_000,
        'batch_size': 64,
        'eps_start': 1.0,
        'eps_end': 0.04,
        'eps_decay_frac': 0.16,
        'train_freq': 4,
        'target_freq': 500,
        'learning_starts': 1_000,
        'max_timesteps': 50_000,
        'n_envs': 1,
    },

    # Expected Return: ~200 ± 80 at 500k timesteps (high variance; DDQN recommended for reliable convergence)
    'LunarLander-v3': {
        'lr_q': 6.3e-4,
        'gamma': 0.99,
        'buffer_limit': 50_000,
        'batch_size': 128,
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_decay_frac': 0.12,
        'target_freq': 250,
        'learning_starts': 0,
        'max_timesteps': 500_000,
        'n_envs': 1,
        'train_freq': 4,
    },

    # Expected Return: -101 ± 10 at ~120k timesteps
    # NOTE: Standard DQN hyperparams fail due to sparse rewards.
    #       Critical: high lr (4e-3), lower gamma (0.98), long target_freq (600).
    'MountainCar-v0': {
        'lr_q': 4e-3,
        'gamma': 0.98,
        'buffer_limit': 10_000,
        'batch_size': 128,
        'eps_start': 1.0,
        'eps_end': 0.07,
        'eps_decay_frac': 0.2,
        'target_freq': 600,
        'learning_starts': 1_000,
        'max_timesteps': 120_000,
        'n_envs': 1,
        'train_freq': 16,
    },

    # Expected Return: -77 ± 12 at ~100k timesteps (solving threshold: -100)
    'Acrobot-v1': {
        'lr_q': 6.3e-4,
        'gamma': 0.99,
        'buffer_limit': 50_000,
        'batch_size': 128,
        'eps_start': 1.0,
        'eps_end': 0.1,
        'eps_decay_frac': 0.12,
        'target_freq': 250,
        'learning_starts': 0,
        'max_timesteps': 100_000,
        'n_envs': 1,
        'train_freq': 4,
    },
}

parser = argparse.ArgumentParser(description='DQN Hyperparameters')
parser.add_argument('--env', type=str, default="CartPole-v1", help='Environment name')
args, _ = parser.parse_known_args()
env_name = args.env

default_cfg = ENV_CONFIGS['CartPole-v1']

if env_name in ENV_CONFIGS:
    cfg = ENV_CONFIGS[env_name]
else:
    print(f"[Warning] '{env_name}' not found. Using default environment (CartPole-v1).")
    cfg = default_cfg

lr_q = cfg['lr_q']
gamma = cfg['gamma']
buffer_limit = cfg['buffer_limit']
batch_size = cfg['batch_size']
eps_start = cfg['eps_start']
eps_end = cfg['eps_end']
eps_decay_frac = cfg['eps_decay_frac']
target_freq = cfg['target_freq']
learning_starts = cfg['learning_starts']
max_timesteps = cfg['max_timesteps']
n_envs = cfg['n_envs']
train_freq = cfg['train_freq']
