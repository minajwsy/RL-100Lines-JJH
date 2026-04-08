# 100LinesRL

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)
![Code Size](https://img.shields.io/github/languages/code-size/jaehyun-jeong/100LinesRL)
![Stars](https://img.shields.io/github/stars/jaehyun-jeong/100LinesRL?style=social)

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

Inspired by [minimalRL](https://github.com/seungeunrho/minimalRL).

## Algorithms

| Algorithm | Lines | Action Space | Vectorized Envs | CUDA | Key Features | Supported Environments |
|:---------:|:-----:|:------------:|:---------------:|:----:|:-------------|:----------------------|
| [PPO](https://github.com/jaehyun-jeong/100LinesRL/blob/master/ppo.py) | 100 | Discrete, Continuous | Yes | Yes | GAE, Obs/Reward normalization, Entropy bonus, Grad clipping, LR annealing | Classic Control, MuJoCo, DMControl |
| [SAC](https://github.com/jaehyun-jeong/100LinesRL/blob/master/sac.py) | 100 | Discrete, Continuous | Yes | Yes | Replay buffer, Twin Q-networks, Auto entropy tuning, Soft target update | Classic Control, MuJoCo, DMControl |
| [TD3](https://github.com/jaehyun-jeong/100LinesRL/blob/master/td3.py) | 100 | Continuous | No | No | Twin Q-networks, Delayed policy update, Target policy smoothing, EMA | Classic Control |
| [DQN](https://github.com/jaehyun-jeong/100LinesRL/blob/master/dqn.py) | 86 | Discrete | No | No | Replay buffer, Target network, ε-greedy | Classic Control |


![](./rewards.png)

| | | | |
|:---:|:---:|:---:|:---:|
| **Classic** | ![](./gifs/sac_CartPole-v1.gif) <br> sac/CartPole-v1 | ![](./gifs/sac_Pendulum-v1.gif) <br> sac/Pendulum-v1 | |
| **MuJoCo** | ![](./gifs/sac_HalfCheetah-v5.gif) <br> sac/HalfCheetah-v5 | ![](./gifs/sac_Hopper-v5.gif) <br> sac/Hopper-v5 | ![](./gifs/sac_Humanoid-v4.gif) <br> sac/Humanoid-v4 |
| **DMControl** | ![](./gifs/sac_cheetah-run-v0.gif) <br> sac/cheetah-run-v0 | ![](./gifs/sac_walker-walk-v0.gif) <br> sac/walker-walk-v0 | |

## Dependencies
1. [PyTorch](https://pytorch.org/) >= 2.10.0
2. [Gymnasium](https://gymnasium.farama.org/) >= 1.2.3 (with `mujoco` extra for MuJoCo environments)
3. [NumPy](https://numpy.org/) >= 2.4.2
4. [Shimmy](https://shimmy.farama.org/) >= 2.0.0 (with `dm-control` extra for DeepMind Control Suite)
5. [tqdm](https://tqdm.github.io/) (for progress bar)

### Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Works only with Python 3.
# e.g.
python3 dqn.py
python3 ppo.py --env "Hopper-v5"
python3 sac.py --env "dm_control/cheetah-run-v0"
python3 td3.py
```
