import copy, numpy as np
import gymnasium as gym, gymnasium.wrappers as wrappers
import torch as T, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from types import SimpleNamespace
from tqdm import tqdm
import dqn_config as conf

def make_envs(env_id, n_envs):
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])
    envs = wrappers.vector.FlattenObservation(wrappers.vector.RecordEpisodeStatistics(envs))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        f"DQN requires Discrete action space. {env_id} is not supported."
    return envs, envs.single_observation_space.shape[0], envs.single_action_space.n

class Buffer:
    def __init__(self, s_dim, a_dim):
        self.b, self.p, self.bsize = [T.zeros(conf.buffer_limit, i) for i in [s_dim, 1, 1, s_dim, 1]], 0, 0  # state, action, reward, s prime, done

    def push(self, transition):
        for buf, val in zip(self.b, transition): buf[self.p].copy_(T.as_tensor(val, dtype=T.float32).view_as(buf[self.p]))
        self.p, self.bsize = (self.p + 1) % conf.buffer_limit, min(self.bsize + 1, conf.buffer_limit)

    def sample(self, n):
        idx = T.randint(0, self.bsize, (n,))
        return SimpleNamespace(**dict(zip(['s', 'a', 'r', 'sp', 'done'], [b[idx].to(conf.device) for b in self.b])))

def sample_action(q, obs, eps, n_envs, a_dim):
    with T.no_grad(): out = q(obs)
    coins = T.rand(n_envs, device=conf.device)
    random_actions = T.randint(0, a_dim, (n_envs,), device=conf.device)
    greedy_actions = out.argmax(dim=-1)
    actions = T.where(coins < eps, random_actions, greedy_actions)
    return actions.cpu().numpy()

class QNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, a_dim)
        )

    def forward(self, x):
        return self.net(x)

def train(q, q_target, memory, optimizer):
    mb = memory.sample(conf.batch_size)
    q_a = q(mb.s).gather(1, mb.a.long())
    with T.no_grad(): max_q_prime = q_target(mb.sp).max(1, True)[0]
    target = mb.r + conf.gamma * max_q_prime * mb.done
    loss = F.smooth_l1_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    envs, s_dim, a_dim = make_envs(conf.env_name, conf.n_envs)
    q = QNet(s_dim, a_dim).to(conf.device)
    q_target = copy.deepcopy(q).to(conf.device)
    memory = Buffer(s_dim, a_dim)
    optimizer = optim.Adam(q.parameters(), lr=conf.lr_q)

    best_score, score, print_interval, n_epi, total_step = -np.inf, 0.0, 20, 0, conf.max_timesteps // conf.n_envs
    s, _ = envs.reset()
    for n_step in tqdm(range(total_step), unit_scale=conf.n_envs, unit="step"):
        fraction = min(1.0, (n_step * conf.n_envs) / (conf.max_timesteps * conf.eps_decay_frac))
        eps = conf.eps_start - fraction * (conf.eps_start - conf.eps_end)

        a = sample_action(q, T.tensor(s, dtype=T.float32).to(conf.device), eps, conf.n_envs, a_dim)
        sp, r, done, trunc, info = envs.step(a)

        for i in range(conf.n_envs):
            is_done = done[i] or trunc[i]
            actual_s_prime = info['final_observation'][i] if is_done and 'final_observation' in info else sp[i]
            memory.push((s[i], a[i], r[i], actual_s_prime, 0. if done[i] else 1.))
        s = sp

        if "episode" in info and "_episode" in info:
            for i in np.where(info["_episode"])[0]:
                if (episode_score := float(np.array(info["episode"]['r'][i]).item())) > best_score:
                    best_score = episode_score
                score += episode_score
                n_epi += 1
                if n_epi % print_interval == 0:
                    tqdm.write(f"step {(n_step+1)*conf.n_envs} | episode {n_epi} | avg score {score/print_interval:.1f} | best score {best_score:.1f} | eps {eps*100:.1f}%")
                    score = 0.0

        if memory.bsize > conf.learning_starts and (n_step + 1) % max(1, conf.train_freq // conf.n_envs) == 0:
            train(q, q_target, memory, optimizer)

        if (n_step + 1) % max(1, conf.target_freq // conf.n_envs) == 0:
            q_target.load_state_dict(q.state_dict())
    envs.close()

if __name__ == "__main__":
    main()
