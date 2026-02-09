import config
import gymnasium as gym, torch as T, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import shimmy
from torch.distributions import Categorical, Normal
from tqdm import tqdm
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    return env

class PPO(nn.Module):
    def __init__(self, s_dim, a_dim, is_cts):
        super(PPO, self).__init__()
        self.is_cts, self.s_dim, self.a_dim= is_cts, s_dim, a_dim
        self.reset_batch()

        self.pi_backbone, self.v_backbone = [nn.Sequential(
            layer_init(nn.Linear(s_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh()
        ) for _ in range(2)]
        if self.is_cts:
            self.mu_head = layer_init(nn.Linear(256, a_dim), std=0.01)
            self.log_std = nn.Parameter(T.zeros(a_dim))
        else:
            self.pi_head  = nn.Sequential(layer_init(nn.Linear(256, a_dim), std=0.01), nn.Softmax(dim=-1))
        self.v_head   = layer_init(nn.Linear(256, 1), std=1.0)
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)

    def pi(self, x, action=None):
        if self.is_cts:
            mu = self.mu_head(self.pi_backbone(x))
            std = T.exp(self.log_std)
            dist = Normal(mu, std)
            if action == None: action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            prob = self.pi_head(self.pi_backbone(x))
            dist = Categorical(prob)
            if action == None: action = dist.sample()
            if action is not None and action.dim() > 1:
                action_in = action.squeeze(-1)
            else:
                action_in = action
            log_prob = dist.log_prob(action_in).unsqueeze(-1)
        return action, log_prob

    def v(self, x): return self.v_head(self.v_backbone(x))

    def reset_batch(self):
        self.s = T.empty(0, self.s_dim, dtype=T.float)
        if self.is_cts:
            self.a = T.empty(0, self.a_dim, dtype=T.float)
        else:
            self.a = T.empty(0, 1, dtype=T.int)
        self.r = T.empty(0, 1, dtype=T.float)
        self.s_prime = T.empty(0, self.s_dim, dtype=T.float)
        self.log_prob_a = T.empty(0, 1, dtype=T.float)
        self.d_mask= T.empty(0, 1, dtype=T.float)

    def push(self, transition):
        s, a, r, s_prime, log_prob_a, d_mask = transition
        self.s = T.cat((self.s, T.tensor(s, dtype=T.float).unsqueeze(0)), 0)
        self.a = T.cat((self.a, a.unsqueeze(0)), 0)
        self.r = T.cat((self.r, T.tensor([r], dtype=T.float).unsqueeze(0)), 0)
        self.s_prime = T.cat((self.s_prime, T.tensor(s_prime, dtype=T.float).unsqueeze(0)), 0)
        self.log_prob_a = T.cat((self.log_prob_a, T.tensor([log_prob_a], dtype=T.float).unsqueeze(0)), 0)
        self.d_mask= T.cat((self.d_mask, T.tensor([d_mask], dtype=T.float).unsqueeze(0)), 0)

    def train_net(self):
        td_target = self.r + config.gamma * self.v(self.s_prime) * self.d_mask
        delta = td_target - self.v(self.s)
        advantages = T.empty(0, 1, dtype=T.float)
        advantage = 0.0
        masks = self.d_mask.detach().flatten().flip(0)
        for i, delta_t in enumerate(delta.detach().flatten().flip(0)):
            with T.no_grad(): advantage = config.gamma * config.lmbda * advantage * masks[i] + delta_t
            advantages = T.cat((T.tensor([[advantage]]), advantages), 0)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for i in range(config.K_epoch):
            perm_inds = T.randperm(self.__len__())
            for start in range(0, self.__len__(), config.mb_size):
                inds = perm_inds[start:start + config.mb_size]

                _, log_prob_a = self.pi(self.s[inds], self.a[inds])
                ratio = T.exp(log_prob_a - self.log_prob_a[inds])

                surr1 = ratio * advantages[inds]
                surr2 = T.clamp(ratio, 1-config.eps_clip, 1+config.eps_clip) * advantages[inds]
                loss = -T.min(surr1, surr2) + F.smooth_l1_loss(self.v(self.s[inds]) , td_target[inds].detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
                self.optimizer.step()
        self.reset_batch()

    def __len__(self): return self.s.shape[0]

def main():
    env = make_env(config.env_name)
    is_cts, s_dim = isinstance(env.action_space, gym.spaces.Box), env.observation_space.shape[0]
    a_dim = env.action_space.shape[0] if is_cts else env.action_space.n
    model = PPO(s_dim, a_dim, is_cts)
    total_step = 1_000_000

    s, _ = env.reset()
    score, n_epi, print_interval = 0.0, 0, 20

    pbar = tqdm(range(total_step))
    for n_step in pbar:
        model.optimizer.param_groups[0]['lr'] = config.lr * (1 - n_step / total_step)
        a, log_prob = model.pi(T.from_numpy(s).float().unsqueeze(0))
        if is_cts:
            a = a.squeeze(0)
            a_input = a.detach().numpy()
        else:
            a_input = a.item()
        s_prime, r, terminated, truncated, info = env.step(a_input)
        d_mask = 0.0 if terminated else 1.0
        done = terminated or truncated
        model.push((s, a, r, s_prime, log_prob.item(), d_mask))
        s = s_prime

        if done:
            score += info['episode']['r']
            if (n_epi+1) % print_interval == 0:
                tqdm.write(f"step {n_step+1} episode {n_epi+1} avg score {score/print_interval:.1f}")
                score = 0.0
            s, _ = env.reset()
            n_epi += 1

        if n_step % config.T_horizon == 0:
            model.train_net()

        pbar.set_postfix(episode=n_epi, lr=f"{model.optimizer.param_groups[0]['lr']:.6f}")

    env.close()

if __name__ == '__main__':
    main()
