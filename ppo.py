import gymnasium as gym, torch as T, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, numpy as np, ppo_config as conf, shimmy
from torch.distributions import Categorical, Normal
from tqdm import tqdm

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(env_id):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    is_cts, s_dim = isinstance(env.action_space, gym.spaces.Box), env.observation_space.shape[0]
    a_dim = env.action_space.shape[0] if is_cts else env.action_space.n
    return env, is_cts, s_dim, a_dim

class PPO(nn.Module):
    def __init__(self, s_dim, a_dim, is_cts):
        super().__init__()
        self.is_cts, self.s_dim, self.a_dim = is_cts, s_dim, a_dim
        self.buf, self.p = [T.zeros(conf.T_horizon, i, device=conf.device) for i in [self.s_dim, self.a_dim if self.is_cts else 1, 1, self.s_dim, 1, 1]], 0

        self.pi_net, self.v_net = [nn.Sequential(
            layer_init(nn.Linear(s_dim, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 256)), nn.ReLU()
        ) for _ in range(2)]
        if self.is_cts:
            self.mu_head = layer_init(nn.Linear(256, a_dim), std=0.01)
            self.log_std = nn.Parameter(T.zeros(a_dim))
        else:
            self.pi_head = nn.Sequential(layer_init(nn.Linear(256, a_dim), std=0.01))
        self.v_head = layer_init(nn.Linear(256, 1), std=1.0)
        self.to(conf.device)
        self.optimizer = optim.Adam(self.parameters(), lr=conf.lr)

    def pi(self, x, a=None):
        if self.is_cts:
            dist = Normal(self.mu_head(self.pi_net(x)), T.exp(self.log_std))
            if a is None: a = dist.sample()
            return a.squeeze(0), a.squeeze(0).detach().cpu().numpy(), dist.log_prob(a).sum(dim=-1, keepdim=True), dist

        dist = Categorical(logits=self.pi_head(self.pi_net(x)))
        if a is None: a = dist.sample()
        return a, a.item() if a.numel() == 1 else None, dist.log_prob(a.squeeze(-1).long() if a.dim() > 1 else a.long()).unsqueeze(-1), dist

    def v(self, x): return self.v_head(self.v_net(x))

    def push(self, transition):
        for val, buf in zip(transition, self.buf): buf[self.p].copy_(T.as_tensor(val, device=conf.device))
        self.p = (self.p + 1) % conf.T_horizon

    def train_net(self):
        s, a, r, sp, log_p, not_d = self.buf
        with T.no_grad():  # Calculate GAE
            vals = self.v(s)
            deltas = r + conf.gamma * T.cat([vals[1:], self.v(sp[-1].unsqueeze(0))]) * not_d - vals
            advantages, gae = T.zeros_like(r), 0
            for t in reversed(range(len(r))):
                advantages[t] = gae = deltas[t] + conf.gamma * conf.lmbda * not_d[t] * gae
            returns = advantages + vals
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(conf.K_epoch):  # Update Parameters
            for inds in T.randperm(len(r), device=conf.device).split(conf.mb_size):
                _, _, log_prob_a, dist = self.pi(s[inds], a[inds])
                ratio = T.exp(log_prob_a - log_p[inds])

                loss = -T.min(ratio * advantages[inds], T.clamp(ratio, 1-conf.eps_clip, 1+conf.eps_clip) * advantages[inds]).mean() \
                    + conf.vf_coef * F.smooth_l1_loss(self.v(s[inds]), returns[inds]) - conf.ent_coef * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), conf.max_grad_norm)
                self.optimizer.step()

if __name__ == '__main__':
    env, is_cts, s_dim, a_dim = make_env(conf.env_name)
    model = PPO(s_dim, a_dim, is_cts)
    s, score, n_epi, print_interval = env.reset()[0], 0.0, 0, 20
    for n_step in tqdm(range(conf.max_timesteps)):
        model.optimizer.param_groups[0]['lr'] = conf.lr * (1 - n_step / conf.max_timesteps)
        a, a_in, log_prob, _ = model.pi(T.from_numpy(s).float().unsqueeze(0).to(conf.device))
        sp, r, done, trunc, info = env.step(a_in)
        model.push((s, a, r, sp, log_prob.item(), float(not done)))
        if done or trunc:
            n_epi, score = n_epi + 1, score + info['episode']['r']
            if (n_epi+1) % print_interval == 0:
                tqdm.write(f"step {n_step+1} episode {n_epi+1} avg score {score/print_interval:.1f} lr {model.optimizer.param_groups[0]['lr']:.6f}")
                score = 0.0
            s, _ = env.reset()
        else: s = sp
        if (n_step+1) % conf.T_horizon == 0: model.train_net()
    env.close()
