import gymnasium as gym, torch as T, torch.nn as nn, torch.nn.functional as F, sac_config as conf, numpy as np, shimmy
from torch.distributions import Normal, Categorical
from types import SimpleNamespace
from tqdm import tqdm

def make_env(env_id):
    env = gym.wrappers.FlattenObservation(gym.wrappers.RecordEpisodeStatistics(gym.make(env_id)))
    if (is_cts := isinstance(env.action_space, gym.spaces.Box)):
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    return env, is_cts, env.observation_space.shape[0], env.action_space.shape[0] if is_cts else env.action_space.n

def optim_step(opt, loss): opt.zero_grad(), loss.mean().backward(), opt.step()

class Buffer:
    def __init__(self, s_dim, a_dim, is_cts):
        self.b, self.p, self.bsize = [T.zeros(conf.buffer_limit, i) for i in [s_dim, a_dim if is_cts else 1, 1, s_dim, 1]], 0, 0  # state, action, reward, s prime, done

    def push(self, transition):
        for buf, val in zip(self.b, transition): buf[self.p].copy_(T.as_tensor(val))
        self.p, self.bsize = (self.p + 1) % conf.buffer_limit, min(self.bsize + 1, conf.buffer_limit)

    def sample(self, n):
        idx = (T.randint(0, self.bsize, (n,)))
        s, a, r, sp, done = [b[idx].to(conf.device) for b in self.b]
        return SimpleNamespace(s=s, a=a, r=r, sp=sp, done=done)

class PolicyNet(nn.Module):
    def __init__(self, s_dim, a_dim, is_cts):
        super().__init__()
        self.is_cts, self.bb = is_cts, nn.Sequential(nn.Linear(s_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        if is_cts: self.mu, self.log_std = nn.Linear(256, a_dim), nn.Sequential(nn.Linear(256, a_dim))
        else: self.pi = nn.Sequential(nn.Linear(256, a_dim))
        self.log_alpha = nn.Parameter(T.tensor(conf.init_alpha, device=conf.device).log())
        self.to(conf.device)
        self.optimizer = T.optim.Adam([p for n, p in self.named_parameters() if 'alpha' not in n], lr=conf.lr_pi)
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=conf.lr_alpha)

    def forward(self, x):
        bb = self.bb(x)
        if self.is_cts:
            a = (dist := Normal(self.mu(bb), T.clamp(self.log_std(bb), -20, 2).exp())).rsample()
            return T.tanh(a), (dist.log_prob(a) - 2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=-1, keepdim=True), None
        prob, log_p = F.softmax((logits := self.pi(bb)), dim=-1), F.log_softmax(logits, dim=-1)
        return Categorical(prob).sample(), log_p, prob

    def train_net(self, q1, q2, mb):  # mb: mini-batch
        a, log_p, prob = self.forward(mb.s)
        alpha = self.log_alpha.exp().detach()
        [q.requires_grad_(False) for q in [q1, q2]]
        q_val = T.min(q1(mb.s, a), q2(mb.s, a)) if self.is_cts else T.min(q1(mb.s), q2(mb.s)).detach()
        loss = -(q_val - alpha * log_p) if self.is_cts else -(prob * (q_val - alpha * log_p)).sum(-1)
        optim_step(self.optimizer, loss)
        [q.requires_grad_(True) for q in [q1, q2]]
        alpha_loss = -(self.log_alpha * ((log_p if self.is_cts else (prob * log_p).sum(-1)) + conf.target_entropy).detach())
        optim_step(self.alpha_optimizer, alpha_loss)

class QNet(nn.Module):
    def __init__(self, s_dim, a_dim, is_cts):
        super().__init__()
        self.is_cts, in_dim, out_dim = is_cts, s_dim + (a_dim if is_cts else 0), 1 if is_cts else a_dim
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))
        self.to(conf.device)
        self.optimizer = T.optim.Adam(self.parameters(), lr=conf.lr_q)

    def forward(self, s, a=None): return self.net(T.cat([s, a], dim=1) if self.is_cts else s)

    def train_net(self, tgt, mb):  # mb: mini-batch
        optim_step(self.optimizer, F.mse_loss(self(mb.s, mb.a) if self.is_cts else self(mb.s).gather(1, mb.a.long()), tgt))

    def soft_update(self, tgt):
        for pt, p in zip(tgt.parameters(), self.parameters()): pt.data.copy_(pt.data * (1 - conf.tau) + p.data * conf.tau)

def calc_target(pi, q1_t, q2_t, mb, is_cts):  # mb: mini-batch
    with T.no_grad():
        ap, log_p, prob = pi(mb.sp)
        v = (T.min(q1_t(mb.sp, ap), q2_t(mb.sp, ap)) if is_cts else T.min(q1_t(mb.sp), q2_t(mb.sp))) - pi.log_alpha.exp() * log_p
        return mb.r + conf.gamma * mb.done * (v if is_cts else (prob * v).sum(-1, keepdim=True))

if __name__ == '__main__':
    env, is_cts, s_dim, a_dim = make_env(conf.env_name)
    buf, pi, s, score, n_epi, print_interval = Buffer(s_dim, a_dim, is_cts), PolicyNet(s_dim, a_dim, is_cts), env.reset()[0], 0.0, 0, 20
    q1, q2, q1_t, q2_t = [QNet(s_dim, a_dim, is_cts) for _ in range(4)]
    [qt.load_state_dict(q.state_dict()) for qt, q in [(q1_t, q1), (q2_t, q2)]]
    for n_step in tqdm(range(conf.max_timesteps)):
        with T.no_grad(): a = T.tensor(env.action_space.sample()) if n_step < conf.learning_starts else pi(T.from_numpy(s).float().to(conf.device))[0]
        sp, r, done, trunc, info = env.step(a.cpu().numpy() if is_cts else a.item())
        buf.push((s, a.detach(), r, sp, 0. if done else 1.))
        if done or trunc:
            n_epi, score = n_epi + 1, score + info['episode']['r']
            if n_epi % print_interval == 0:
                tqdm.write(f"step {n_step+1} episode {n_epi} avg score {score/print_interval:.1f}")
                score = 0.0
            s, _ = env.reset()
        else: s = sp
        if n_step > conf.learning_starts:
            target = calc_target(pi, q1_t, q2_t, mb := buf.sample(conf.batch_size), is_cts)
            [q.train_net(target, mb) for q in [q1, q2]]
            pi.train_net(q1, q2, mb)
            if (n_step+1) % conf.target_freq == 0:
                [q.soft_update(q_t) for q, q_t in [(q1, q1_t), (q2, q2_t)]]
    env.close()
