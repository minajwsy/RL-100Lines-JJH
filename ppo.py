import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

# Hyperparameters
lr, gamma, lmbda, eps_clip, K_epoch, T_horizon = 0.0005, 0.98, 0.95, 0.1, 3, 20
mb_size = 5

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, is_cts):
        super(PPO, self).__init__()

        self.is_cts = is_cts
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.reset_batch()
        self.backbone = nn.Sequential(nn.Linear(state_dim, 256), nn.Tanh())
        if self.is_cts:
            self.mu_head = nn.Linear(256, action_dim)
            self.log_std = nn.Parameter(T.zeros(action_dim))
        else:
            self.pi_head  = nn.Sequential(nn.Linear(256, action_dim), nn.Softmax(dim=-1))

        self.v_head   = nn.Sequential(nn.Linear(256, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def pi(self, x, action=None):
        if self.is_cts:
            mu = T.tanh(self.mu_head(self.backbone(x)))
            std = T.exp(self.log_std)
            dist = Normal(mu, std)
            if action == None: action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            prob = self.pi_head(self.backbone(x))
            dist = Categorical(prob)
            if action == None: action = dist.sample()
            if action is not None and action.dim() > 1:
                action_in = action.squeeze(-1)
            else:
                action_in = action
            log_prob = dist.log_prob(action_in).unsqueeze(-1)
        return action, log_prob

    def v(self, x): return self.v_head(self.backbone(x))

    def reset_batch(self):
        self.s = T.empty(0, self.state_dim, dtype=T.float)
        if self.is_cts:
            self.a = T.empty(0, self.action_dim, dtype=T.float)
        else:
            self.a = T.empty(0, 1, dtype=T.int)
        self.r = T.empty(0, 1, dtype=T.float)
        self.s_prime = T.empty(0, self.state_dim, dtype=T.float)
        self.log_prob_a = T.empty(0, 1, dtype=T.float)
        self.done = T.empty(0, 1, dtype=T.float)

    def push(self, transition):
        s, a, r, s_prime, log_prob_a, done = transition
        self.s = T.cat((self.s, T.tensor(s, dtype=T.float).unsqueeze(0)), 0)
        self.a = T.cat((self.a, a.unsqueeze(0)), 0)
        self.r = T.cat((self.r, T.tensor([r], dtype=T.float).unsqueeze(0)), 0)
        self.s_prime = T.cat((self.s_prime, T.tensor(s_prime, dtype=T.float).unsqueeze(0)), 0)
        self.log_prob_a = T.cat((self.log_prob_a, T.tensor([log_prob_a], dtype=T.float).unsqueeze(0)), 0)
        self.done = T.cat((self.done, T.tensor([0 if done else 1], dtype=T.float).unsqueeze(0)), 0)

    def train_net(self):
        td_target = self.r + gamma * self.v(self.s_prime) * self.done
        delta = td_target - self.v(self.s)
        advantages = T.empty(0, 1, dtype=T.float)
        advantage = 0.0
        for delta_t in delta.detach().flatten().flip(0):
            with T.no_grad(): advantage = gamma * lmbda * advantage + delta_t
            advantages = T.cat((T.tensor([[advantage]]), advantages), 0)

        for i in range(K_epoch):
            perm_inds = T.randperm(self.__len__())
            for start in range(0, self.__len__(), mb_size):
                inds = perm_inds[start:start + mb_size]

                _, log_prob_a = self.pi(self.s[inds], self.a[inds])
                ratio = T.exp(log_prob_a - self.log_prob_a[inds])

                surr1 = ratio * advantages[inds]
                surr2 = T.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages[inds]
                loss = -T.min(surr1, surr2) + F.smooth_l1_loss(self.v(self.s[inds]) , td_target[inds].detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        self.reset_batch()

    def __len__(self): return self.s.shape[0]

def main():
    env = gym.make('CartPole-v1')
    is_cts, state_dim = isinstance(env.action_space, gym.spaces.Box), env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if is_cts else env.action_space.n
    model = PPO(state_dim, action_dim, is_cts)
    score, print_interval = 0.0, 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                a, log_prob = model.pi(T.from_numpy(s).float().unsqueeze(0))
                if is_cts:
                    a = a.squeeze(0)
                    a_input = a.detach().numpy()
                else:
                    a_input = a.item()
                s_prime, r, done, truncated, info = env.step(a_input)
                model.push((s, a, r/100.0, s_prime, log_prob.item(), done))
                s = s_prime
                score += r

                if done or truncated: break
            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print(f"# of episode :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
