import copy

import gymnasium as gym
import torch as T, torch.nn as nn
import torch.nn.functional as F, torch.optim as optim
from types import SimpleNamespace

learning_rate, gamma, buffer_limit, batch_size = 0.0005, 0.98, 50000, 32

class ReplayBuffer:
    def __init__(self):
        self.s, self.a, self.r, self.sp, self.d = [T.zeros(buffer_limit, i) for i in [4, 1, 1, 4, 1]]  # state, action, reward, s prime, done
        self.a = self.a.int()
        self.buffer_ptr, self.buffer_size = 0, 0

    def push(self, transition):
        s, a, r, sp, d = transition
        for new, buffer in zip([s, [a], [r], sp, [d]], [self.s, self.a, self.r, self.sp, self.d]):
            buffer[self.buffer_ptr].copy_(T.as_tensor(new))
        self.buffer_ptr = (self.buffer_ptr + 1) % buffer_limit
        self.buffer_size = min(self.buffer_size + 1, buffer_limit)

    def sample(self, n):
        idx = T.randperm(len(self))[:n]
        return SimpleNamespace(s=self.s[idx], a=self.a[idx], r=self.r[idx], sp=self.sp[idx], done=self.d[idx])

    def __len__(self): return self.s.shape[0]

def sample_action(q, obs, eps):
    out = q(obs)
    coin = T.rand(())
    return (T.randint(0, 2, ()).item() if coin < eps else out.argmax().item())

def train(q, q_target, memory, optimizer):
    for i in range(10):
        mb = memory.sample(batch_size)
        q_a = q(mb.s).gather(1, mb.a)
        max_q_prime = q_target(mb.sp).max(1, True)[0]
        target = mb.r + gamma * max_q_prime * mb.done
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make("CartPole-v1")
    q = nn.Sequential(
        nn.Linear(4, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 2)
    )
    q_target = copy.deepcopy(q)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20

    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        eps = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        while not done:
            a = sample_action(q, T.tensor(s), eps)
            s_prime, r, done, _, _ = env.step(a)
            memory.push((s, a, r/100.0, s_prime, 0. if done else 1.))
            s = s_prime

            score += r

        if len(memory) > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(f"n_episode :{n_epi}, score : {score/print_interval:.1f}, " \
                  f"n_buffer : {len(memory)}, epsilon : {eps*100:.1f}%")
            score = 0.0

    env.close()

if __name__ == "__main__":
    main()
