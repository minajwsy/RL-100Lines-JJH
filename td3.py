import gymnasium as gym, torch as T, torch.nn as nn, torch.nn.functional as F
from types import SimpleNamespace

q_lr, pi_lr, gamma, tau, pi_noise, noise_clip, pi_freq, exploration_noise = 1e-3, 1e-3, 0.98, 0.005, 0.1, 0.5, 1, 0.1
learning_starts, total_timesteps, buffer_limit, batch_size, action_scale = 1_000, 1_000_000, 200_000, 256, T.tensor(2, dtype=T.float32)

class ReplayBuffer:
    def __init__(self):
        self.s, self.a, self.r, self.sp, self.d = [T.zeros(buffer_limit, i) for i in [3, 1, 1, 3, 1]]  # state, action, reward, s prime, done
        self.buffer_ptr, self.buffer_size = 0, 0

    def push(self, transition):
        s, a, r, sp, d = transition
        for new, buffer in zip([s, a, [r], sp, [d]], [self.s, self.a, self.r, self.sp, self.d]):
            buffer[self.buffer_ptr].copy_(T.as_tensor(new))
        self.buffer_ptr = (self.buffer_ptr + 1) % buffer_limit
        self.buffer_size = min(self.buffer_size + 1, buffer_limit)

    def sample(self, n):
        idx = T.randperm(len(self))[:n]
        return SimpleNamespace(s=self.s[idx], a=self.a[idx], r=self.r[idx], sp=self.sp[idx], done=self.d[idx])

    def __len__(self): return self.s.shape[0]

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, a): return self.fc(T.cat([x, a], 1))

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_mu = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x): return T.tanh(self.fc_mu(x)) * action_scale

def calc_target(pi_t, q1_t, q2_t, mb):
    with T.no_grad():
        clipped_noise = (T.randn_like(mb.a) * pi_noise).clamp(
            -noise_clip, noise_clip
        ) * action_scale
        next_state_action = (pi_t(mb.sp) + clipped_noise).clamp(
            env.action_space.low[0], env.action_space.high[0]
        )
        q1_next_target, q2_next_target = [q_t(mb.sp, next_state_action) for q_t in [q1_t, q2_t]]
        min_q_next_target = T.min(q1_next_target, q2_next_target)
        return (mb.r + gamma * mb.done * min_q_next_target).view(-1)

if __name__ == "__main__":
    env, memory = gym.make('Pendulum-v1'), ReplayBuffer()
    q1, q2, q1_t, q2_t, pi, pi_t = QNet(), QNet(), QNet(), QNet(), PolicyNet(), PolicyNet()
    pi_t.load_state_dict(pi.state_dict())
    pi_optimizer = T.optim.Adam(list(pi.parameters()), lr=pi_lr)
    [qt.load_state_dict(q.state_dict()) for qt, q in [(q1_t, q1), (q2_t, q2)]]
    q_optimizer = T.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=q_lr)

    s, _ = env.reset()
    score = 0.
    for n_epi in range(10000):
        for num_step in range(200):
            if len(memory) < learning_starts:
                a = env.action_space.sample()
            else:
                with T.no_grad():
                    a = pi(T.Tensor(s))
                    a += T.normal(0, action_scale * exploration_noise)  # Smoothing q funciton
                    a = a.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            sp, r, done, _, _ = env.step(a)
            memory.push((s, a, r/10.0, sp, 0. if done else 1.))
            score += r
            s = sp

            if len(memory) > learning_starts:
                mb = memory.sample(batch_size)
                next_q_value = calc_target(pi_t, q1_t, q2_t, mb)
                q1_a_value, q2_a_value = [q(mb.s, mb.a).view(-1) for q in [q1, q2]]
                q_loss = F.mse_loss(q1_a_value, next_q_value) + F.mse_loss(q2_a_value, next_q_value)

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                if (num_step+1) % pi_freq == 0:  # Delayed Policy Update
                    pi_loss = -q1(mb.s, pi(mb.s)).mean()
                    pi_optimizer.zero_grad()
                    pi_loss.backward()
                    pi_optimizer.step()

                    for nn, nn_t in zip([pi, q1, q2], [pi_t, q1_t, q2_t]):  # EMA
                        for param, param_t in zip(nn.parameters(), nn_t.parameters()):
                            param_t.data.copy_(tau * param.data + (1 - tau) * param_t.data)
            if done: break

        if n_epi % 20 == 0 and n_epi != 0:
            print(f"Episode: {n_epi}, Avg Score: {score/20:.1f}")
            score = 0.0

    env.close()
