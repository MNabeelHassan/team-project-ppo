import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOAgent(nn.Module):
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 4,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        K_epoch: int = 4,
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, action_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.K_epoch = K_epoch

        self.data = []

    def pi(self, x: torch.Tensor, softmax_dim: int = -1) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(s_lst, dtype=torch.float32)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst, dtype=torch.float32)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float32)
        done_mask = torch.tensor(done_lst, dtype=torch.float32)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float32)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        if not self.data:
            return

        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for _ in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.gae_lambda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

