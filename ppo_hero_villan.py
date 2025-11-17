import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class HeroVillainPPO(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        K_epoch: int = 4,
    ):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_agent = nn.Linear(hidden_dim, 4)
        self.pi_adv = nn.Linear(hidden_dim, 4)
        self.v_head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.K_epoch = K_epoch

        self.memory = []

    def _forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def pi(self, x: torch.Tensor):
        z = self._forward_shared(x)
        probs_agent = F.softmax(self.pi_agent(z), dim=-1)
        probs_adv = F.softmax(self.pi_adv(z), dim=-1)
        return probs_agent, probs_adv

    def v(self, x: torch.Tensor) -> torch.Tensor:
        z = self._forward_shared(x)
        return self.v_head(z)

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs_agent, probs_adv = self.pi(state_tensor)

        dist_agent = Categorical(probs_agent)
        dist_adv = Categorical(probs_adv)

        action_agent = dist_agent.sample()
        action_adv = dist_adv.sample()

        logprob_agent = dist_agent.log_prob(action_agent)
        logprob_adv = dist_adv.log_prob(action_adv)

        return (
            action_agent.item(),
            action_adv.item(),
            logprob_agent.item(),
            logprob_adv.item(),
        )

    def put_data(self, transition):
        self.memory.append(transition)

    def make_batch(self):
        s_lst, a_agent_lst, a_adv_lst, r_lst, s_prime_lst, done_lst, logprob_agent_lst, logprob_adv_lst = zip(
            *self.memory
        )

        s = torch.tensor(s_lst, dtype=torch.float32)
        a_agent = torch.tensor(a_agent_lst)
        a_adv = torch.tensor(a_adv_lst)
        r = torch.tensor(r_lst, dtype=torch.float32).unsqueeze(1)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float32)
        done_mask = torch.tensor(done_lst, dtype=torch.float32).unsqueeze(1)
        logprob_agent = torch.tensor(logprob_agent_lst, dtype=torch.float32).unsqueeze(1)
        logprob_adv = torch.tensor(logprob_adv_lst, dtype=torch.float32).unsqueeze(1)

        self.memory.clear()
        return s, a_agent, a_adv, r, s_prime, done_mask, logprob_agent, logprob_adv

    def train_net(self):
        if not self.memory:
            return

        s, a_agent, a_adv, r, s_prime, done_mask, logprob_agent_old, logprob_adv_old = self.make_batch()

        with torch.no_grad():
            td_target = r + self.gamma * self.v(s_prime) * (1 - done_mask)
            delta = td_target - self.v(s)
            advantage = torch.zeros_like(r)
            advantage_sum = 0.0
            for t in reversed(range(len(delta))):
                advantage_sum = delta[t] + self.gamma * self.gae_lambda * (1 - done_mask[t]) * advantage_sum
                advantage[t] = advantage_sum

        for _ in range(self.K_epoch):
            probs_agent, probs_adv = self.pi(s)
            dist_agent = Categorical(probs_agent)
            dist_adv = Categorical(probs_adv)

            logprob_agent = dist_agent.log_prob(a_agent)
            logprob_adv = dist_adv.log_prob(a_adv)

            logprob_agent = logprob_agent.unsqueeze(1)
            logprob_adv = logprob_adv.unsqueeze(1)

            logprob_old = logprob_agent_old + logprob_adv_old
            logprob_now = logprob_agent + logprob_adv
            ratio = torch.exp(logprob_now - logprob_old)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            policy_loss = -torch.min(surr1, surr2)

            value_loss = F.mse_loss(self.v(s), td_target)
            loss = policy_loss.mean() + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

