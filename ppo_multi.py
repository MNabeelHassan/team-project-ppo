# ppo_multi.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Hyperparameters
# LEARNING_RATE = 3e-4
# GAMMA = 0.99
# LMBDA = 0.95
# EPS_CLIP = 0.2
# K_EPOCH = 4
LEARNING_RATE = 0.0015
GAMMA = 0.98
LMBDA = 0.95
EPS_CLIP = 0.15
K_EPOCH = 5
HIDDEN_UNITS = 128
ENTROPY_COEF = 0.01
REWARD_SCALE = {"step": -0.01, "goal": 1.0, "blocked": 1.0}


class PPOAgent(nn.Module):
    def __init__(self, state_dim=6, action_dim=4, hidden=128, lr=LEARNING_RATE):
        super(PPOAgent, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc_pi = nn.Linear(hidden, action_dim)
        self.fc_v = nn.Linear(hidden, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.memory = []

    def forward(self, x):
        return F.relu(self.fc1(x))

    def get_action(self, state):
        s = torch.tensor(state, dtype=torch.float32)
        x = self.forward(s)
        logits = self.fc_pi(x)
        pi = F.softmax(logits, dim=-1)
        dist = Categorical(pi)
        action = dist.sample()
        return action.item(), pi[action].item()

    def evaluate_actions(self, states, actions):
        # Ensure batch dimension
        if states.dim() == 1:
            states = states.unsqueeze(0)

        x = self.forward(states)
        logits = self.fc_pi(x)

        # Ensure logits is shape [batch, action_dim]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        pi = F.softmax(logits, dim=1)
        dist = Categorical(pi)

        # Actions: shape fix
        if actions.dim() == 2:
            actions = actions.squeeze()
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)

        logprob = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        value = self.fc_v(x).squeeze()

        return logprob, value, entropy, pi

    def put_data(self, transition):
        self.memory.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, sp_lst, prob_lst, done_lst = [], [], [], [], [], []

        for (s, a, r, sp, prob, done) in self.memory:
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            sp_lst.append(sp)
            prob_lst.append([prob])
            done_lst.append([0 if done else 1])

        self.memory = []

        return (
            torch.tensor(s_lst, dtype=torch.float32),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float32),
            torch.tensor(sp_lst, dtype=torch.float32),
            torch.tensor(done_lst, dtype=torch.float32),
            torch.tensor(prob_lst, dtype=torch.float32)
        )

    def train_net(self):
        if not self.memory:
            return {}

        s, a, r, sp, done_mask, prob_a = self.make_batch()

        # TD target
        with torch.no_grad():
            td_target = r + GAMMA * self.fc_v(self.forward(sp)) * done_mask

        # Advantage (GAE)
        delta = (td_target - self.fc_v(self.forward(s))).detach().numpy()
        adv_list = []
        adv = 0.0
        for d in delta[::-1]:
            adv = d[0] + GAMMA * LMBDA * adv
            adv_list.append([adv])
        adv_list.reverse()
        advantage = torch.tensor(adv_list, dtype=torch.float32).squeeze()

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "kl": 0}

        for _ in range(K_EPOCH):
            logprob, values, entropy, pi = self.evaluate_actions(s, a)

            old_logprob = torch.log(prob_a.squeeze() + 1e-10)
            ratio = torch.exp(logprob - old_logprob)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(values, td_target.squeeze())
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ==== SAFE KL COMPUTATION ====
            # pi shape: [batch, action_dim] OR [action_dim]
            if pi.dim() == 1:
                pi = pi.unsqueeze(0)

            act = a.squeeze()
            if act.dim() == 0:
                act = act.unsqueeze(0)

            # get probability of taken action
            new_probs = pi.gather(1, act.unsqueeze(1)).squeeze()

            old_probs = prob_a.squeeze()

            kl = torch.mean(
                old_probs * (torch.log(old_probs + 1e-10) -
                             torch.log(new_probs + 1e-10))
            )
            # ==============================

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            metrics["kl"] += kl.item()

        # Average over epochs
        for k in metrics:
            metrics[k] /= K_EPOCH

        return metrics
