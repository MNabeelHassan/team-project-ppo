# ppo_hero_villan.py
# Co-evolution PPO Agent for Hero vs Villain Grid

import torch
import torch.nn as nn #module for neural network layers
import torch.nn.functional as F #inlude activation func and loss functions
import torch.optim as optim

class PPOAgent(nn.Module):
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=256, lr=0.0005, gamma=0.98, lmbda=0.95, eps_clip=0.1, K_epoch=3):
        super(PPOAgent, self).__init__()

        # Store hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch

        # Network layers
        #state input -> hidden layer -> policy and value output layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, action_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) #adam used for optization 
        self.data = []

    def pi(self, x, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        return self.fc_v(x)

    def put_data(self, transition):
        self.data.append(transition)
    #store for each transitiom 
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])

        s = torch.tensor(s_lst, dtype=torch.float32)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float32)
        done_mask = torch.tensor(done_lst, dtype=torch.float32)
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float32)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        if len(self.data) == 0:
            return

        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for _ in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
