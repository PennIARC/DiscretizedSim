import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared Feature Extractor
        # We use Tanh activations which generally work better for continuous control
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor Head (The Pilot)
        # Outputs Mean (mu) and Log Standard Deviation (log_std)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Enforce [-1, 1] range for mean
        )
        
        # Learnable Standard Deviation (starts at 0.5 approx)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)
        
        # Critic Head (The Coach)
        # Estimates "How good is this state?"
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        feats = self.features(x)
        return feats

    def get_action(self, x):
        feats = self.features(x)
        mean = self.actor_mean(feats)
        std = torch.exp(self.actor_log_std)
        
        dist = Normal(mean, std)
        action = dist.sample()
        
        # Sum log probs for multi-dimensional actions (independent dimensions)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, self.critic(feats)

    def evaluate(self, x, action):
        feats = self.features(x)
        mean = self.actor_mean(feats)
        std = torch.exp(self.actor_log_std)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, self.critic(feats), entropy

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.device = torch.device("cpu") # Use CPU for stability unless you have CUDA setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"PPO: Using CUDA Backend ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps") # Mac Metal Acceleration
            print("PPO: Using Apple MPS Backend")

        # Initialize Network
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
        # Rollout Buffer
        self.buffer = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'dones': [], 'values': []
        }

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, log_prob, value = self.policy.get_action(state)
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def store(self, state, action, log_prob, reward, done, value):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['values'].append(value)

    def update(self):
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        rewards = self.buffer['rewards']
        dones = self.buffer['dones']
        values = torch.FloatTensor(np.array(self.buffer['values'])).to(self.device).squeeze()

        # Calculate GAE (Generalized Advantage Estimation)
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalizing returns helps stability significantly
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        advantages = returns - values.detach()

        # PPO Update Loop
        for _ in range(self.K_epochs):
            # Evaluate old actions in new network
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            state_values = state_values.squeeze()
            
            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final Loss = -ActorLoss + 0.5*CriticLoss - 0.01*Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Clear buffer
        for k in self.buffer: self.buffer[k] = []