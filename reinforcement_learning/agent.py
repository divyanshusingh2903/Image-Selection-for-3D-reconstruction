import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        experiences = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
        return experiences
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network for Deep Q-Learning."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize Q-Network.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super(QNetwork, self).__init__()
        
        # Define network architecture
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network agent for camera view selection."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 64,
        update_every: int = 4,
        device: str = 'cpu'
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            update_every: How often to update the target network
            device: Device to use for computation
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Initialize Q-Networks (local and target)
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add experience to memory and learn if it's time.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self._learn()
    
    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            eps: Epsilon value for exploration
            
        Returns:
            Selected action
        """
        # Convert state to tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set model to evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        # Set model back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def _learn(self):
        """Update value parameters using given batch of experience tuples."""
        # Sample batch from replay buffer
        experiences = self.memory.sample(self.batch_size)
        
        # Extract batch data
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        # Get Q values for current states
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Get max predicted Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._soft_update()
    
    def _soft_update(self):
        """Soft update target network parameters."""
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")