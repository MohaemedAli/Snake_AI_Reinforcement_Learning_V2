import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import DuelingQNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """Double DQN + Dueling network + replay buffer."""

    def __init__(
        self,
        state_dim,
        action_dim=3,
        learning_rate=0.001,
        gamma=0.95,
        batch_size=64,
        replay_capacity=100000,
        target_update_interval=500,
        seed=None,
        device=None
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size

        self.target_update_interval = target_update_interval

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Online network
        self.policy_net = DuelingQNetwork(
            state_dim, action_dim
        ).to(self.device)

        # Target network
        self.target_net = DuelingQNetwork(
            state_dim, action_dim
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            capacity=replay_capacity,
            seed=seed
        )

        self.optimization_steps = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, state, epsilon=0.0):

        # Exploration
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        # Exploitation
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def q_values(self, state):
        """Q-values of every action (used by the AI viewer)."""

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            return self.policy_net(state_tensor).squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):

        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )

        # Current Q values
        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN:
        # Online network selects the next action,
        # target network evaluates it.
        with torch.no_grad():

            next_actions = self.policy_net(
                next_states
            ).argmax(dim=1, keepdim=True)

            next_q = self.target_net(
                next_states
            ).gather(1, next_actions).squeeze(1)

            target_q = (
                rewards
                + self.gamma * (1.0 - dones) * next_q
            )

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.optimization_steps += 1

        # Periodically update target network.
        if self.optimization_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(loss.item())
