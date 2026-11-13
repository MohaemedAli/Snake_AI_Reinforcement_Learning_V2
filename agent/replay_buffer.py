import numpy as np


class ReplayBuffer:

    def __init__(self, state_dim, capacity=100000, seed=None):

        self.capacity = capacity
        self.state_dim = state_dim

        self.position = 0
        self.size = 0

        self.rng = np.random.default_rng(seed)

        # NOTE: the state is now a vector of floats in [0, 1]
        # (e.g. 0.37 free-space fraction), so it must be float32.
        # The old uint8 buffer would have truncated every value to 0.
        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32
        )

        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32
        )

        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        if self.size < batch_size:
            raise ValueError("Not enough experiences in ReplayBuffer.")

        indices = self.rng.choice(
            self.size,
            size=batch_size,
            replace=False
        )

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):

        return self.size
