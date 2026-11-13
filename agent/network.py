import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network for the compact feature vector.

    The old network had 512/256 units for a 1204-dim sparse input.
    The new input is only 16 dense features, so a smaller network
    trains faster and generalises better.
    """

    def __init__(self, state_dim, action_dim=3, hidden=128):

        super().__init__()

        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Value stream: how good is this state?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream: how good is each action?
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):

        features = self.features(state)

        value = self.value_stream(features)

        advantages = self.advantage_stream(features)

        # Dueling aggregation
        return (
            value
            + advantages
            - advantages.mean(dim=1, keepdim=True)
        )
