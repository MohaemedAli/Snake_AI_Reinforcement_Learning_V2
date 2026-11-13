import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from agent.replay_buffer import ReplayBuffer
from game.environment import SnakeEnvironment
from utils.checkpoint import load_checkpoint, save_checkpoint


class TestDQNAgent(unittest.TestCase):

    def setUp(self):

        self.env = SnakeEnvironment(seed=42)

        state = self.env.reset()

        self.agent = DQNAgent(
            state_dim=len(state),
            action_dim=3,
            batch_size=4,
            replay_capacity=100,
            target_update_interval=2,
            seed=42,
            device="cpu"
        )

    def test_network_output(self):

        state = self.env.reset()

        with torch.no_grad():
            output = self.agent.policy_net(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            )

        self.assertEqual(tuple(output.shape), (1, 3))

    def test_action_selection(self):
        state = self.env.reset()
        action = self.agent.select_action(state, epsilon=0.0)
        self.assertIn(action, [0, 1, 2])

    def test_random_action_selection(self):
        state = self.env.reset()
        action = self.agent.select_action(state, epsilon=1.0)
        self.assertIn(action, [0, 1, 2])

    def test_q_values_shape(self):
        state = self.env.reset()
        self.assertEqual(self.agent.q_values(state).shape, (3,))

    def test_replay_buffer_keeps_float_values(self):
        # Regression test: the old buffer used uint8 and would have
        # turned 0.37 into 0.
        buffer = ReplayBuffer(state_dim=3, capacity=10)
        state = np.array([0.37, 0.5, 1.0], dtype=np.float32)

        buffer.add(state, 0, 1.0, state, False)

        np.testing.assert_allclose(buffer.states[0], state)

    def test_replay_buffer(self):

        state = self.env.reset()

        for _ in range(4):
            self.agent.store_transition(
                state, action=0, reward=1.0, next_state=state, done=False
            )

        self.assertEqual(len(self.agent.replay_buffer), 4)

    def test_training_step(self):

        state = self.env.reset()

        for _ in range(4):
            self.agent.store_transition(
                state, action=0, reward=1.0, next_state=state, done=False
            )

        loss = self.agent.train_step()

        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(self.agent.optimization_steps, 1)

    def test_checkpoint_roundtrip(self):

        state = self.env.reset()
        before = self.agent.q_values(state)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"

            save_checkpoint(self.agent, path, 5, 100, 0.5, 3, 1.5)

            other = DQNAgent(
                state_dim=len(state), action_dim=3, device="cpu", seed=999
            )

            checkpoint = load_checkpoint(other, path)

        np.testing.assert_allclose(other.q_values(state), before, rtol=1e-5)
        self.assertEqual(checkpoint["episode"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
