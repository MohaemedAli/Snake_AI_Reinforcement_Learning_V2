import unittest
from collections import deque

from game.config import GameConfig
from game.environment import SnakeEnvironment, Direction, STATE_SIZE


class TestSnakeEnvironment(unittest.TestCase):

    def setUp(self):
        config = GameConfig(
            GRID_WIDTH=8,
            GRID_HEIGHT=8,
            CELL_SIZE=20,
            INITIAL_SNAKE_LENGTH=3,
            MAX_STEPS_WITHOUT_FOOD=50
        )

        self.env = SnakeEnvironment(config=config, seed=42)

    def test_initial_state_shape(self):
        self.assertEqual(self.env.get_state().shape, (STATE_SIZE,))

    def test_state_values_in_range(self):
        state = self.env.get_state()
        self.assertTrue((state >= 0.0).all())
        self.assertTrue((state <= 1.0).all())

    def test_grid_state_shape(self):
        self.assertEqual(self.env.get_grid_state().shape, (3, 8, 8))

    def test_initial_snake_length(self):
        self.assertEqual(len(self.env.snake), 3)

    def test_forward_movement(self):
        old_head = self.env.snake[0]
        self.env.step(0)

        self.assertEqual(
            self.env.snake[0],
            (old_head[0] + 1, old_head[1])
        )

    def test_turn_right(self):
        self.env.step(1)
        self.assertEqual(self.env.direction, Direction.DOWN)

    def test_turn_left(self):
        self.env.step(2)
        self.assertEqual(self.env.direction, Direction.UP)

    def test_eating_food(self):
        self.env.snake = deque([(3, 3), (2, 3), (1, 3)])
        self.env.direction = Direction.RIGHT
        self.env.food = (4, 3)

        old_length = len(self.env.snake)

        _, reward, done, info = self.env.step(0)

        self.assertEqual(self.env.score, 1)
        self.assertEqual(len(self.env.snake), old_length + 1)
        self.assertGreater(reward, 0)
        self.assertFalse(done)

    def test_wall_collision(self):
        self.env.snake = deque([(7, 3), (6, 3), (5, 3)])
        self.env.direction = Direction.RIGHT
        self.env.food = (0, 0)

        _, reward, done, info = self.env.step(0)

        self.assertTrue(done)
        self.assertEqual(reward, -10.0)

    def test_body_collision(self):
        self.env.snake = deque([(3, 3), (3, 2), (4, 2), (4, 3)])
        self.env.direction = Direction.RIGHT
        self.env.food = (7, 7)

        _, reward, done, info = self.env.step(2)

        self.assertTrue(done)

    def test_tail_position_can_be_reused(self):
        self.env.snake = deque([(2, 2), (2, 3), (3, 3), (3, 2)])
        self.env.direction = Direction.RIGHT
        self.env.food = (7, 7)

        _, reward, done, info = self.env.step(0)

        self.assertFalse(done)
        self.assertEqual(self.env.snake[0], (3, 2))

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            self.env.step(5)

    def test_full_board_victory(self):
        self.env.snake = deque([
            (x, y) for y in range(8) for x in range(8)
        ])

        self.env._spawn_food()

        self.assertTrue(self.env.won)
        self.assertTrue(self.env.game_over)
        self.assertIsNone(self.env.food)

    # ---- new tests ----

    def test_danger_features_next_to_wall(self):
        # Head at the right wall, moving right: straight is dangerous.
        self.env.snake = deque([(7, 3), (6, 3), (5, 3)])
        self.env.direction = Direction.RIGHT
        self.env.food = (0, 0)

        state = self.env.get_state()

        self.assertEqual(state[0], 1.0)  # straight -> wall
        self.assertEqual(state[1], 0.0)  # right (down) is free
        self.assertEqual(state[2], 0.0)  # left (up) is free

    def test_food_direction_features(self):
        self.env.snake = deque([(3, 3), (2, 3), (1, 3)])
        self.env.direction = Direction.RIGHT
        self.env.food = (6, 1)

        state = self.env.get_state()

        # indices 7..10 = food left, right, up, down
        self.assertEqual(list(state[7:11]), [0.0, 1.0, 1.0, 0.0])

    def test_dead_end_has_less_free_space(self):
        # Snake wraps around so that turning right enters a tiny pocket.
        self.env.snake = deque([
            (1, 1), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2),
            (2, 1), (2, 0)
        ])
        self.env.direction = Direction.RIGHT
        self.env.food = (7, 7)

        state = self.env.get_state()

        # Going straight is blocked (2,1) -> danger, space 0.
        self.assertEqual(state[0], 1.0)
        self.assertEqual(state[11], 0.0)

    def test_starvation_ends_game(self):
        self.env.food = (0, 7)
        self.env.snake = deque([(3, 3), (2, 3), (1, 3)])
        self.env.direction = Direction.RIGHT

        # Turning right every step makes the snake circle in a 2x2 loop
        # without ever reaching the food.
        for _ in range(self.env.max_steps_without_food + 5):
            if self.env.game_over:
                break
            self.env.step(1)

        self.assertTrue(self.env.game_over)
        self.assertTrue(self.env.truncated)

    # ---- level 2 tests: tail / food reachability ----

    def test_tail_reachable_in_open_board(self):
        state = self.env.get_state()
        # indices 14..16 = tail reachable after straight / right / left
        self.assertEqual(list(state[14:17]), [1.0, 1.0, 1.0])

    def test_food_reachable_in_open_board(self):
        state = self.env.get_state()
        self.assertEqual(list(state[17:20]), [1.0, 1.0, 1.0])

    def test_walled_in_food_is_not_reachable(self):
        # Food in the corner (0,0) is closed by body cells (1,0) and (0,1).
        self.env.snake = deque([
            (2, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3)
        ])
        self.env.direction = Direction.RIGHT
        self.env.food = (0, 0)
        self.assertFalse(self.env._food_reachable_from_head())
        state = self.env.get_state()
        self.assertEqual(list(state[17:20]), [0.0, 0.0, 0.0])

    def test_no_shaping_reward_when_food_unreachable(self):
        # Food in the corner (0,0) is closed by body cells (1,0) and (0,1).
        self.env.snake = deque([
            (2, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3)
        ])
        self.env.direction = Direction.RIGHT
        self.env.food = (0, 0)
        self.env.previous_distance = self.env._distance_to_food()

        # Turning right goes down to (2,1): farther from the food, but no
        # penalty because the food is walled in.
        _, reward, done, _ = self.env.step(1)

        self.assertFalse(done)
        self.assertAlmostEqual(reward, self.env.config.REWARD_STEP)

    def test_tail_unreachable_inside_closed_pocket(self):
        # Head about to enter a 1-cell pocket enclosed by its own body:
        # the tail must not be reachable through that move.
        self.env.snake = deque([
            (2, 2), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (3, 1),
            (2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4),
            (4, 4), (5, 4), (6, 4), (6, 3)
        ])
        self.env.direction = Direction.RIGHT
        self.env.food = (7, 7)
        state = self.env.get_state()
        # Straight goes to (3,2): the pocket inside the coil, with no
        # connection to the tail at (6,3).
        self.assertEqual(state[14], 0.0)


    def test_reset_with_same_seed_is_reproducible(self):
        self.env.reset(seed=7)
        food_a = self.env.food
        self.env.reset(seed=7)
        self.assertEqual(food_a, self.env.food)


if __name__ == "__main__":
    unittest.main(verbosity=2)
