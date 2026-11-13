import random
from collections import deque
from enum import IntEnum

import numpy as np

from game.config import GAME_CONFIG


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# Relative actions
# 0 = Straight
# 1 = Turn Right
# 2 = Turn Left

DIRECTION_VECTORS = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
}

# Number of values returned by SnakeEnvironment.get_state()
STATE_SIZE = 22


class SnakeEnvironment:
    """
    Snake game used both by the human player and by the RL agent.

    The observation is a compact feature vector (see get_state) instead of
    the raw 20x20x3 grid. A raw grid flattened into an MLP is extremely
    sparse and needs hundreds of thousands of steps to learn anything;
    the compact vector lets the agent learn in a few minutes.
    """

    def __init__(self, config=GAME_CONFIG, seed=None):

        self.config = config
        self.rng = random.Random(seed)

        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT

        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def reset(self, seed=None):

        if seed is not None:
            self.rng.seed(seed)

        center_x = self.width // 2
        center_y = self.height // 2

        self.snake = deque([
            (center_x - i, center_y)
            for i in range(self.config.INITIAL_SNAKE_LENGTH)
        ])

        self.direction = Direction.RIGHT

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        self.game_over = False
        self.won = False
        self.truncated = False  # True when the snake starved

        self.food = None
        self._spawn_food()

        self.previous_distance = self._distance_to_food()

        return self.get_state()

    def _spawn_food(self):

        occupied = set(self.snake)

        available = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]

        if not available:
            self.food = None
            self.won = True
            self.game_over = True
            return

        self.food = self.rng.choice(available)

    def _distance_to_food(self):

        if self.food is None:
            return 0

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        return abs(head_x - food_x) + abs(head_y - food_y)

    @property
    def max_steps_without_food(self):
        return (
            self.config.MAX_STEPS_WITHOUT_FOOD
            + self.config.STARVATION_PER_CELL * len(self.snake)
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _get_new_direction(self, action, direction=None):

        action = int(action)

        if action not in (0, 1, 2):
            raise ValueError(
                "Action must be 0 (straight), "
                "1 (right), or 2 (left)."
            )

        if direction is None:
            direction = self.direction

        rotation = (0, 1, -1)[action]

        return Direction((int(direction) + rotation) % 4)

    def _is_collision(self, position, will_eat):

        x, y = position

        # Wall collision
        if x < 0 or x >= self.width:
            return True

        if y < 0 or y >= self.height:
            return True

        # The tail moves away when not eating.
        body = list(self.snake)

        if not will_eat:
            body = body[:-1]

        return position in body

    def step(self, action):

        if self.game_over:
            raise RuntimeError(
                "Game is over. Call reset() before another step."
            )

        self.direction = self._get_new_direction(action)

        dx, dy = DIRECTION_VECTORS[self.direction]

        head_x, head_y = self.snake[0]

        new_head = (head_x + dx, head_y + dy)

        will_eat = new_head == self.food

        reward = self.config.REWARD_STEP

        # Collision
        if self._is_collision(new_head, will_eat):

            self.game_over = True

            reward = self.config.REWARD_COLLISION

            return (
                self.get_state(),
                reward,
                True,
                self._info()
            )

        # Move snake
        self.snake.appendleft(new_head)

        if will_eat:

            self.score += 1
            self.steps_since_food = 0

            reward += self.config.REWARD_FOOD

            self._spawn_food()

            if self.won:
                reward += self.config.REWARD_WIN
                self.steps += 1

                return (
                    self.get_state(),
                    reward,
                    True,
                    self._info()
                )

        else:
            self.snake.pop()
            self.steps_since_food += 1

            # Reward shaping. Only when the food can really be reached:
            # otherwise "getting closer" would push the snake to hug a
            # wall of its own body instead of finding a way around.
            current_distance = self._distance_to_food()

            if self._food_reachable_from_head():

                if current_distance < self.previous_distance:
                    reward += self.config.REWARD_CLOSER_TO_FOOD

                elif current_distance > self.previous_distance:
                    reward += self.config.REWARD_FARTHER_FROM_FOOD

        self.previous_distance = self._distance_to_food()

        self.steps += 1

        # Prevent endless wandering (starvation)
        if self.steps_since_food >= self.max_steps_without_food:
            self.game_over = True
            self.truncated = True
            reward = self.config.REWARD_COLLISION

        return (
            self.get_state(),
            reward,
            self.game_over,
            self._info()
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _blocked_cells_after_move(self, will_eat=False):
        """Cells that kill the snake on the next move."""
        body = list(self.snake)

        if not will_eat:
            body = body[:-1]

        return set(body)

    def _reachable_cells(self, start, blocked):
        """Flood fill (BFS): the set of cells reachable from `start`."""

        x, y = start

        if (
            x < 0 or x >= self.width
            or y < 0 or y >= self.height
            or start in blocked
        ):
            return set()

        seen = {start}
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()

            for dx, dy in DIRECTION_VECTORS.values():
                nxt = (cx + dx, cy + dy)

                if (
                    0 <= nxt[0] < self.width
                    and 0 <= nxt[1] < self.height
                    and nxt not in blocked
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append(nxt)

        return seen

    def _food_reachable_from_head(self):
        """True if the head can currently walk to the food."""

        if self.food is None:
            return False

        # The tail will move away, so it is not an obstacle.
        blocked = set(list(self.snake)[1:-1])

        return self.food in self._reachable_cells(self.snake[0], blocked)

    def get_state(self):
        """
        Feature vector (STATE_SIZE = 22 values, all in [0, 1]):

        0-2   danger straight / right / left (immediate collision)
        3-6   current direction one-hot (right, down, left, up)
        7-10  food is left / right / up / down of the head
        11-13 free space reachable after going straight / right / left
              (fraction of the board) - lets the agent avoid dead ends
        14-16 the TAIL is reachable after straight / right / left
              (1 = the snake can always escape by following its tail)
        17-19 the FOOD is reachable after straight / right / left
              (0 = the food is walled in by the body)
        20    normalised Manhattan distance to the food
        21    snake length / board size
        """

        head = self.snake[0]
        head_x, head_y = head
        tail = self.snake[-1]

        blocked = self._blocked_cells_after_move()
        total_cells = self.width * self.height

        danger = []
        space = []
        tail_reach = []
        food_reach = []

        for action in (0, 1, 2):
            direction = self._get_new_direction(action)
            dx, dy = DIRECTION_VECTORS[direction]
            nxt = (head_x + dx, head_y + dy)

            is_danger = (
                nxt[0] < 0 or nxt[0] >= self.width
                or nxt[1] < 0 or nxt[1] >= self.height
                or nxt in blocked
            )

            danger.append(1.0 if is_danger else 0.0)

            # The head itself becomes a body cell after moving.
            reach = self._reachable_cells(nxt, blocked | {head})

            space.append(len(reach) / total_cells)
            tail_reach.append(1.0 if tail in reach else 0.0)
            food_reach.append(
                1.0 if (self.food is not None and self.food in reach)
                else 0.0
            )

        direction_one_hot = [0.0] * 4
        direction_one_hot[int(self.direction)] = 1.0

        if self.food is not None:
            food_x, food_y = self.food
            food_dir = [
                float(food_x < head_x),
                float(food_x > head_x),
                float(food_y < head_y),
                float(food_y > head_y),
            ]
            distance = (
                abs(food_x - head_x) + abs(food_y - head_y)
            ) / (self.width + self.height)
        else:
            food_dir = [0.0] * 4
            distance = 0.0

        length = len(self.snake) / total_cells

        return np.array(
            danger
            + direction_one_hot
            + food_dir
            + space
            + tail_reach
            + food_reach
            + [distance, length],
            dtype=np.float32
        )

    def get_grid_state(self):
        """Raw (3, H, W) grid: body, head, food. Useful for a CNN."""

        state = np.zeros(
            (3, self.height, self.width),
            dtype=np.float32
        )

        for x, y in list(self.snake)[1:]:
            state[0, y, x] = 1.0

        head_x, head_y = self.snake[0]
        state[1, head_y, head_x] = 1.0

        if self.food is not None:
            food_x, food_y = self.food
            state[2, food_y, food_x] = 1.0

        return state

    def _info(self):

        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps,
            "steps_since_food": self.steps_since_food,
            "won": self.won,
            "truncated": self.truncated,
            "game_over": self.game_over,
        }
