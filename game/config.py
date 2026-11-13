from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    # Board
    GRID_WIDTH: int = 20
    GRID_HEIGHT: int = 20
    CELL_SIZE: int = 25

    # Display
    FPS: int = 8
    MIN_FPS: int = 2
    MAX_FPS: int = 60
    WINDOW_TITLE: str = "Snake AI - Reinforcement Learning"

    # Initial snake
    INITIAL_SNAKE_LENGTH: int = 3
    HEADER_HEIGHT: int = 50

    # The snake "starves" if it does not eat for this many steps
    # (+ STARVATION_PER_CELL steps for every body segment).
    MAX_STEPS_WITHOUT_FOOD: int = 100
    STARVATION_PER_CELL: int = 10

    # Rewards
    REWARD_FOOD: float = 10.0
    REWARD_COLLISION: float = -10.0
    REWARD_WIN: float = 50.0
    REWARD_STEP: float = -0.01
    REWARD_CLOSER_TO_FOOD: float = 0.1
    REWARD_FARTHER_FROM_FOOD: float = -0.1

    # Colors: RGB
    BACKGROUND_COLOR: tuple = (18, 24, 32)
    GRID_COLOR: tuple = (30, 38, 48)
    GRID_ALT_COLOR: tuple = (22, 29, 38)
    SNAKE_COLOR: tuple = (40, 190, 110)
    SNAKE_TAIL_COLOR: tuple = (28, 120, 75)
    SNAKE_HEAD_COLOR: tuple = (80, 230, 145)
    FOOD_COLOR: tuple = (235, 75, 75)
    TEXT_COLOR: tuple = (240, 240, 240)
    MUTED_COLOR: tuple = (140, 150, 165)
    ACCENT_COLOR: tuple = (70, 150, 230)

    @property
    def WINDOW_WIDTH(self):
        return self.GRID_WIDTH * self.CELL_SIZE

    @property
    def WINDOW_HEIGHT(self):
        return self.GRID_HEIGHT * self.CELL_SIZE + self.HEADER_HEIGHT


GAME_CONFIG = GameConfig()
