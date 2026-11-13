"""
Play Snake yourself.

Controls
    Arrows / WASD   steer (up to 2 turns can be queued per tick)
    P               pause / resume
    + / -           faster / slower
    R               restart
    ESC             quit
"""

import json
from collections import deque
from pathlib import Path

import pygame

from game.config import GAME_CONFIG
from game.environment import SnakeEnvironment, Direction
from game.renderer import draw_game


HIGHSCORE_FILE = Path(__file__).resolve().parent / "highscore.json"

# Keyboard controls: arrows and WASD
KEY_DIRECTIONS = {
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_d: Direction.RIGHT,

    pygame.K_DOWN: Direction.DOWN,
    pygame.K_s: Direction.DOWN,

    pygame.K_LEFT: Direction.LEFT,
    pygame.K_a: Direction.LEFT,

    pygame.K_UP: Direction.UP,
    pygame.K_w: Direction.UP,
}


def key_to_action(desired_direction, current_direction):
    """
    Convert an absolute direction into a relative action.

    0 = Straight, 1 = Turn Right, 2 = Turn Left.
    Returning None means an illegal 180-degree turn.
    """

    difference = (int(desired_direction) - int(current_direction)) % 4

    if difference == 0:
        return 0

    if difference == 1:
        return 1

    if difference == 3:
        return 2

    return None


def load_high_score():
    try:
        return int(json.loads(HIGHSCORE_FILE.read_text())["high_score"])
    except (OSError, ValueError, KeyError):
        return 0


def save_high_score(value):
    try:
        HIGHSCORE_FILE.write_text(json.dumps({"high_score": value}))
    except OSError:
        pass


def main():

    pygame.init()

    config = GAME_CONFIG

    screen = pygame.display.set_mode(
        (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    )
    pygame.display.set_caption(config.WINDOW_TITLE)

    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 34)
    small_font = pygame.font.Font(None, 23)

    env = SnakeEnvironment()

    high_score = load_high_score()
    games_played = 0
    fps = config.FPS

    running = True
    started = False
    paused = False

    # Queue of requested directions. Without a queue, pressing two keys
    # quickly inside one tick (e.g. UP then LEFT) loses the first turn.
    pending = deque(maxlen=2)

    while running:

        clock.tick(fps)

        # -------------------------
        # 1. Handle keyboard events
        # -------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    env.reset()
                    started = False
                    paused = False
                    pending.clear()

                elif event.key == pygame.K_p and started and not env.game_over:
                    paused = not paused

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                   pygame.K_KP_PLUS):
                    fps = min(config.MAX_FPS, fps + 2)

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(config.MIN_FPS, fps - 2)

                elif not env.game_over and not paused:

                    requested = KEY_DIRECTIONS.get(event.key)

                    if requested is not None:

                        # Compare with the last queued direction so that
                        # queued turns are validated in order.
                        reference = pending[-1] if pending else env.direction

                        if requested == reference:
                            # Pressing the current direction just starts
                            # the game if it has not started yet.
                            started = True

                        elif key_to_action(requested, reference) is not None:
                            pending.append(requested)
                            started = True

        if not running:
            break

        # -------------------------
        # 2. Move the snake
        # -------------------------

        if started and not paused and not env.game_over:

            action = 0

            if pending:
                action = key_to_action(pending.popleft(), env.direction)

                if action is None:
                    action = 0

            env.step(action)

            if env.game_over:
                games_played += 1

                if env.score > high_score:
                    high_score = env.score
                    save_high_score(high_score)

        # -------------------------
        # 3. Draw the game
        # -------------------------

        draw_game(
            screen, env, font, small_font,
            started=started,
            high_score=high_score,
            paused=paused,
            mode="HUMAN",
            fps=fps,
            games_played=games_played,
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
