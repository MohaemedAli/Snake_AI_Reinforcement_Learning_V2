"""
Watch the trained agent play.

    python play_ai.py                       # uses checkpoints/best_model.pt
    python play_ai.py --checkpoint checkpoints/latest.pt

Controls
    + / -    faster / slower
    P        pause / resume
    R        restart now
    ESC      quit
"""

import argparse
from pathlib import Path

import pygame

from agent.dqn_agent import DQNAgent
from game.config import GAME_CONFIG
from game.environment import SnakeEnvironment
from game.renderer import draw_game
from utils.checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parent
AUTO_RESTART_DELAY_MS = 1200


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "checkpoints" / "best_model.pt")
    )
    parser.add_argument("--fps", type=int, default=GAME_CONFIG.FPS * 2)
    args = parser.parse_args()

    config = GAME_CONFIG
    env = SnakeEnvironment()
    state = env.reset()

    agent = DQNAgent(state_dim=len(state), action_dim=3, device="cpu")

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Train first with:  python -m training.train"
        )

    checkpoint = load_checkpoint(agent, args.checkpoint, load_optimizer=False)
    print(
        f"Loaded {args.checkpoint} "
        f"(episode {checkpoint['episode']}, "
        f"best score {checkpoint['best_score']})"
    )

    pygame.init()

    screen = pygame.display.set_mode(
        (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    )
    pygame.display.set_caption(config.WINDOW_TITLE + " (AI)")

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 34)
    small_font = pygame.font.Font(None, 23)

    fps = max(config.MIN_FPS, min(config.MAX_FPS, args.fps))
    high_score = 0
    games_played = 0
    paused = False
    game_over_since = None

    running = True

    while running:

        clock.tick(fps)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    state = env.reset()
                    game_over_since = None

                elif event.key == pygame.K_p:
                    paused = not paused

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                   pygame.K_KP_PLUS):
                    fps = min(config.MAX_FPS, fps + 2)

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(config.MIN_FPS, fps - 2)

        if not running:
            break

        q_values = agent.q_values(state) if not env.game_over else None

        if not paused and not env.game_over:
            action = agent.select_action(state, epsilon=0.0)
            state, _, _, _ = env.step(action)

            if env.game_over:
                games_played += 1
                high_score = max(high_score, env.score)
                game_over_since = pygame.time.get_ticks()

        # Automatic restart after a short pause.
        if (
            env.game_over
            and game_over_since is not None
            and pygame.time.get_ticks() - game_over_since
            > AUTO_RESTART_DELAY_MS
        ):
            state = env.reset()
            game_over_since = None

        draw_game(
            screen, env, font, small_font,
            started=True,
            high_score=high_score,
            paused=paused,
            mode="AI",
            fps=fps,
            q_values=q_values,
            games_played=games_played,
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
