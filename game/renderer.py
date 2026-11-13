import pygame

from game.environment import DIRECTION_VECTORS


def _lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _draw_overlay(screen, config, text, font, color, sub_text=None,
                  small_font=None):
    """Dim the board and show a centered message."""

    board_rect = pygame.Rect(
        0, config.HEADER_HEIGHT,
        config.WINDOW_WIDTH, config.GRID_HEIGHT * config.CELL_SIZE
    )

    shade = pygame.Surface(board_rect.size, pygame.SRCALPHA)
    shade.fill((0, 0, 0, 150))
    screen.blit(shade, board_rect.topleft)

    center_x = board_rect.centerx
    center_y = board_rect.centery

    label = font.render(text, True, color)
    screen.blit(label, label.get_rect(center=(center_x, center_y)))

    if sub_text and small_font:
        sub = small_font.render(sub_text, True, config.TEXT_COLOR)
        screen.blit(sub, sub.get_rect(center=(center_x, center_y + 35)))


def _draw_eyes(screen, env, cell_rect):
    """Two small eyes on the snake's head, looking in its direction."""

    dx, dy = DIRECTION_VECTORS[env.direction]

    cx, cy = cell_rect.center
    size = max(2, cell_rect.width // 8)

    # Perpendicular offset for the two eyes.
    px, py = -dy, dx
    forward = cell_rect.width // 5
    side = cell_rect.width // 4

    for sign in (-1, 1):
        ex = cx + dx * forward + px * side * sign
        ey = cy + dy * forward + py * side * sign
        pygame.draw.circle(screen, (15, 20, 25), (ex, ey), size)


def draw_game(
    screen,
    env,
    font,
    small_font,
    started=False,
    high_score=0,
    paused=False,
    mode="HUMAN",
    fps=None,
    q_values=None,
    games_played=None,
):

    config = env.config
    cell = config.CELL_SIZE
    offset_y = config.HEADER_HEIGHT

    screen.fill(config.BACKGROUND_COLOR)

    # -------------------------
    # Header
    # -------------------------

    screen.blit(
        font.render(f"Score: {env.score}", True, config.TEXT_COLOR),
        (12, 6)
    )

    best = small_font.render(
        f"Best: {max(high_score, env.score)}", True, config.ACCENT_COLOR
    )
    screen.blit(best, (150, 10))

    length = small_font.render(
        f"Length: {len(env.snake)}", True, config.TEXT_COLOR
    )
    screen.blit(length, (235, 10))

    mode_text = mode if fps is None else f"{mode}  |  {fps} FPS"
    mode_surface = small_font.render(mode_text, True, config.MUTED_COLOR)
    screen.blit(
        mode_surface,
        mode_surface.get_rect(topright=(config.WINDOW_WIDTH - 10, 10))
    )

    if q_values is not None:
        names = ("Straight", "Right", "Left")
        best_index = int(max(range(3), key=lambda i: q_values[i]))
        x = 12

        for i, name in enumerate(names):
            color = (
                config.SNAKE_HEAD_COLOR if i == best_index
                else config.MUTED_COLOR
            )
            surf = small_font.render(
                f"{name}: {q_values[i]:+.1f}", True, color
            )
            screen.blit(surf, (x, 32))
            x += surf.get_width() + 14

    elif games_played is not None:
        info = small_font.render(
            f"Games: {games_played}", True, config.MUTED_COLOR
        )
        screen.blit(info, (12, 32))

    # -------------------------
    # Grid (checkerboard)
    # -------------------------

    for gx in range(env.width):
        for gy in range(env.height):
            color = (
                config.GRID_COLOR if (gx + gy) % 2 == 0
                else config.GRID_ALT_COLOR
            )
            pygame.draw.rect(
                screen, color,
                (gx * cell, offset_y + gy * cell, cell, cell)
            )

    # -------------------------
    # Food
    # -------------------------

    if env.food is not None:
        food_x, food_y = env.food

        center = (
            food_x * cell + cell // 2,
            offset_y + food_y * cell + cell // 2
        )

        pygame.draw.circle(
            screen, _lerp_color(config.FOOD_COLOR, (0, 0, 0), 0.35),
            center, cell // 2 - 2
        )
        pygame.draw.circle(
            screen, config.FOOD_COLOR, center, cell // 2 - 4
        )
        pygame.draw.circle(
            screen, (255, 170, 170),
            (center[0] - 3, center[1] - 3), max(2, cell // 9)
        )

    # -------------------------
    # Snake (color fades from head to tail)
    # -------------------------

    snake_len = len(env.snake)

    for index, (x, y) in enumerate(env.snake):

        rect = pygame.Rect(
            x * cell + 2, offset_y + y * cell + 2, cell - 4, cell - 4
        )

        if index == 0:
            color = config.SNAKE_HEAD_COLOR
        else:
            t = (index - 1) / max(1, snake_len - 2)
            color = _lerp_color(
                config.SNAKE_COLOR, config.SNAKE_TAIL_COLOR, t
            )

        pygame.draw.rect(screen, color, rect, border_radius=6)

        if index == 0:
            _draw_eyes(screen, env, rect)

    # -------------------------
    # Overlays
    # -------------------------

    if env.game_over:
        message = "YOU WIN!" if env.won else "GAME OVER"
        color = config.SNAKE_HEAD_COLOR if env.won else config.FOOD_COLOR
        hint = (
            "Press R to restart" if mode == "HUMAN"
            else "Restarting... (R = restart now)"
        )
        _draw_overlay(screen, config, message, font, color, hint, small_font)

    elif paused:
        _draw_overlay(
            screen, config, "PAUSED", font, config.ACCENT_COLOR,
            "Press P to resume", small_font
        )

    elif not started:
        _draw_overlay(
            screen, config, "READY", font, config.TEXT_COLOR,
            "Press an arrow key or WASD to start", small_font
        )
