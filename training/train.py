import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from agent.exploration import EpsilonScheduler
from game.environment import SnakeEnvironment
from training.evaluate import evaluate
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.plotting import plot_training


# =====================================
# Training configuration
# =====================================

NUM_EPISODES = 3000

LEARNING_STARTS = 1000

BATCH_SIZE = 64

REPLAY_CAPACITY = 100000

LEARNING_RATE = 0.0003

GAMMA = 0.98

TARGET_UPDATE_INTERVAL = 500

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_STEPS = 40000

SEED = 42

SAVE_EVERY = 100

# A greedy evaluation (no exploration) is run every EVAL_EVERY episodes.
# best_model.pt is the checkpoint with the best *evaluation* score, not
# the best single lucky training episode.
EVAL_EVERY = 100
EVAL_EPISODES = 50


# =====================================
# Project directories
# =====================================

ROOT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
PLOT_DIR = ROOT_DIR / "plots"

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

METRICS_FILE = LOG_DIR / "training_metrics.csv"
EVAL_FILE = LOG_DIR / "evaluation_metrics.csv"


def parse_args():

    parser = argparse.ArgumentParser(description="Train the Snake DQN agent")

    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from checkpoints/latest.pt"
    )

    return parser.parse_args()


def main():

    args = parse_args()
    num_episodes = args.episodes

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Environments (a separate one for evaluation)
    env = SnakeEnvironment(seed=SEED)
    eval_env = SnakeEnvironment(seed=SEED + 1)

    state_dim = len(env.reset())
    action_dim = 3

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        replay_capacity=REPLAY_CAPACITY,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        seed=SEED
    )

    epsilon_scheduler = EpsilonScheduler(
        start=EPSILON_START,
        end=EPSILON_END,
        decay_steps=EPSILON_DECAY_STEPS
    )

    global_step = 0
    start_episode = 1
    best_score = -1
    best_eval = -1.0
    file_mode = "w"

    if args.resume:
        latest = CHECKPOINT_DIR / "latest.pt"

        if latest.exists():
            ckpt = load_checkpoint(agent, latest)
            global_step = ckpt["global_step"]
            start_episode = ckpt["episode"] + 1
            best_score = ckpt["best_score"]
            best_eval = ckpt.get("best_eval") or -1.0
            file_mode = "a"
            print(f"Resumed from episode {ckpt['episode']}")
        else:
            print("No latest.pt found - starting from scratch.")

    print("=" * 60)
    print("SNAKE AI - DOUBLE DQN TRAINING")
    print("=" * 60)
    print(f"Device:           {agent.device}")
    print(f"State dimension:  {state_dim}")
    print(f"Episodes:         {start_episode} -> {num_episodes}")
    print(f"Learning rate:    {LEARNING_RATE}   Gamma: {GAMMA}")
    print(f"Epsilon:          {EPSILON_START} -> {EPSILON_END} "
          f"over {EPSILON_DECAY_STEPS} steps")
    print("=" * 60)

    recent_scores = deque(maxlen=100)

    fieldnames = [
        "episode", "score", "snake_length", "steps",
        "epsilon", "average_score_100", "average_loss"
    ]

    eval_fieldnames = [
        "episode", "mean_score", "max_score", "min_score", "mean_steps"
    ]

    epsilon = epsilon_scheduler.get_epsilon(global_step)

    with open(METRICS_FILE, mode=file_mode, newline="", encoding="utf-8") as csv_file, \
         open(EVAL_FILE, mode=file_mode, newline="", encoding="utf-8") as eval_file:

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        eval_writer = csv.DictWriter(eval_file, fieldnames=eval_fieldnames)

        if file_mode == "w":
            writer.writeheader()
            eval_writer.writeheader()

        for episode in range(start_episode, num_episodes + 1):

            state = env.reset()
            done = False
            episode_losses = []
            info = {"score": 0, "snake_length": 3, "steps": 0}

            while not done:

                epsilon = epsilon_scheduler.get_epsilon(global_step)

                action = agent.select_action(state, epsilon=epsilon)

                next_state, reward, done, info = env.step(action)

                agent.store_transition(
                    state, action, reward, next_state, done
                )

                if len(agent.replay_buffer) >= LEARNING_STARTS:
                    loss = agent.train_step()

                    if loss is not None:
                        episode_losses.append(loss)

                state = next_state
                global_step += 1

            # ---------------- Episode statistics ----------------

            score = info["score"]
            recent_scores.append(score)
            average_score = float(np.mean(recent_scores))
            average_loss = (
                float(np.mean(episode_losses)) if episode_losses else 0.0
            )

            best_score = max(best_score, score)

            writer.writerow({
                "episode": episode,
                "score": score,
                "snake_length": info["snake_length"],
                "steps": info["steps"],
                "epsilon": epsilon,
                "average_score_100": average_score,
                "average_loss": average_loss
            })
            csv_file.flush()

            # ---------------- Greedy evaluation ----------------

            if episode % EVAL_EVERY == 0:

                result = evaluate(agent, eval_env, episodes=EVAL_EPISODES)

                eval_writer.writerow({"episode": episode, **result})
                eval_file.flush()

                marker = ""

                if result["mean_score"] > best_eval:
                    best_eval = result["mean_score"]
                    save_checkpoint(
                        agent, CHECKPOINT_DIR / "best_model.pt",
                        episode, global_step, epsilon, best_score, best_eval
                    )
                    marker = "  <- new best model"

                print(
                    f"   [EVAL ep {episode}] greedy mean score "
                    f"{result['mean_score']:.2f} | "
                    f"max {result['max_score']}{marker}"
                )

            # ---------------- Periodic checkpoint ----------------

            if episode % SAVE_EVERY == 0:
                save_checkpoint(
                    agent, CHECKPOINT_DIR / "latest.pt",
                    episode, global_step, epsilon, best_score, best_eval
                )

            # ---------------- Console output ----------------

            if episode % 20 == 0 or episode == start_episode:
                print(
                    f"Episode {episode:5d}/{num_episodes} | "
                    f"Score {score:3d} | "
                    f"Avg(100) {average_score:6.2f} | "
                    f"Best {best_score:3d} | "
                    f"Eps {epsilon:.3f} | "
                    f"Loss {average_loss:.4f}"
                )

    # Final checkpoint + plot
    save_checkpoint(
        agent, CHECKPOINT_DIR / "latest.pt",
        num_episodes, global_step, epsilon, best_score, best_eval
    )

    plot_path = plot_training(METRICS_FILE, PLOT_DIR / "training.png")

    print("=" * 60)
    print("TRAINING FINISHED")
    print(f"Best training score:   {best_score}")
    print(f"Best evaluation score: {best_eval:.2f}")
    print(f"Metrics:     {METRICS_FILE}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    if plot_path:
        print(f"Plot:        {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
