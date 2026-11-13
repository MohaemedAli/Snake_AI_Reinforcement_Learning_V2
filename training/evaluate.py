import numpy as np


def evaluate(agent, env, episodes=20, epsilon=0.0, seed_base=10_000):
    """
    Run `episodes` games with the (almost) greedy policy and return
    statistics. Seeds are fixed so every evaluation uses the same
    food sequences and results are comparable between checkpoints.
    """

    scores = []
    steps = []

    for i in range(episodes):

        state = env.reset(seed=seed_base + i)
        done = False
        info = {"score": 0, "steps": 0}

        while not done:
            action = agent.select_action(state, epsilon=epsilon)
            state, _, done, info = env.step(action)

        scores.append(info["score"])
        steps.append(info["steps"])

    return {
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "mean_steps": float(np.mean(steps)),
    }


if __name__ == "__main__":

    import argparse
    from pathlib import Path

    from agent.dqn_agent import DQNAgent
    from game.environment import SnakeEnvironment
    from utils.checkpoint import load_checkpoint

    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=str(root / "checkpoints" / "best_model.pt")
    )
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    env = SnakeEnvironment()
    agent = DQNAgent(state_dim=len(env.reset()), action_dim=3)
    load_checkpoint(agent, args.checkpoint, load_optimizer=False)

    print(evaluate(agent, env, episodes=args.episodes))
