import csv
from pathlib import Path

from utils.metrics import moving_average


def plot_training(metrics_file, output_file, window=100):
    """Save a 3-panel figure (score, steps, loss) from the metrics CSV."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, scores, steps, losses, evals = [], [], [], [], []

    with open(metrics_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
            steps.append(float(row["steps"]))
            losses.append(float(row["average_loss"]))

    if not episodes:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, scores, alpha=0.25, label="score")
    axes[0].plot(
        episodes, moving_average(scores, window),
        label=f"moving avg ({window})"
    )
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, steps, alpha=0.25)
    axes[1].plot(episodes, moving_average(steps, window))
    axes[1].set_ylabel("Steps / episode")
    axes[1].grid(alpha=0.3)

    axes[2].plot(episodes, losses)
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Snake DQN training")
    fig.tight_layout()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=120)
    plt.close(fig)

    return output_file
