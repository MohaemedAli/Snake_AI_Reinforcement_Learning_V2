from pathlib import Path

import torch


def build_checkpoint(agent, episode, global_step, epsilon, best_score,
                     best_eval=None):

    return {
        "episode": episode,
        "global_step": global_step,
        "epsilon": epsilon,
        "best_score": best_score,
        "best_eval": best_eval,

        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),

        "state_dim": agent.state_dim,
        "action_dim": agent.action_dim,

        "gamma": agent.gamma,
        "batch_size": agent.batch_size,

        "optimization_steps": agent.optimization_steps,
    }


def save_checkpoint(agent, path, episode, global_step, epsilon,
                    best_score, best_eval=None):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        build_checkpoint(
            agent, episode, global_step, epsilon, best_score, best_eval
        ),
        path
    )


def load_checkpoint(agent, path, load_optimizer=True):
    """Load weights into `agent`. Returns the checkpoint dict."""

    checkpoint = torch.load(
        path, map_location=agent.device, weights_only=False
    )

    if checkpoint["state_dim"] != agent.state_dim:
        raise ValueError(
            f"Checkpoint state_dim={checkpoint['state_dim']} does not "
            f"match the agent (state_dim={agent.state_dim}). Old "
            f"checkpoints trained with the raw-grid state are not "
            f"compatible - delete them and retrain."
        )

    agent.policy_net.load_state_dict(checkpoint["policy_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])

    if load_optimizer and "optimizer" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer"])

    agent.optimization_steps = checkpoint.get("optimization_steps", 0)

    return checkpoint
