"""
Export a trained checkpoint for the browser demo.

Run from the project root (with the .venv active):

    python export_weights.py checkpoints/best_model.pt ../snake-ai-demo

Writes into the output folder:
  weights.js   the network weights (loaded by index.html)
  golden.json  300 game positions with the Python features and Q-values;
               tests/parity.js checks that the JavaScript port reproduces them.
"""
import json
import sys
from pathlib import Path

import numpy as np

from agent.dqn_agent import DQNAgent
from game.environment import SnakeEnvironment
from utils.checkpoint import load_checkpoint

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "demo_export")
out.mkdir(parents=True, exist_ok=True)

env = SnakeEnvironment(seed=7)
agent = DQNAgent(state_dim=len(env.reset()), action_dim=3, device="cpu")
load_checkpoint(agent, checkpoint, load_optimizer=False)

state_dict = {k: v.cpu().numpy() for k, v in agent.policy_net.state_dict().items()}
layers = {
    "f1": "features.0", "f2": "features.2",
    "v1": "value_stream.0", "v2": "value_stream.2",
    "a1": "advantage_stream.0", "a2": "advantage_stream.2",
}
net = {
    name: {
        "W": state_dict[key + ".weight"].round(6).tolist(),
        "b": state_dict[key + ".bias"].round(6).tolist(),
    }
    for name, key in layers.items()
}
(out / "weights.js").write_text(
    "window.SNAKE_WEIGHTS=" + json.dumps(net, separators=(",", ":")) + ";"
)

# Golden vectors: play greedily and record what Python computes at every step.
golden = []
state = env.reset(seed=11)
for _ in range(300):
    q = agent.q_values(state)
    golden.append({
        "snake": [list(c) for c in env.snake],
        "dir": int(env.direction),
        "food": list(env.food) if env.food else None,
        "state": [float(x) for x in state],
        "q": [float(x) for x in q],
    })
    state, _, done, _ = env.step(int(np.argmax(q)))
    if done:
        state = env.reset(seed=len(golden))
(out / "golden.json").write_text(json.dumps(golden))

print(f"Wrote {out / 'weights.js'} and {out / 'golden.json'}")