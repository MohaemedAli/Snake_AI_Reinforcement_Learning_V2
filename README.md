# Snake AI — Deep Reinforcement Learning

[![Live demo](https://img.shields.io/badge/live-demo-brightgreen)](https://mohaemedali.github.io/Snake_AI_Reinforcement_Learning_V2/)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-orange)

A Snake game written from scratch in Python, and an agent that learns to play it with **Double + Dueling DQN** (PyTorch). The game is used as a small, fully controlled environment to study how an agent learns decisions from rewards: how the state is designed, how the reward is shaped, how learning is evaluated, and where it fails.

**[▶ Try it in your browser](https://mohaemedali.github.io/Snake_AI_Reinforcement_Learning_V2/)** — watch the trained agent play, or play yourself. It runs entirely client-side, no server.

<!-- TODO: add a GIF of the agent playing: docs/images/demo.gif, then uncomment
![The agent playing](docs/images/demo.gif)
-->

## Highlights

- Custom Snake environment with a clean `reset()` / `step(action)` interface.
- Compact **22-feature state** (danger, direction, food, reachable space, tail/food reachability) instead of the raw grid.
- **Double DQN + Dueling network + experience replay**, with checkpointing, resumable training and fixed-seed greedy evaluation.
- Browser demo: the 22 features and the network were re-implemented in JavaScript and checked against the Python code on 300 game positions.
- Unit tests for the environment and the agent (31 tests).

## Results

Greedy policy (no exploration), **200 games** on fixed seeds, 20×20 board:

| Metric | Value |
|---|---|
| Mean score | **85.8** |
| Best game | 139 |
| Worst game | 2 |
| Mean steps per game | ≈ 1,986 |

*Training:* one run (seed 42), about 3,300 episodes, stopped manually because the greedy evaluation had stopped improving. The checkpoint is the one with the best evaluation score during training; the 200-game figure above is the one to trust, since picking the best of many noisy evaluations tends to overestimate.

<!-- TODO: add the training curve: python -c "from utils.plotting import plot_training; plot_training('logs/training_metrics.csv', 'docs/images/training.png')"
![Training curves](docs/images/training.png)
-->

### Reference points (same environment)

Measured during development with throwaway scripts (to be added under `baselines/`):

| Policy | Result |
|---|---|
| Hand-written rules, using the 16-feature information | mean ≈ 70 (40 games) |
| Hand-written rules, using the 22-feature information | mean ≈ 320 (40 games), 6 games filled the whole board |
| Hamiltonian cycle (no learning) | filled the board: 397 / 397 (1 deterministic game, 36,517 steps) |
| **DQN agent (this repo)** | **mean 85.8** (200 games) |

A full board is 397 points (400 cells minus the 3 starting cells). These numbers suggest the information in the state is enough for much stronger play than the agent currently shows, so the bottleneck is learning, not information.

## How it works

### Environment (`game/environment.py`)
20×20 grid. The snake starts with length 3 in the centre. Actions are **relative** to the heading: `0` straight, `1` turn right, `2` turn left (a 180° turn is impossible by construction). The tail cell is treated as free when the snake does not eat, because it moves away in the same step.

### State — 22 features, all in [0, 1]

| Index | Meaning |
|---|---|
| 0–2 | Immediate danger: straight / right / left |
| 3–6 | Current direction (one-hot) |
| 7–10 | Food is left / right / up / down of the head |
| 11–13 | Free space reachable after each action (flood fill, fraction of the board) |
| 14–16 | Tail reachable after each action (a way out exists) |
| 17–19 | Food reachable after each action (0 = walled in by the body) |
| 20 | Manhattan distance to the food (normalised) |
| 21 | Snake length (normalised) |

The reachability features were added after watching the first agent trap itself and chase food it could not reach.

### Reward

| Event | Reward |
|---|---|
| Eat food | +10 |
| Collision (wall or body) | −10 |
| Starvation (no food for `100 + 10 × length` steps) | −10 |
| Each step | −0.01 |
| Moving closer to / farther from the food | +0.1 / −0.1, **only when the food is reachable** |
| Filling the board | +50 |

### Agent (`agent/`)
- **Dueling network**: shared layers `22 → 128 → 128`, then a value stream and an advantage stream, `Q = V + A − mean(A)`.
- **Double DQN** target: the online network picks the next action, the target network evaluates it:
  `y = r + γ · (1 − done) · Q_target(s', argmax_a Q_online(s', a))`.
- Replay buffer (100k transitions), batch 64, Huber loss, Adam (lr 3·10⁻⁴), γ = 0.98, gradient clipping at 10, target network copied every 500 updates.
- ε-greedy exploration, linear decay 1.0 → 0.02 over 40,000 steps.

### Evaluation
Every 100 episodes the agent plays 50 greedy games on fixed seeds, and `best_model.pt` is the checkpoint with the best evaluation mean (not the luckiest training episode).

## Quick start

```bash
git clone https://github.com/MohaemedAli/Snake_AI_Reinforcement_Learning_V2.git
cd Snake_AI_Reinforcement_Learning_V2

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python main.py                                   # play yourself (arrows/WASD, P pause, +/- speed)
python play_ai.py --checkpoint models/best_model.pt          # watch the trained agent
python -m training.evaluate --checkpoint models/best_model.pt --episodes 200
python -m training.train --episodes 3000         # train from scratch
python -m training.train --resume --episodes 5000   # continue from checkpoints/latest.pt
python -m unittest test_environment test_agent   # run the tests
```

Training writes `checkpoints/`, `logs/` (CSV metrics) and `plots/`.

## Project structure

```
agent/       Dueling network, replay buffer, epsilon schedule, Double DQN agent
game/        Environment (rules + features), renderer, configuration
training/    Training loop and evaluation
utils/       Checkpointing, metrics, plotting
docs/        Browser demo (served with GitHub Pages)
main.py      Play the game yourself
play_ai.py   Watch the trained agent (shows its Q-values)
export_weights.py   Export a checkpoint to weights.js for the browser demo
test_*.py    Unit tests
```

## Known limitations

- The agent usually dies by **trapping itself** once its body is long, and it never completes the board.
- The greedy evaluation plateaued at a mean of about 80–90 after roughly 1,400 episodes.
- Some games end very early (worst game: 2 points); the cause has not been investigated yet.
- A single training run with one seed: no confidence intervals across runs.

## Roadmap

- Record the cause of death of every evaluation game, and investigate the early failures.
- Better features: a flood fill that knows the body frees up over time, and real path lengths to the food and the tail.
- Training changes: n-step returns, lower final ε, starts from long snakes (curriculum).
- Add the classical baselines (rule-based policy, Hamiltonian cycle) as reproducible scripts, and a controlled comparison between versions.
- Compare against a policy-gradient method (PPO) as a separate experiment.

## Browser demo

`docs/` contains a static page: the environment, the 22 features and the dueling network forward pass re-implemented in JavaScript, plus the exported weights (`weights.js`). To update it after training a new model:

```bash
python export_weights.py models/best_model.pt demo_export
# copy demo_export/weights.js into docs/
```