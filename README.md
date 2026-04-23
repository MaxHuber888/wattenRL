# Watten RL

RL agents for Watten, a 4-player Bavarian trick-taking card game. Trains A2C and DQN agents with behavioral cloning warm-start, supervised critic pretraining, and RL fine-tuning. Includes a supervised TrumpPredictor that infers the hidden trump from trick history.

## Files

- `train.py` - training entry point, runs any phase by name
- `plot_metrics.py` - plots training metrics from CSV files in `checkpoints/`
- `agents.py` - agent classes (Random, Heuristic, DQN, A2C, TrumpPredictor, A2CWithTrumpPredictor) and observation utilities
- `env.py` - `WattenEnv` Gymnasium environment (336-bit observation, Discrete(32) action)
- `cards.py` - card IDs, suit/rank encoding, card strength functions
- `config.py` - `PhaseConfig` dataclass and all phase configs in `PHASES`
- `replay_buffer.py` - circular experience replay buffer for DQN
- `old/` - archived scripts not needed for training

## Training

```bash
source .venv/bin/activate

python train.py --phase supervised_warmstart        # BC warm-start (actor only)
python train.py --phase standard_vs_heuristic       # A2C fine-tuning, warm-started actor
python train.py --phase critic_pretrain             # supervised critic pretraining
python train.py --phase a2c_warmstart_with_critic   # A2C fine-tuning, warm-started actor + critic
python train.py --phase baseline_a2c_vs_heuristic   # A2C baseline from scratch
python train.py --phase baseline_dqn_vs_heuristic   # DQN baseline from scratch
python train.py --phase trump_predictor             # supervised trump inference MLP
python train.py --list-phases                       # list all phases
```

Checkpoints go to `checkpoints/<phase>/a2c/` or `checkpoints/<phase>/dqn/` as `best.pt` and `final.pt`. Metrics are logged to `metrics.csv` in the same directory.

## Plotting

```bash
python plot_metrics.py checkpoints/
python plot_metrics.py checkpoints/baseline_a2c_vs_heuristic/a2c/metrics.csv
```

Plots are saved as PNGs in a `plots/` subdirectory next to each CSV.

## Environment

Observation (336 bits): `trump(12) + hand(32) + seat(4) + team0_tricks(4) + team1_tricks(4) + history(4x56) + current_trick(56)`

Action: `Discrete(32)` - play any card by ID. Reward: `+/-0.3` per trick, `+/-1.3` at round end.

In blind mode, only the dealer and forehand see the trump encoding; other players see zeros.

## Dependencies

```
numpy, gymnasium, torch, matplotlib
```
