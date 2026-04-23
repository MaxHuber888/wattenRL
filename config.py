# CS5180 Final Project
# Max Huber

# This file defines the PhaseConfig dataclass and all training phase configurations.

# IMPORTS
from dataclasses import dataclass, field
from typing import List, Optional

# PHASE CONFIG DATACLASS


# PHASE CONFIG
# Dataclass holding all hyperparameters for a single training phase.
@dataclass
class PhaseConfig:
    # Identity
    name: str = "unnamed"

    # Model type: "dqn" or "a2c"
    model_type: str = "a2c"

    # Environment
    blind_mode: bool = False

    # Opponents and teammate
    opponent_type: str = "heuristic"  # "random", "heuristic", "snapshot_pool"
    teammate_type: str = "clone"  # "clone", "heuristic", "random"

    # Checkpoint to load weights from (None = start fresh)
    load_from: Optional[str] = None

    # Training budget
    total_episodes: int = 200_000
    updates_per_episode: int = 1

    # Network
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"

    # Replay buffer
    replay_buffer_size: int = 100_000
    batch_size: int = 64

    # Optimization
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gradient_clip: float = 10.0
    target_update_freq: int = 100

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 100_000

    # A2C-specific (ignored for DQN)
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    critic_warmup_episodes: int = 10000
    batch_episodes: int = 32
    rollouts_per_deal: int = 1

    # Snapshot pool (for opponent_type="snapshot_pool")
    snapshot_interval: int = 10_000
    snapshot_pool_size: int = 20
    snapshot_recent_bias: float = 0.5

    # Evaluation
    eval_interval: int = 10_000
    eval_games: int = 1000
    eval_vs: List[str] = field(default_factory=lambda: ["heuristic", "random"])

    # Training mode
    training_mode: str = (
        "rl"  # "rl", "supervised", "critic_supervised", "trump_predictor"
    )

    # Supervised (behavioral cloning) settings
    supervised_teacher: str = "heuristic"
    supervised_data_episodes: int = 50_000
    supervised_epochs: int = 20
    supervised_batch_size: int = 256
    val_episodes: int = 0  # held-out episodes for validation (trump_predictor only)

    # Advancement (None = run until total_episodes)
    advance_metric: Optional[str] = None
    advance_threshold: Optional[float] = None

    # Early stopping
    early_stopping_patience: int = 5

    # Logging and checkpoints
    log_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    seed: Optional[int] = 42


# TRAINING PHASES

# SUPERVISED ACTOR
# Behavioral cloning from heuristic data to warm-start the actor network.
SUPERVISED_ACTOR = PhaseConfig(
    name="supervised_actor",
    training_mode="supervised",
    blind_mode=False,
    opponent_type="heuristic",
    teammate_type="clone",
    load_from=None,
    total_episodes=1,
    hidden_layers=[128, 64],
    gamma=1.0,
    target_update_freq=100,
    learning_rate=1e-3,
    supervised_teacher="heuristic",
    supervised_data_episodes=50_000,
    supervised_epochs=100,
    supervised_batch_size=256,
    eval_interval=1,
    eval_games=1000,
    eval_vs=["heuristic", "random"],
    advance_metric="action_accuracy",
    advance_threshold=0.70,
    checkpoint_dir="checkpoints/supervised_actor",
)

# STANDARD VS HEURISTIC
# A2C fine-tuning with warm-started actor against heuristic opponents.
STANDARD_VS_HEURISTIC = PhaseConfig(
    name="standard_vs_heuristic",
    blind_mode=False,
    opponent_type="heuristic",
    teammate_type="heuristic",
    load_from="checkpoints/supervised_warmstart/best.pt",
    total_episodes=200_000,
    updates_per_episode=4,
    hidden_layers=[128, 64],
    gamma=1.0,
    target_update_freq=100,
    learning_rate=1e-4,
    epsilon_start=0.1,
    epsilon_end=0.05,
    epsilon_decay_episodes=50_000,
    batch_episodes=32,
    eval_vs=["heuristic", "random"],
    advance_metric="win_rate_vs_heuristic",
    advance_threshold=0.60,
    checkpoint_dir="checkpoints/standard_vs_heuristic",
)

# SUPERVISED CRITIC
# Supervised critic pretraining: learns V(s) from heuristic trajectories.
SUPERVISED_CRITIC = PhaseConfig(
    name="SUPERVISED_CRITIC",
    model_type="a2c",
    training_mode="critic_supervised",
    blind_mode=False,
    load_from="checkpoints/supervised_actor/best.pt",
    hidden_layers=[128, 64],
    gamma=1.0,
    learning_rate=3e-4,
    gradient_clip=10.0,
    supervised_data_episodes=300_000,
    supervised_epochs=200,
    supervised_batch_size=256,
    eval_interval=5,
    checkpoint_dir="checkpoints/supervised_critic",
)

# A2C WARMSTART WITH CRITIC
# A2C fine-tuning with both actor and critic warm-started, no critic warmup phase.
A2C_WARMSTART_WITH_CRITIC = PhaseConfig(
    name="a2c_warmstart_with_critic",
    model_type="a2c",
    blind_mode=False,
    opponent_type="heuristic",
    teammate_type="heuristic",
    load_from="checkpoints/SUPERVISED_CRITIC/best.pt",
    total_episodes=500_000,
    hidden_layers=[128, 64],
    gamma=1.0,
    learning_rate=3e-4,
    entropy_coeff=0.01,
    critic_coeff=0.5,
    critic_warmup_episodes=0,
    batch_episodes=64,
    eval_interval=10_000,
    eval_games=1000,
    eval_vs=["heuristic", "random"],
    advance_metric=None,
    checkpoint_dir="checkpoints/a2c_warmstart_with_critic",
)

# BASELINE A2C VS HEURISTIC
# Baseline A2C trained from scratch against heuristic (no warm-start).
BASELINE_A2C_VS_HEURISTIC = PhaseConfig(
    name="baseline_a2c_vs_heuristic",
    model_type="a2c",
    blind_mode=False,
    opponent_type="heuristic",
    teammate_type="heuristic",
    load_from=None,
    total_episodes=100_000,
    hidden_layers=[128, 64],
    gamma=1.0,
    learning_rate=3e-4,
    entropy_coeff=0.01,
    critic_coeff=0.5,
    critic_warmup_episodes=0,
    batch_episodes=32,
    eval_interval=10_000,
    eval_games=1000,
    eval_vs=["heuristic", "random"],
    advance_metric=None,
    checkpoint_dir="checkpoints/baseline_a2c_vs_heuristic",
)

# BASELINE DQN VS HEURISTIC
# Baseline Deep Monte Carlo DQN trained from scratch against heuristic.
BASELINE_DQN_VS_HEURISTIC = PhaseConfig(
    name="baseline_dqn_vs_heuristic",
    model_type="dqn",
    blind_mode=False,
    opponent_type="heuristic",
    teammate_type="heuristic",
    load_from=None,
    total_episodes=100_000,
    hidden_layers=[128, 64],
    gamma=1.0,
    learning_rate=3e-4,
    replay_buffer_size=100_000,
    batch_size=64,
    updates_per_episode=4,
    target_update_freq=100,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=50_000,
    eval_interval=10_000,
    eval_games=1000,
    eval_vs=["heuristic", "random"],
    advance_metric=None,
    checkpoint_dir="checkpoints/baseline_dqn_vs_heuristic",
)

# TRUMP PREDICTOR
# Supervised training of the TrumpPredictor MLP to infer trump from trick history.
TRUMP_PREDICTOR = PhaseConfig(
    name="trump_predictor",
    training_mode="trump_predictor",
    model_type="a2c",
    supervised_data_episodes=300_000,
    val_episodes=100_000,
    supervised_batch_size=512,
    supervised_epochs=100,
    eval_interval=5,
    hidden_layers=[256, 128, 64],
    learning_rate=3e-4,
    checkpoint_dir="checkpoints/trump_predictor",
    seed=42,
)

# REGISTRY

# Registry: look up phase by name
PHASES = {
    "supervised_actor": SUPERVISED_ACTOR,
    "standard_vs_heuristic": STANDARD_VS_HEURISTIC,
    "supervised_critic": SUPERVISED_CRITIC,
    "a2c_warmstart_with_critic": A2C_WARMSTART_WITH_CRITIC,
    "baseline_a2c_vs_heuristic": BASELINE_A2C_VS_HEURISTIC,
    "baseline_dqn_vs_heuristic": BASELINE_DQN_VS_HEURISTIC,
    "trump_predictor": TRUMP_PREDICTOR,
}

# VALIDATION


# VALIDATE CONFIG
# Raises ValueError if any required field in cfg has an invalid value.
def validate_config(cfg: PhaseConfig) -> None:
    if cfg.model_type not in ("dqn", "a2c"):
        raise ValueError(f"model_type must be dqn or a2c, got {cfg.model_type}")
    if cfg.total_episodes <= 0:
        raise ValueError(f"total_episodes must be positive, got {cfg.total_episodes}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {cfg.batch_size}")
    if not (0.0 < cfg.gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], got {cfg.gamma}")
    if cfg.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {cfg.learning_rate}")
    if not (0.0 <= cfg.epsilon_start <= 1.0):
        raise ValueError(f"epsilon_start must be in [0, 1], got {cfg.epsilon_start}")
    if not (0.0 <= cfg.epsilon_end <= 1.0):
        raise ValueError(f"epsilon_end must be in [0, 1], got {cfg.epsilon_end}")
    if cfg.epsilon_start < cfg.epsilon_end:
        raise ValueError(
            f"epsilon_start ({cfg.epsilon_start}) < epsilon_end ({cfg.epsilon_end})"
        )
    if len(cfg.hidden_layers) == 0:
        raise ValueError("hidden_layers must have at least one layer")
    if cfg.opponent_type not in ("random", "heuristic", "snapshot_pool"):
        raise ValueError(
            f"opponent_type must be random/heuristic/snapshot_pool, got {cfg.opponent_type}"
        )
    if cfg.target_update_freq <= 0:
        raise ValueError(
            f"target_update_freq must be positive, got {cfg.target_update_freq}"
        )
    if cfg.early_stopping_patience <= 0:
        raise ValueError(
            f"early_stopping_patience must be > 0, got {cfg.early_stopping_patience}"
        )
    if cfg.training_mode not in (
        "rl",
        "supervised",
        "critic_supervised",
        "trump_predictor",
    ):
        raise ValueError(
            f"training_mode must be 'rl', 'supervised', 'critic_supervised', or 'trump_predictor', got {cfg.training_mode}"
        )
    if cfg.training_mode == "supervised":
        if cfg.supervised_teacher not in ("heuristic", "random"):
            raise ValueError(
                f"supervised_teacher must be 'heuristic' or 'random', got {cfg.supervised_teacher}"
            )
        if cfg.supervised_data_episodes <= 0:
            raise ValueError(
                f"supervised_data_episodes must be positive, got {cfg.supervised_data_episodes}"
            )
        if cfg.supervised_epochs <= 0:
            raise ValueError(
                f"supervised_epochs must be positive, got {cfg.supervised_epochs}"
            )
        if cfg.supervised_batch_size <= 0:
            raise ValueError(
                f"supervised_batch_size must be positive, got {cfg.supervised_batch_size}"
            )


# MAKE DEFAULT CONFIG
# Returns a default PhaseConfig for use in tests.
def make_default_config() -> PhaseConfig:
    return PhaseConfig(
        total_episodes=100_000,
        batch_size=64,
        replay_buffer_size=100_000,
        gamma=1.0,
        learning_rate=3e-4,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=50_000,
        hidden_layers=[128, 64],
        eval_interval=1_000,
        eval_games=500,
        opponent_type="heuristic",
        blind_mode=True,
    )


# MAIN
if __name__ == "__main__":
    print("Available phases:")
    for name, cfg in PHASES.items():
        validate_config(cfg)
        print(f"  {name:35s}  mode={cfg.training_mode:<20}  blind={cfg.blind_mode}")
