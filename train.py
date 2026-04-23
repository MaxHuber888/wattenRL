# CS5180 Final Project
# Max Huber

# This file trains Watten RL agents and supervised networks.
# Run with --phase <name> to select a training configuration.

# IMPORTS
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents import (
    CURRENT_TRICK_START,
    HAND_START,
    OBS_SIZE,
    TEAM_TRICKS_START,
    TRICK_SLOT_SIZE,
    TRUMP_SIZE,
    A2CAgent,
    A2CWithTrumpPredictor,
    DQNAgent,
    HeuristicAgent,
    QNetwork,
    RandomAgent,
    TrumpPredictor,
    parse_hand,
)
from cards import DECK_SIZE, hand_strength
from config import PHASES, PhaseConfig, validate_config
from env import NUM_PLAYERS, TEAMS, WattenEnv
from replay_buffer import ReplayBuffer

# HELPERS


# RESOLVE LOAD PATH
# Resolves a checkpoint load path by inserting the model_type subdirectory.
def _resolve_load_path(base_path: str, model_type: str) -> str:
    if base_path is None:
        return None
    dirname = os.path.dirname(base_path)
    basename = os.path.basename(base_path)
    return os.path.join(dirname, model_type, basename)


# MAKE OPPONENTS
# Creates opponent agents for seats 1 and 3 based on the phase config.
def make_opponents(cfg: PhaseConfig, snapshot_dir: str = None):
    if cfg.opponent_type == "random":
        return RandomAgent(), RandomAgent()
    elif cfg.opponent_type == "heuristic":
        return HeuristicAgent(), HeuristicAgent()
    elif cfg.opponent_type == "snapshot_pool":
        # Start with heuristic — snapshots added during training
        return HeuristicAgent(), HeuristicAgent()
    else:
        raise ValueError(f"Unknown opponent_type: {cfg.opponent_type}")


# MAKE TEAMMATE
# Creates the teammate agent for seat 2 based on the phase config.
def make_teammate(cfg: PhaseConfig, learner):
    if cfg.teammate_type == "clone":
        return learner.clone()
    elif cfg.teammate_type == "heuristic":
        return HeuristicAgent()
    elif cfg.teammate_type == "random":
        return RandomAgent()
    else:
        raise ValueError(f"Unknown teammate_type: {cfg.teammate_type}")


# LOAD SNAPSHOT OPPONENT
# Loads a random past agent snapshot from the pool as an opponent, with recency bias.
def load_snapshot_opponent(snapshot_dir: str, cfg: PhaseConfig):
    if not os.path.exists(snapshot_dir):
        return None

    snapshots = sorted(
        [
            f
            for f in os.listdir(snapshot_dir)
            if f.startswith("snap_") and f.endswith(".pt")
        ]
    )
    if not snapshots:
        return None

    # Bias toward recent snapshots
    if np.random.random() < cfg.snapshot_recent_bias:
        chosen = snapshots[-1]
    else:
        chosen = np.random.choice(snapshots)

    snap_path = os.path.join(snapshot_dir, chosen)
    if cfg.model_type == "a2c":
        opponent = A2CAgent(hidden_layers=cfg.hidden_layers)
        opponent.load(snap_path)
        return opponent
    else:
        buf = ReplayBuffer(capacity=1, obs_size=OBS_SIZE)  # dummy buffer
        opponent = DQNAgent(
            replay_buffer=buf,
            hidden_layers=cfg.hidden_layers,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
        opponent.load(snap_path)
        opponent.epsilon = 0.0
        return opponent


# EVALUATE
# Runs evaluation games against each opponent type and returns a dict of win rates.
def evaluate(
    learner, cfg: PhaseConfig, eval_seed: int = 10000, model_type: str = "dqn"
) -> dict:
    results = {}

    for opp_name in cfg.eval_vs:
        env = WattenEnv(blind_mode=cfg.blind_mode)
        teammate = learner.clone()
        wins = 0

        if opp_name == "random":
            opp1, opp3 = RandomAgent(), RandomAgent()
        elif opp_name == "heuristic":
            opp1, opp3 = HeuristicAgent(), HeuristicAgent()
        else:
            continue

        agents = {0: learner, 1: opp1, 2: teammate, 3: opp3}

        for game in range(cfg.eval_games):
            obs, info = env.reset(seed=eval_seed + game)

            while not env.done:
                p = env.current_player
                if p in (0, 2):
                    if model_type == "dqn":
                        action = agents[p].act(obs, epsilon=0.0)
                    else:
                        result = agents[p].act(obs, greedy=True)
                        action = result[0]  # A2C returns (action, log_prob, value)
                else:
                    action = agents[p].act(obs)
                obs, reward, terminated, truncated, info = env.step(action)

            if reward > 0:
                wins += 1

        results[f"win_rate_vs_{opp_name}"] = wins / cfg.eval_games

    # Also eval in the opposite blind mode for cross-comparison
    other_blind = not cfg.blind_mode
    env_cross = WattenEnv(blind_mode=other_blind)
    teammate_cross = learner.clone()
    opp1_cross, opp3_cross = HeuristicAgent(), HeuristicAgent()
    cross_wins = 0

    for game in range(cfg.eval_games):
        obs, info = env_cross.reset(seed=eval_seed + cfg.eval_games + game)

        while not env_cross.done:
            p = env_cross.current_player
            if p in (0, 2):
                if model_type == "dqn":
                    action = (
                        learner.act(obs, epsilon=0.0)
                        if p == 0
                        else teammate_cross.act(obs, epsilon=0.0)
                    )
                else:
                    result = (
                        learner.act(obs, greedy=True)
                        if p == 0
                        else teammate_cross.act(obs, greedy=True)
                    )
                    action = result[0]  # A2C returns (action, log_prob, value)
            else:
                action = opp1_cross.act(obs) if p == 1 else opp3_cross.act(obs)
            obs, reward, terminated, truncated, info = env_cross.step(action)

        if reward > 0:
            cross_wins += 1

    mode_label = "blind" if other_blind else "standard"
    results[f"win_rate_vs_heuristic_{mode_label}_cross"] = cross_wins / cfg.eval_games

    return results


# SUPERVISED TRAINING


# COLLECT SUPERVISED DATA
# Collects (obs, action) pairs from teacher agent play for behavioral cloning.
def _collect_supervised_data(cfg: PhaseConfig) -> tuple:
    env = WattenEnv(blind_mode=cfg.blind_mode)

    if cfg.supervised_teacher == "heuristic":
        teacher = HeuristicAgent()
    else:
        teacher = RandomAgent()

    observations = []
    actions = []

    n_rounds = cfg.supervised_data_episodes
    print(f"  Collecting {n_rounds:,} rounds of teacher data...")

    for seed in range(n_rounds):
        if seed % 10000 == 0:
            print(f"sProgress: {seed:,}/{n_rounds:,}")
        obs, _ = env.reset(seed=seed)

        while not env.done:
            action = teacher.act(obs)
            observations.append(obs.copy())
            actions.append(action)
            obs, reward, done, _, _ = env.step(action)

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


# CRITIC PRETRAINING


# TRAIN CRITIC SUPERVISED
# Pre-trains the A2C critic with MSE loss on Monte Carlo returns from heuristic trajectories.
def train_critic_supervised(cfg: PhaseConfig, load_override: str = None):
    validate_config(cfg)
    print(f"Phase: {cfg.name} (CRITIC SUPERVISED)")
    print(f"blind_mode = {cfg.blind_mode}")
    print(f"data_episodes = {cfg.supervised_data_episodes:,}")
    print(f"epochs = {cfg.supervised_epochs}, batch_size = {cfg.supervised_batch_size}")

    ckpt_dir = os.path.join(cfg.checkpoint_dir, "a2c")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create agent and optionally load existing weights (e.g. BC actor)
    agent = A2CAgent(
        hidden_layers=cfg.hidden_layers,
        actor_learning_rate=cfg.learning_rate,
        critic_learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        entropy_coeff=cfg.entropy_coeff,
        critic_coeff=cfg.critic_coeff,
        gradient_clip=cfg.gradient_clip,
    )

    load_path = load_override or cfg.load_from
    if load_path:
        candidates = [_resolve_load_path(load_path, "a2c"), load_path]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                print(f"Loading weights from: {candidate}")
                agent.load(candidate)
                break
        else:
            print(f"WARNING: load path not found: {load_path}. Starting fresh.")

    # Step 1: Collect (obs, MC_return) pairs from all-heuristic games
    print(f"\n[Step 1] Collecting {cfg.supervised_data_episodes:,} heuristic games...")
    env = WattenEnv(blind_mode=cfg.blind_mode)
    heuristic = HeuristicAgent()

    all_obs = []
    all_returns = []

    for ep in range(cfg.supervised_data_episodes):
        if ep % 10_000 == 0:
            print(f"  Progress: {ep:,}/{cfg.supervised_data_episodes:,}")

        obs, _ = env.reset(seed=cfg.seed + ep if cfg.seed else None)

        seat0_obs = []
        rewards_ep = []
        pending_reward = 0.0

        while not env.done:
            p = env.current_player
            if p == 0:
                if (
                    seat0_obs
                ):  # finalize previous action's reward before recording new action
                    rewards_ep.append(pending_reward)
                    pending_reward = 0.0
                seat0_obs.append(obs.copy())
            obs, reward, _, _, _ = env.step(heuristic.act(obs))
            pending_reward += reward

        # Append reward for the last seat-0 action
        if seat0_obs:
            rewards_ep.append(pending_reward)

        # MC returns with gamma=1: G_t = sum of rewards_ep[t:]
        if seat0_obs:
            G = 0
            episode_returns = []
            for r in reversed(rewards_ep):
                G = r + cfg.gamma * G
                episode_returns.insert(0, G)
            all_obs.extend(seat0_obs)
            all_returns.extend(episode_returns)

    all_obs = np.array(all_obs, dtype=np.float32)
    all_returns = np.array(all_returns, dtype=np.float32)
    dataset_size = len(all_obs)
    print(f"Collected {dataset_size:,} (obs, return) pairs")
    print(
        f"Return range: [{all_returns.min():.3f}, {all_returns.max():.3f}]  "
        f"mean = {all_returns.mean():.3f}\n"
    )

    # ------------------------------------------------------------------
    # Step 2: Train critic (+ shared layers) via MSE
    # ------------------------------------------------------------------
    print(f"[Step 2] Training critic for {cfg.supervised_epochs} epochs...")
    device = next(agent.network.parameters()).device
    n_batches = (
        dataset_size + cfg.supervised_batch_size - 1
    ) // cfg.supervised_batch_size

    best_loss = float("inf")
    start_time = time.time()

    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file, fieldnames=["epoch", "critic_loss", "elapsed_s"]
    )
    csv_writer.writeheader()
    csv_file.flush()

    for epoch in range(cfg.supervised_epochs):
        epoch_start = time.time()
        perm = np.random.permutation(dataset_size)
        obs_shuffled = all_obs[perm]
        ret_shuffled = all_returns[perm]

        total_loss = 0.0
        for b in range(n_batches):
            s = b * cfg.supervised_batch_size
            e = min(s + cfg.supervised_batch_size, dataset_size)

            obs_b = torch.from_numpy(obs_shuffled[s:e]).float().to(device)
            ret_b = torch.from_numpy(ret_shuffled[s:e]).float().to(device)

            # Build hand mask (required by network forward, but only critic output used)
            hand_b = obs_shuffled[s:e, HAND_START : HAND_START + 32]
            hand_mask = torch.full((e - s, DECK_SIZE), -1e9, device=device)
            for i, hand_vec in enumerate(hand_b):
                hand_mask[i, np.where(hand_vec == 1.0)[0]] = 0.0

            _, value = agent.network(obs_b, hand_mask)
            loss = nn.functional.mse_loss(value.squeeze(1), ret_b)

            agent.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.network.parameters(), agent.gradient_clip)
            agent.critic_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:>3}/{cfg.supervised_epochs}:"
            f"critic_loss = {avg_loss:.4f}  ({epoch_time:.1f}s)"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            agent.save(best_path)
            print(f"NEW BEST: {best_loss:.4f} → saved to {best_path}")

        csv_writer.writerow(
            {"epoch": epoch + 1, "critic_loss": avg_loss, "elapsed_s": elapsed}
        )
        csv_file.flush()

    final_path = os.path.join(ckpt_dir, "final.pt")
    agent.save(final_path)
    csv_file.close()
    print(f"\nCritic pre-training finished. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {final_path}")
    return agent


# TRAIN SUPERVISED
# Trains the agent via behavioral cloning (cross-entropy) on teacher data.
def train_supervised(cfg: PhaseConfig, load_override: str = None):
    validate_config(cfg)
    model_type = cfg.model_type
    print(f"Phase: {cfg.name} (SUPERVISED, model={model_type})")
    print(f"blind_mode = {cfg.blind_mode}")
    print(f"teacher = {cfg.supervised_teacher}")
    print(f"data_episodes = {cfg.supervised_data_episodes:,}")
    print(f"epochs = {cfg.supervised_epochs}, batch_size = {cfg.supervised_batch_size}")

    # Checkpoint path includes model type
    ckpt_dir = os.path.join(cfg.checkpoint_dir, model_type)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Collect data (same for both model types)
    print("\n[Step 1] Collecting supervised data...")
    observations, actions = _collect_supervised_data(cfg)
    dataset_size = len(observations)
    print(f"Collected {dataset_size:,} (obs, action) pairs")
    print(f"Obs shape: {observations.shape}, dtype: {observations.dtype}")
    print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")

    device = torch.device("cpu")

    if model_type == "dqn":
        # DQN path: train QNetwork with cross-entropy on full 32-logit output
        network = QNetwork(OBS_SIZE, cfg.hidden_layers).to(device)
        optimizer = optim.Adam(network.parameters(), lr=cfg.learning_rate)

        # Load from checkpoint if specified
        load_path = load_override or cfg.load_from
        if load_path:
            # Try load_override directly, then try with model_type subdirectory
            candidates = []
            if load_override:
                candidates.append(load_override)
                candidates.append(_resolve_load_path(load_override, model_type))
            if cfg.load_from:
                candidates.append(_resolve_load_path(cfg.load_from, model_type))
                if not load_override:
                    candidates.append(cfg.load_from)
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    load_path = candidate
                    print(f"Loading weights from: {load_path}")
                    checkpoint = torch.load(
                        load_path, map_location=device, weights_only=False
                    )
                    network.load_state_dict(checkpoint["network"])
                    break
            else:
                print(f"WARNING: load path not found: {load_path}")
                print("Starting with fresh weights.")
                load_path = None

        # Training loop
        print(f"\n[Step 2] Training QNetwork for {cfg.supervised_epochs} epochs...")
        best_win_rate = 0.0
        start_time = time.time()

        csv_path = os.path.join(ckpt_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "loss",
                "action_accuracy",
                "win_rate_vs_heuristic",
                "win_rate_vs_random",
                "elapsed_s",
            ],
        )
        csv_writer.writeheader()
        csv_file.flush()

        def dqn_eval(epoch_num, avg_loss, epoch_time):
            nonlocal best_win_rate
            buf = ReplayBuffer(capacity=1, obs_size=OBS_SIZE)
            learner = DQNAgent(
                replay_buffer=buf,
                hidden_layers=cfg.hidden_layers,
                epsilon_start=0.0,
                epsilon_end=0.0,
            )
            learner.network = network
            learner.target_network = QNetwork(OBS_SIZE, cfg.hidden_layers).to(device)
            learner.target_network.load_state_dict(network.state_dict())

            eval_results = evaluate(learner, cfg, model_type=model_type)
            elapsed = time.time() - start_time

            network.eval()
            correct = 0
            with torch.no_grad():
                for b in range(0, dataset_size, cfg.supervised_batch_size):
                    obs_b = (
                        torch.from_numpy(
                            observations[b : b + cfg.supervised_batch_size]
                        )
                        .float()
                        .to(device)
                    )
                    act_b = actions[b : b + cfg.supervised_batch_size]
                    logits = network(obs_b)
                    hand_b = observations[
                        b : b + cfg.supervised_batch_size, HAND_START : HAND_START + 32
                    ]
                    mask = torch.full(logits.shape, -float("inf"), device=device)
                    for i, hand_vec in enumerate(hand_b):
                        mask[i, np.where(hand_vec == 1.0)[0]] = 0.0
                    preds = (logits + mask).argmax(dim=1).cpu().numpy()
                    correct += (preds == act_b).sum()
            network.train()
            action_accuracy = correct / dataset_size
            eval_results["action_accuracy"] = action_accuracy

            parts = [f"{k}={v:.3f}" for k, v in eval_results.items()]
            print(
                f"Epoch {epoch_num:>3}/{cfg.supervised_epochs}: "
                f"loss = {avg_loss:.4f}  "
                f"{'  '.join(parts)}  "
                f"time={epoch_time:.1f}s"
            )

            heuristic_wr = eval_results.get("win_rate_vs_heuristic", 0.0)
            if heuristic_wr > best_win_rate:
                best_win_rate = heuristic_wr
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save(
                    {
                        "network": network.state_dict(),
                        "target_network": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epsilon": 0.0,
                        "total_steps": epoch_num
                        * (dataset_size // cfg.supervised_batch_size),
                        "epoch": epoch_num,
                        "best_win_rate": best_win_rate,
                    },
                    best_path,
                )
                print(f"    NEW BEST: {best_win_rate:.3f} → saved to {best_path}")

            csv_writer.writerow(
                {
                    "epoch": epoch_num,
                    "loss": avg_loss,
                    "action_accuracy": action_accuracy,
                    "win_rate_vs_heuristic": eval_results.get(
                        "win_rate_vs_heuristic", 0.0
                    ),
                    "win_rate_vs_random": eval_results.get("win_rate_vs_random", 0.0),
                    "elapsed_s": elapsed,
                }
            )
            csv_file.flush()

            stop_metric = (
                eval_results.get(cfg.advance_metric, 0.0)
                if cfg.advance_metric
                else None
            )
            return stop_metric

        # Eval before any training
        dqn_eval(0, 0.0, 0.0)

        for epoch in range(cfg.supervised_epochs):
            epoch_start = time.time()

            perm = np.random.permutation(dataset_size)
            obs_shuffled = observations[perm]
            act_shuffled = actions[perm]

            n_batches = (
                dataset_size + cfg.supervised_batch_size - 1
            ) // cfg.supervised_batch_size
            total_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * cfg.supervised_batch_size
                end = min(start + cfg.supervised_batch_size, dataset_size)

                obs_batch = torch.from_numpy(obs_shuffled[start:end]).float().to(device)
                act_batch = torch.from_numpy(act_shuffled[start:end]).long().to(device)

                logits = network(obs_batch)

                hand_obs = obs_shuffled[start:end, HAND_START : HAND_START + 32]
                mask = torch.full((end - start, 32), -float("inf"), device=device)
                for i, hand_vec in enumerate(hand_obs):
                    hand_cards = np.where(hand_vec == 1.0)[0]
                    mask[i, hand_cards] = 0.0

                masked_logits = logits + mask
                loss = nn.functional.cross_entropy(masked_logits, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batches
            epoch_time = time.time() - epoch_start

            stop_metric = dqn_eval(epoch + 1, avg_loss, epoch_time)
            if (
                cfg.advance_threshold is not None
                and stop_metric is not None
                and stop_metric >= cfg.advance_threshold
            ):
                print(
                    f"  Reached {cfg.advance_metric}={stop_metric:.3f} >= {cfg.advance_threshold:.2f} — stopping early."
                )
                break

        final_path = os.path.join(ckpt_dir, "final.pt")
        torch.save(
            {
                "network": network.state_dict(),
                "target_network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epsilon": 0.0,
                "total_steps": cfg.supervised_epochs
                * (dataset_size // cfg.supervised_batch_size),
                "epoch": cfg.supervised_epochs,
                "best_win_rate": best_win_rate,
            },
            final_path,
        )
        csv_file.close()
        print(f"\nPhase {cfg.name} finished. Final checkpoint: {final_path}")
        print(f"Best win rate vs heuristic: {best_win_rate:.3f}")
        return network

    else:
        # A2C path: train A2CAgent with bc_update on actor head
        agent = A2CAgent(
            hidden_layers=cfg.hidden_layers,
            actor_learning_rate=cfg.learning_rate,
            critic_learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            entropy_coeff=cfg.entropy_coeff,
            critic_coeff=cfg.critic_coeff,
            gradient_clip=cfg.gradient_clip,
        )

        # Load from checkpoint if specified
        load_path = load_override or cfg.load_from
        if load_path:
            # Try load_override directly, then try with model_type subdirectory
            candidates = []
            if load_override:
                candidates.append(load_override)
                candidates.append(_resolve_load_path(load_override, model_type))
            if cfg.load_from:
                candidates.append(_resolve_load_path(cfg.load_from, model_type))
                if not load_override:
                    candidates.append(cfg.load_from)
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    load_path = candidate
                    print(f"Loading weights from: {load_path}")
                    agent.load(load_path)
                    break
            else:
                print(f"WARNING: load path not found: {load_path}")
                print("Starting with fresh weights.")
                load_path = None

        print(f"\n[Step 2] Training A2C agent for {cfg.supervised_epochs} epochs...")
        best_win_rate = 0.0
        start_time = time.time()

        csv_path = os.path.join(ckpt_dir, "metrics.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "loss",
                "action_accuracy",
                "win_rate_vs_heuristic",
                "win_rate_vs_random",
                "elapsed_s",
            ],
        )
        csv_writer.writeheader()
        csv_file.flush()

        def a2c_eval(epoch_num, avg_loss, epoch_time):
            nonlocal best_win_rate
            eval_results = evaluate(agent, cfg, model_type=model_type)
            elapsed = time.time() - start_time

            # Compute action-matching accuracy against teacher data
            correct = 0
            device_a2c = next(agent.network.parameters()).device
            agent.network.eval()
            with torch.no_grad():
                for b in range(0, dataset_size, cfg.supervised_batch_size):
                    obs_b = (
                        torch.from_numpy(
                            observations[b : b + cfg.supervised_batch_size]
                        )
                        .float()
                        .to(device_a2c)
                    )
                    act_b = actions[b : b + cfg.supervised_batch_size]
                    hand_b = observations[
                        b : b + cfg.supervised_batch_size, HAND_START : HAND_START + 32
                    ]
                    hand_mask = torch.full(
                        (len(obs_b), DECK_SIZE), -1e9, device=device_a2c
                    )
                    for i, hand_vec in enumerate(hand_b):
                        hand_mask[i, np.where(hand_vec == 1.0)[0]] = 0.0
                    probs, _ = agent.network(obs_b, hand_mask)
                    preds = probs.argmax(dim=1).cpu().numpy()
                    correct += (preds == act_b).sum()
            agent.network.train()
            action_accuracy = correct / dataset_size
            eval_results["action_accuracy"] = action_accuracy

            parts = [f"{k}={v:.3f}" for k, v in eval_results.items()]
            print(
                f"Epoch {epoch_num:>3}/{cfg.supervised_epochs}: "
                f"loss = {avg_loss:.4f}  "
                f"{'  '.join(parts)}  "
                f"time = {epoch_time:.1f}s"
            )

            heuristic_wr = eval_results.get("win_rate_vs_heuristic", 0.0)
            if heuristic_wr > best_win_rate:
                best_win_rate = heuristic_wr
                best_path = os.path.join(ckpt_dir, "best.pt")
                agent.save(best_path)
                print(f"    NEW BEST: {best_win_rate:.3f} → saved to {best_path}")

            csv_writer.writerow(
                {
                    "epoch": epoch_num,
                    "loss": avg_loss,
                    "action_accuracy": action_accuracy,
                    "win_rate_vs_heuristic": eval_results.get(
                        "win_rate_vs_heuristic", 0.0
                    ),
                    "win_rate_vs_random": eval_results.get("win_rate_vs_random", 0.0),
                    "elapsed_s": elapsed,
                }
            )
            csv_file.flush()

            stop_metric = (
                eval_results.get(cfg.advance_metric, 0.0)
                if cfg.advance_metric
                else None
            )
            return stop_metric

        # Eval before any training
        a2c_eval(0, 0.0, 0.0)

        for epoch in range(cfg.supervised_epochs):
            epoch_start = time.time()

            perm = np.random.permutation(dataset_size)
            obs_shuffled = observations[perm]
            act_shuffled = actions[perm]

            n_batches = (
                dataset_size + cfg.supervised_batch_size - 1
            ) // cfg.supervised_batch_size
            total_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * cfg.supervised_batch_size
                end = min(start + cfg.supervised_batch_size, dataset_size)

                obs_batch = obs_shuffled[start:end]
                act_batch = act_shuffled[start:end]

                loss = agent.bc_update(obs_batch, act_batch)
                total_loss += loss

            avg_loss = total_loss / n_batches
            epoch_time = time.time() - epoch_start

            stop_metric = a2c_eval(epoch + 1, avg_loss, epoch_time)
            if (
                cfg.advance_threshold is not None
                and stop_metric is not None
                and stop_metric >= cfg.advance_threshold
            ):
                print(
                    f"  Reached {cfg.advance_metric}={stop_metric:.3f} >= {cfg.advance_threshold:.2f} — stopping early."
                )
                break

        final_path = os.path.join(ckpt_dir, "final.pt")
        agent.save(final_path)
        csv_file.close()
        print(f"\nPhase {cfg.name} finished. Final checkpoint: {final_path}")
        print(f"Best win rate vs heuristic: {best_win_rate:.3f}")
        return agent


# TRAIN DISPATCHER


# TRAIN
# Routes training to the correct phase function based on cfg.training_mode.
def train(cfg: PhaseConfig, load_override: str = None):
    validate_config(cfg)
    model_type = cfg.model_type

    # Route to supervised or RL training
    if cfg.training_mode == "supervised":
        return train_supervised(cfg, load_override)

    if cfg.training_mode == "critic_supervised":
        return train_critic_supervised(cfg, load_override)

    if cfg.training_mode == "trump_predictor":
        return train_trump_predictor(cfg, load_override)

    if model_type == "dqn":
        return train_dqn(cfg, load_override)
    else:
        return train_a2c(cfg, load_override)


# DQN TRAINING


# TRAIN DQN
# Runs the Deep Monte Carlo DQN training phase.
def train_dqn(cfg: PhaseConfig, load_override: str = None):
    print(f"\n{'=' * 60}")
    print(f"Phase: {cfg.name} (DQN)")
    print(f"blind_mode = {cfg.blind_mode}, opponents = {cfg.opponent_type}")
    print(
        f"episodes = {cfg.total_episodes:,}, ε = {cfg.epsilon_start}→{cfg.epsilon_end}"
    )
    print(f"{'=' * 60}\n")

    # Create environment
    env = WattenEnv(blind_mode=cfg.blind_mode)

    buf = ReplayBuffer(
        capacity=cfg.replay_buffer_size,
        obs_size=OBS_SIZE,
    )

    # Create learner
    learner = DQNAgent(
        replay_buffer=buf,
        hidden_layers=cfg.hidden_layers,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_episodes,
        target_update_interval=cfg.target_update_freq,
        gradient_clip=cfg.gradient_clip,
    )

    # Load weights from previous phase or override
    load_path = load_override or cfg.load_from
    if load_path:
        candidates = []
        if load_override:
            candidates.append(load_override)
            candidates.append(_resolve_load_path(load_override, "dqn"))
        if cfg.load_from:
            candidates.append(_resolve_load_path(cfg.load_from, "dqn"))
            if not load_override:
                candidates.append(cfg.load_from)
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                load_path = candidate
                print(f"Loading weights from: {load_path}")
                learner.load(load_path)
                learner.epsilon = cfg.epsilon_start
                break
        else:
            print(f"WARNING: load path not found: {load_path}")
            print("Starting with fresh weights.")
            load_path = None

    # Create agents for all seats
    opp1, opp3 = make_opponents(cfg)
    teammate = make_teammate(cfg, learner)
    agents = {0: learner, 1: opp1, 2: teammate, 3: opp3}

    # Snapshot directory
    snapshot_dir = os.path.join(cfg.checkpoint_dir, "dqn", "snapshots")
    ckpt_dir = os.path.join(cfg.checkpoint_dir, "dqn")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    best_win_rate = 0.0
    total_losses = []
    episode_rewards = []
    start_time = time.time()

    # CSV logging
    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "win_rate_vs_heuristic",
            "win_rate_vs_random",
            "loss",
            "epsilon",
            "elapsed_s",
        ],
    )
    csv_writer.writeheader()
    csv_file.flush()

    for episode in range(cfg.total_episodes):
        obs, info = env.reset(seed=cfg.seed + episode if cfg.seed else None)

        # Collect per-step data for seat 0 (learner only)
        episode_transitions = []

        while not env.done:
            p = env.current_player

            # Get action
            if p == 0:
                action = learner.act(obs)
            elif p == 2:
                action = teammate.act(obs)
            else:
                action = agents[p].act(obs)

            # Store pre-step data for seat 0 (learner)
            if p == 0:
                episode_transitions.append(
                    {
                        "obs": obs.copy(),
                        "action": action,
                    }
                )

            obs, reward, terminated, truncated, info = env.step(action)

        # Round over — assign rewards and store transitions.
        # DMC: every transition gets the full round reward and done=True
        # so that no bootstrapping occurs (targets = reward only).
        team_0_reward = reward

        n = len(episode_transitions)
        for i, t in enumerate(episode_transitions):
            next_obs = obs if i == n - 1 else episode_transitions[i + 1]["obs"]
            buf.add(
                obs=t["obs"],
                action=t["action"],
                reward=team_0_reward,
                next_obs=next_obs,
                done=True,
            )

        episode_rewards.append(team_0_reward)

        # Gradient updates
        if len(buf) >= cfg.batch_size:
            for _ in range(cfg.updates_per_episode):
                loss_dict = learner.update()
                total_losses.append(loss_dict["total_loss"])

        # Epsilon decay
        learner.decay_epsilon()

        # Sync teammate clone
        if cfg.teammate_type == "clone" and episode % 100 == 0:
            learner.sync_clone(teammate)

        # Target network update
        if episode % cfg.target_update_freq == 0:
            learner.update_target_network()

        # Save snapshot for snapshot pool
        if (
            cfg.opponent_type == "snapshot_pool"
            and episode % cfg.snapshot_interval == 0
            and episode > 0
        ):
            os.makedirs(snapshot_dir, exist_ok=True)
            snap_path = os.path.join(snapshot_dir, f"snap_{episode}.pt")
            learner.save(snap_path)

            # Refresh opponents from pool
            loaded_opp = load_snapshot_opponent(snapshot_dir, cfg)
            if loaded_opp is not None:
                opp1 = loaded_opp
                opp3 = loaded_opp.clone()
                agents[1] = opp1
                agents[3] = opp3

        # Logging
        if episode % cfg.log_interval == 0 and episode > 0:
            recent_rewards = episode_rewards[-cfg.log_interval :]
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            recent_loss = (
                np.mean(total_losses[-cfg.log_interval :]) if total_losses else 0.0
            )
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0

            print(
                f"[{episode:>7,}/{cfg.total_episodes:,}] "
                f"win_rate = {win_rate:.3f}  "
                f"loss = {recent_loss:.4f}  "
                f"ε = {learner.epsilon:.3f}  "
                f"buf = {len(buf):,}  "
                f"eps/s = {eps_per_sec:.0f}"
            )

        # Evaluation
        if episode % cfg.eval_interval == 0 and episode > 0:
            eval_results = evaluate(learner, cfg, model_type="dqn")
            parts = [f"{k}={v:.3f}" for k, v in eval_results.items()]
            print(f"EVAL: {', '.join(parts)}")

            # Save best checkpoint
            heuristic_wr = eval_results.get("win_rate_vs_heuristic", 0.0)
            if heuristic_wr > best_win_rate:
                best_win_rate = heuristic_wr
                best_path = os.path.join(ckpt_dir, "best.pt")
                learner.save(best_path)
                print(f"NEW BEST: {best_win_rate:.3f} → saved to {best_path}")

            # Check advancement (no auto-break — just print)
            if cfg.advance_metric and cfg.advance_threshold:
                metric_val = eval_results.get(cfg.advance_metric, 0.0)
                if metric_val >= cfg.advance_threshold:
                    print(
                        f"\n[ADVANCEMENT METRIC] {cfg.advance_metric}={metric_val:.3f} >= {cfg.advance_threshold}"
                    )

            # Early stopping: if rolling loss hasn't improved in 3 consecutive evals, stop
            recent_loss = (
                np.mean(total_losses[-cfg.log_interval :]) if total_losses else 0.0
            )
            eval_row = {
                "episode": episode,
                "win_rate_vs_heuristic": eval_results.get("win_rate_vs_heuristic", 0.0),
                "win_rate_vs_random": eval_results.get("win_rate_vs_random", 0.0),
                "loss": recent_loss,
                "epsilon": learner.epsilon,
                "elapsed_s": time.time() - start_time,
            }
            csv_writer.writerow(eval_row)
            csv_file.flush()

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, "final.pt")
    learner.save(final_path)
    csv_file.close()
    print(f"\nPhase {cfg.name} finished. Final checkpoint: {final_path}")
    print(f"Best win rate vs heuristic: {best_win_rate:.3f}")

    return learner


# A2C TRAINING


# TRAIN A2C
# Runs the A2C reinforcement learning training phase with batch Monte Carlo returns.
def train_a2c(cfg: PhaseConfig, load_override: str = None):
    print(f"Phase: {cfg.name} (A2C)")
    print(f"blind_mode = {cfg.blind_mode}, opponents = {cfg.opponent_type}")
    print(f"episodes = {cfg.total_episodes:,}")
    print(
        f"gamma = {cfg.gamma}, entropy_coeff = {cfg.entropy_coeff}, critic_coeff = {cfg.critic_coeff}"
    )

    env = WattenEnv(blind_mode=cfg.blind_mode)

    # Create A2C agent
    agent = A2CAgent(
        hidden_layers=cfg.hidden_layers,
        actor_learning_rate=cfg.learning_rate,
        critic_learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        entropy_coeff=cfg.entropy_coeff,
        critic_coeff=cfg.critic_coeff,
        gradient_clip=cfg.gradient_clip,
    )

    # Load weights from previous phase or override
    load_path = load_override or cfg.load_from
    if load_path:
        # Try load_override directly, then try with model_type subdirectory
        candidates = []
        if load_override:
            candidates.append(load_override)
            candidates.append(_resolve_load_path(load_override, "a2c"))
        if cfg.load_from:
            candidates.append(_resolve_load_path(cfg.load_from, "a2c"))
            if not load_override:
                candidates.append(cfg.load_from)
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                load_path = candidate
                print(f"Loading weights from: {load_path}")
                agent.load(load_path)
                break
        else:
            print(f"WARNING: load path not found: {load_path}")
            print("Starting with fresh weights.")
            load_path = None

    # Create opponents and teammate
    opp1, opp3 = make_opponents(cfg)
    teammate = make_teammate(cfg, agent)
    agents = {0: agent, 1: opp1, 2: teammate, 3: opp3}

    # Checkpoint directory with model type
    ckpt_dir = os.path.join(cfg.checkpoint_dir, "a2c")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Snapshot pool directory for self-play phases
    snapshot_dir = os.path.join(cfg.checkpoint_dir, "a2c", "snapshots")

    # Training loop
    best_win_rate = 0.0
    total_losses = []
    episode_rewards = []
    start_time = time.time()

    batch_episodes = cfg.batch_episodes
    critic_warmup_episodes = cfg.critic_warmup_episodes

    # Track last loss dict for CSV logging
    last_loss_dict = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }

    # CSV logging
    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "win_rate_vs_heuristic",
            "win_rate_vs_random",
            "actor_loss",
            "critic_loss",
            "entropy",
            "total_loss",
            "elapsed_s",
        ],
    )
    csv_writer.writeheader()
    csv_file.flush()

    episode_count = 0

    # Batch accumulation buffers (flushed every batch_episodes)
    batch_log_probs = []
    batch_values = []
    batch_returns = []
    batch_count = 0

    while episode_count < cfg.total_episodes:
        current_seed = cfg.seed + episode_count if cfg.seed else None
        obs, info = env.reset(seed=current_seed)

        # Collect trajectory for seat 0
        log_probs_ep = []
        values_ep = []
        rewards_ep = []
        pending_reward = 0.0

        while not env.done:
            p = env.current_player

            if p == 0:
                if (
                    log_probs_ep
                ):  # finalize previous action's reward before recording new action
                    rewards_ep.append(pending_reward)
                    pending_reward = 0.0
                action, log_prob, value = agent.act(obs, greedy=False)
                log_probs_ep.append(log_prob)
                values_ep.append(value)
            elif p == 2:
                if isinstance(teammate, A2CAgent):
                    result = teammate.act(obs, greedy=False)
                    action = result[0]
                else:
                    action = teammate.act(obs)
            else:
                action = agents[p].act(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            pending_reward += reward

        # Append reward for the last seat-0 action
        if log_probs_ep:
            rewards_ep.append(pending_reward)

        episode_rewards.append(sum(rewards_ep) if rewards_ep else 0.0)
        episode_count += 1
        batch_count += 1

        # Accumulate returns for this episode
        if log_probs_ep:
            G = 0
            episode_returns = []
            for r in reversed(rewards_ep):
                G = r + cfg.gamma * G
                episode_returns.insert(0, G)
            batch_log_probs.extend(log_probs_ep)
            batch_values.extend(values_ep)
            batch_returns.append(torch.tensor(episode_returns, dtype=torch.float32))

        # Sync teammate clone every 100 episodes
        if cfg.teammate_type == "clone" and episode_count % 100 == 0:
            teammate.network.load_state_dict(agent.network.state_dict())

        # Save snapshot and refresh opponents from pool (snapshot_pool phases only)
        if (
            cfg.opponent_type == "snapshot_pool"
            and episode_count % cfg.snapshot_interval == 0
            and episode_count > 0
        ):
            os.makedirs(snapshot_dir, exist_ok=True)
            snap_path = os.path.join(snapshot_dir, f"snap_{episode_count}.pt")
            agent.save(snap_path)

            loaded_opp = load_snapshot_opponent(snapshot_dir, cfg)
            if loaded_opp is not None:
                opp1 = loaded_opp
                opp3 = loaded_opp.clone()
                agents[1] = opp1
                agents[3] = opp3

        # Update every batch_episodes
        if batch_count >= batch_episodes:
            actor_frozen = episode_count < critic_warmup_episodes

            if (
                episode_count >= critic_warmup_episodes
                and (episode_count - batch_count) < critic_warmup_episodes
            ):
                print(
                    f"\n[CRITIC WARMUP COMPLETE at episode {episode_count}] Actor now training."
                )

            if batch_log_probs:
                stacked_log_probs = torch.stack(batch_log_probs)
                stacked_values = torch.stack(batch_values)
                stacked_returns = torch.cat(batch_returns)
                last_loss_dict = agent.update(
                    stacked_log_probs,
                    stacked_values,
                    stacked_returns,
                    actor_frozen=actor_frozen,
                )
                total_losses.append(last_loss_dict["total_loss"])

            batch_log_probs = []
            batch_values = []
            batch_returns = []
            batch_count = 0

        # Logging
        if episode_count % cfg.log_interval == 0:
            recent_rewards = episode_rewards[-cfg.log_interval :]
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            recent_loss = (
                np.mean(total_losses[-cfg.log_interval :]) if total_losses else 0.0
            )
            elapsed = time.time() - start_time
            eps_per_sec = episode_count / elapsed if elapsed > 0 else 0
            pct = episode_count / cfg.total_episodes * 100
            actor_frozen = episode_count < critic_warmup_episodes
            frozen_str = " [FROZEN]" if actor_frozen else ""

            print(
                f"[{episode_count:>7,}/{cfg.total_episodes:,} ({pct:5.1f}%)]"
                f"win_rate = {win_rate:.3f}  loss = {recent_loss:.4f}"
                f"({eps_per_sec:.1f} ep/s){frozen_str}"
            )

        # Evaluation
        if episode_count % cfg.eval_interval == 0 and episode_count > 0:
            eval_results = evaluate(agent, cfg, model_type="a2c")
            parts = [f"{k}={v:.3f}" for k, v in eval_results.items()]
            print(f"EVAL: {', '.join(parts)}")

            heuristic_wr = eval_results.get("win_rate_vs_heuristic", 0.0)
            if heuristic_wr > best_win_rate:
                best_win_rate = heuristic_wr
                best_path = os.path.join(ckpt_dir, "best.pt")
                agent.save(best_path)
                print(f"NEW BEST: {best_win_rate:.3f} → saved to {best_path}")

            recent_loss = (
                np.mean(total_losses[-cfg.log_interval :]) if total_losses else 0.0
            )
            eval_row = {
                "episode": episode_count,
                "win_rate_vs_heuristic": eval_results.get("win_rate_vs_heuristic", 0.0),
                "win_rate_vs_random": eval_results.get("win_rate_vs_random", 0.0),
                "actor_loss": last_loss_dict.get("actor_loss", 0.0),
                "critic_loss": last_loss_dict.get("critic_loss", 0.0),
                "entropy": last_loss_dict.get("entropy", 0.0),
                "total_loss": recent_loss,
                "elapsed_s": time.time() - start_time,
            }
            csv_writer.writerow(eval_row)
            csv_file.flush()

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, "final.pt")
    agent.save(final_path)
    csv_file.close()
    print(f"\nPhase {cfg.name} finished. Final checkpoint: {final_path}")
    print(f"Best win rate vs heuristic: {best_win_rate:.3f}")

    return agent


# TRUMP PREDICTOR TRAINING


# COLLECT TRUMP PREDICTOR DATA
# Collects (observation, true_trump_card_id) pairs from non-blind heuristic games.
def _collect_trump_predictor_data(num_episodes: int, seed: int = None) -> tuple:
    env = WattenEnv(blind_mode=False)
    heuristic = HeuristicAgent()
    observations = []
    trump_cards = []

    for ep in range(num_episodes):
        if ep % 100_000 == 0:
            print(f"  Progress: {ep:,}/{num_episodes:,}")
        obs, info = env.reset(seed=seed + ep if seed else None)
        trump_suit = env.trump_suit
        trump_rank = env.trump_rank
        true_trump_card = get_card_id(trump_suit, trump_rank)
        while not env.done:
            observations.append(obs.copy())
            trump_cards.append(true_trump_card)
            action = heuristic.act(obs)
            obs, _, _, _, _ = env.step(action)

    return np.array(observations, dtype=np.float32), np.array(
        trump_cards, dtype=np.int64
    )


# EVAL TRUMP PREDICTOR
# Computes top-1 and top-3 validation accuracy in batches.
def _eval_trump_predictor(
    predictor, obs: np.ndarray, trump: np.ndarray, batch_size: int = 2048
) -> tuple:

    device = next(predictor.network.parameters()).device
    top1_correct = 0
    top3_correct = 0
    n = len(obs)
    input_slice = obs[:, TEAM_TRICKS_START : CURRENT_TRICK_START + TRICK_SLOT_SIZE]

    predictor.network.eval()
    with torch.no_grad():
        for b in range(0, n, batch_size):
            obs_b = torch.from_numpy(input_slice[b : b + batch_size]).float().to(device)
            trump_b = trump[b : b + batch_size]
            logits = predictor.network(obs_b)
            preds = logits.argmax(dim=1).cpu().numpy()
            top1_correct += (preds == trump_b).sum()
            top3_idx = logits.topk(3, dim=1).indices.cpu().numpy()
            for i, true in enumerate(trump_b):
                if true in top3_idx[i]:
                    top3_correct += 1
    predictor.network.train()

    return top1_correct / n, top3_correct / n


# TRAIN TRUMP PREDICTOR
# Trains the TrumpPredictor MLP to infer trump card from trick history via cross-entropy.
def train_trump_predictor(cfg: PhaseConfig, load_override: str = None):
    train_episodes = cfg.supervised_data_episodes
    val_episodes = cfg.val_episodes
    batch_size = cfg.supervised_batch_size
    epochs = cfg.supervised_epochs
    eval_interval = cfg.eval_interval
    checkpoint_dir = cfg.checkpoint_dir
    seed = cfg.seed if cfg.seed is not None else 42

    print(f"Training TrumpPredictor (supervised)")
    print(f"train_episodes = {train_episodes:,}, val_episodes = {val_episodes:,}")
    print(f"epochs = {epochs}, batch_size = {batch_size}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("[Step 1] Collecting training data...")
    train_obs, train_trump = _collect_trump_predictor_data(train_episodes, seed=seed)
    print(f"Train: {len(train_obs):,} (obs, trump) pairs")

    print("[Step 2] Collecting validation data (held-out games)...")
    val_obs, val_trump = _collect_trump_predictor_data(
        val_episodes, seed=seed + train_episodes
    )
    print(f"Val: {len(val_obs):,} (obs, trump) pairs\n")

    predictor = TrumpPredictor(
        hidden_layers=cfg.hidden_layers, learning_rate=cfg.learning_rate
    )

    print(f"[Step 3] Training for {epochs} epochs...")
    best_acc = 0.0
    start_time = time.time()

    csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_loss", "val_top1", "val_top3", "elapsed_s"],
    )
    csv_writer.writeheader()
    csv_file.flush()

    dataset_size = len(train_obs)
    n_batches = (dataset_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        epoch_start = time.time()
        perm = np.random.permutation(dataset_size)
        obs_shuffled = train_obs[perm]
        trump_shuffled = train_trump[perm]

        total_loss = 0.0
        for batch_idx in range(n_batches):
            s = batch_idx * batch_size
            e = min(s + batch_size, dataset_size)
            total_loss += predictor.update(obs_shuffled[s:e], trump_shuffled[s:e])

        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            top1, top3 = _eval_trump_predictor(predictor, val_obs, val_trump)
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch + 1:>3}/{epochs}:"
                f" train_loss = {avg_loss:.4f}  val_top1 = {top1:.3f}  val_top3 = {top3:.3f}"
                f"({epoch_time:.1f}s)"
            )

            if top1 > best_acc:
                best_acc = top1
                best_path = os.path.join(checkpoint_dir, "best.pt")
                predictor.save(best_path)
                print(f"NEW BEST: {best_acc:.3f} → saved to {best_path}")

            csv_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_top1": top1,
                    "val_top3": top3,
                    "elapsed_s": elapsed,
                }
            )
            csv_file.flush()

    final_path = os.path.join(checkpoint_dir, "final.pt")
    predictor.save(final_path)
    csv_file.close()
    print(f"\nTrumpPredictor training finished. Best val top1: {best_acc:.3f}")
    print(f"Checkpoint: {final_path}")
    return predictor


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Watten RL agent")
    parser.add_argument("--phase", type=str, help="Phase name to train")
    parser.add_argument(
        "--load", type=str, default=None, help="Override checkpoint to load"
    )
    parser.add_argument(
        "--list-phases", action="store_true", help="List available phases"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["dqn", "a2c"],
        help="Override model type from config (dqn or a2c)",
    )
    args = parser.parse_args()

    if args.list_phases:
        print("Available phases:")
        for name, cfg in PHASES.items():
            print(f"  {name:35s}  mode={cfg.training_mode:<20}  blind={cfg.blind_mode}")
        exit(0)

    if not args.phase:
        parser.error("--phase is required (use --list-phases to see options)")

    if args.phase not in PHASES:
        parser.error(f"Unknown phase: {args.phase}. Use --list-phases to see options.")

    cfg = PHASES[args.phase]
    if args.model is not None:
        cfg.model_type = args.model
    train(cfg, load_override=args.load)
