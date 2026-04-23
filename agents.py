# CS5180 Final Project
# Max Huber

# This file defines all agent classes (Random, Heuristic, DQN, A2C, TrumpPredictor) and observation utilities.

# IMPORTS
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cards import (
    CARD_OBS_SIZE,
    DECK_SIZE,
    NUM_RANKS,
    NUM_SUITS,
    RANK_START,
    SUIT_START,
    UNKNOWN_RANK,
    UNKNOWN_SUIT,
    card_id_to_rank,
    card_id_to_suit,
    general_card_strength,
    get_card_id,
    trick_card_strength,
)
from env import WattenEnv

# CONSTANTS

TRUMP_SIZE = NUM_SUITS + NUM_RANKS  # 12
TRUMP_START = 0
HAND_START = TRUMP_SIZE  # 12
SEAT_START = HAND_START + DECK_SIZE  # 44
TEAM0_TRICKS_START = SEAT_START + 4  # 48
TEAM1_TRICKS_START = TEAM0_TRICKS_START + 4  # 52
TRICK_SLOT_SIZE = 56  # 4 × 12-bit cards + leader (4) + winner (4)
HISTORY_START = TEAM1_TRICKS_START + 4  # 56
CURRENT_TRICK_START = HISTORY_START + 4 * TRICK_SLOT_SIZE  # 280
OBS_SIZE = CURRENT_TRICK_START + TRICK_SLOT_SIZE  # 336

# Also used by train.py for the trump predictor input slice
TEAM_TRICKS_START = TEAM0_TRICKS_START  # 48

# Team assignments: seats 0,2 = team 0; seats 1,3 = team 1
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}

# OBSERVATION PARSING


# PARSE CARD OBS 14BIT
# Parses a 14-bit card encoding (suit + unknown_suit + rank + unknown_rank); returns None if empty.
def parse_card_obs_14bit(obs_slice: np.ndarray):
    if np.sum(obs_slice) == 0:
        return None

    suit_known = obs_slice[UNKNOWN_SUIT] == 0
    rank_known = obs_slice[UNKNOWN_RANK] == 0

    suit = (
        int(np.argmax(obs_slice[SUIT_START : SUIT_START + NUM_SUITS]))
        if suit_known
        else None
    )
    rank = (
        int(np.argmax(obs_slice[RANK_START : RANK_START + NUM_RANKS]))
        if rank_known
        else None
    )
    return (suit, rank)


# PARSE CARD OBS 12BIT
# Parses a 12-bit trick-slot card encoding (suit + rank, no unknown flags); returns None if empty.
def parse_card_obs_12bit(card_obs: np.ndarray):
    if np.sum(card_obs) == 0:
        return None
    suit = int(np.argmax(card_obs[:4]))
    rank = int(np.argmax(card_obs[4:12]))
    return (suit, rank)


# PARSE HAND
# Returns the list of card_ids currently in hand from the 32-bit hand section.
def parse_hand(obs: np.ndarray) -> list:
    hand_bits = obs[HAND_START : HAND_START + DECK_SIZE]
    return [i for i in range(DECK_SIZE) if hand_bits[i] == 1.0]


# PARSE SEAT
# Returns the current player's seat index (0-3) from the 4-bit seat one-hot.
def parse_seat(obs: np.ndarray) -> int:
    return int(np.argmax(obs[SEAT_START : SEAT_START + 4]))


# PARSE TRUMP
# Returns (trump_suit, trump_rank, knows_trump); returns (None, None, False) when blind.
def parse_trump(obs: np.ndarray):
    trump_bits = obs[TRUMP_START : TRUMP_START + TRUMP_SIZE]

    suit_bits = trump_bits[:NUM_SUITS]
    rank_bits = trump_bits[NUM_SUITS:]

    if np.sum(trump_bits) == 0:
        return None, None, False

    trump_suit = int(np.argmax(suit_bits))
    trump_rank = int(np.argmax(rank_bits))
    return trump_suit, trump_rank, True


# PARSE CURRENT TRICK
# Returns (card_ids, leader_seat) for cards played so far in the current trick.
def parse_current_trick(obs: np.ndarray):
    slot = obs[CURRENT_TRICK_START : CURRENT_TRICK_START + TRICK_SLOT_SIZE]
    cards = []
    for i in range(4):
        card_obs = slot[i * 12 : (i + 1) * 12]  # 12-bit per card
        parsed = parse_card_obs_12bit(card_obs)
        if parsed is None:
            break
        suit, rank = parsed
        if suit is not None and rank is not None:
            cards.append(get_card_id(suit, rank))

    # Leader is at offset 48 (4 cards × 12 bits = 48 bits)
    leader_offset = 48
    leader_bits = slot[leader_offset : leader_offset + 4]
    leader = int(np.argmax(leader_bits)) if np.sum(leader_bits) > 0 else -1

    return cards, leader


# RANDOM AGENT
# Plays a uniformly random legal card each turn.
class RandomAgent:
    def act(self, obs: np.ndarray) -> int:
        hand = parse_hand(obs)
        return random.choice(hand)


# HEURISTIC AGENT
# Plays the strongest available card according to general card strength.
class HeuristicAgent:
    def act(self, obs: np.ndarray, epsilon: float = None) -> int:
        hand = parse_hand(obs)
        seat = parse_seat(obs)
        trump_suit, trump_rank, knows_trump = parse_trump(obs)
        trick_cards, trick_leader = parse_current_trick(obs)

        if knows_trump:
            return self._act_informed(
                hand, seat, trump_suit, trump_rank, trick_cards, trick_leader
            )
        else:
            return self._act_blind(hand, trick_cards)

    def _act_informed(
        self, hand, seat, trump_suit, trump_rank, trick_cards, trick_leader
    ):
        if not trick_cards:
            return max(
                hand, key=lambda c: general_card_strength(c, trump_suit, trump_rank)
            )

        led_suit = card_id_to_suit(trick_cards[0])

        best_idx = 0
        best_strength = trick_card_strength(
            trick_cards[0], trump_suit, trump_rank, led_suit
        )
        for i, card in enumerate(trick_cards[1:], 1):
            s = trick_card_strength(card, trump_suit, trump_rank, led_suit)
            if s > best_strength:
                best_idx = i
                best_strength = s

        winner_seat = (trick_leader + best_idx) % 4

        if TEAMS[winner_seat] == TEAMS[seat]:
            return min(
                hand, key=lambda c: general_card_strength(c, trump_suit, trump_rank)
            )

        winners = [
            c
            for c in hand
            if trick_card_strength(c, trump_suit, trump_rank, led_suit) > best_strength
        ]
        if winners:
            return min(
                winners, key=lambda c: general_card_strength(c, trump_suit, trump_rank)
            )

        return min(hand, key=lambda c: general_card_strength(c, trump_suit, trump_rank))

    def _act_blind(self, hand, trick_cards):
        if not trick_cards:
            return max(hand, key=lambda c: card_id_to_rank(c))

        led_suit = card_id_to_suit(trick_cards[0])

        best_led_rank = max(
            (card_id_to_rank(c) for c in trick_cards if card_id_to_suit(c) == led_suit),
            default=-1,
        )
        same_suit = [c for c in hand if card_id_to_suit(c) == led_suit]
        if same_suit:
            beaters = [c for c in same_suit if card_id_to_rank(c) > best_led_rank]
            if beaters:
                return min(beaters, key=lambda c: card_id_to_rank(c))

        return min(hand, key=lambda c: card_id_to_rank(c))


# Q NETWORK
# Feedforward MLP mapping observations to Q-values over all 32 card actions.
class QNetwork(nn.Module):
    def __init__(self, obs_size: int, hidden_layers: list, activation: str = "relu"):
        super().__init__()
        layers = []
        prev_size = obs_size
        for hsize in hidden_layers:
            layers.append(nn.Linear(prev_size, hsize))
            if activation == "relu":
                layers.append(nn.ReLU())
            prev_size = hsize
        layers.append(nn.Linear(prev_size, DECK_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# DQN AGENT
# Deep Monte Carlo DQN agent with target network, hand masking, and epsilon-greedy exploration.
class DQNAgent:
    def __init__(
        self,
        replay_buffer,
        hidden_layers=(256, 128),
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100_000,
        target_update_interval=1000,
        gradient_clip=10.0,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval
        self.gradient_clip = gradient_clip

        self.network = QNetwork(OBS_SIZE, list(hidden_layers)).to(self.device)
        self.target_network = QNetwork(OBS_SIZE, list(hidden_layers)).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.epsilon = epsilon_start
        self._total_steps = 0

    # Action selection

    def act(self, obs: np.ndarray, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon

        hand = parse_hand(obs)
        if not hand:
            raise ValueError("No legal actions: hand is empty")

        if np.random.random() < epsilon:
            return int(np.random.choice(hand))

        q_values = self.get_q_values(obs)
        mask = np.full(DECK_SIZE, -np.inf)
        for card in hand:
            mask[card] = 0.0
        return int(np.argmax(q_values + mask))

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.network(obs_t).squeeze(0).cpu().numpy()

    # Training

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {"rl_loss": 0.0, "total_loss": 0.0}

        batch = self.replay_buffer.sample(self.batch_size)

        obs_t = torch.from_numpy(batch["obs"]).float().to(self.device)
        action_t = torch.from_numpy(batch["action"]).long().to(self.device)
        reward_t = torch.from_numpy(batch["reward"]).float().to(self.device)
        next_obs_t = torch.from_numpy(batch["next_obs"]).float().to(self.device)
        done_t = torch.from_numpy(batch["done"]).float().to(self.device)

        # Current Q-values for chosen actions
        q_values = self.network(obs_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        # Double DQN targets
        with torch.no_grad():
            best_actions = self.network(next_obs_t).argmax(dim=1)
            next_q = (
                self.target_network(next_obs_t)
                .gather(1, best_actions.unsqueeze(1))
                .squeeze(1)
            )
            targets = reward_t + self.gamma * next_q * (1.0 - done_t)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()

        return {"rl_loss": loss.item(), "total_loss": loss.item()}

    # Epsilon decay

    def decay_epsilon(self):
        step_size = (self.epsilon_start - self.epsilon_end) / max(
            self.epsilon_decay_steps, 1
        )
        self.epsilon = max(self.epsilon_end, self.epsilon - step_size)
        self._total_steps += 1

    # Target network

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    # Clone (for teammate)

    def clone(self) -> "DQNAgent":
        c = DQNAgent(
            replay_buffer=None,
            hidden_layers=[
                l.out_features for l in self.network.net if isinstance(l, nn.Linear)
            ][:-1],
            epsilon_start=0.0,
            epsilon_end=0.0,
            device=self.device,
        )
        c.network.load_state_dict(self.network.state_dict())
        c.target_network.load_state_dict(self.target_network.state_dict())
        return c

    def sync_clone(self, clone: "DQNAgent"):
        clone.network.load_state_dict(self.network.state_dict())
        clone.target_network.load_state_dict(self.target_network.state_dict())

    # Checkpointing

    def save(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(
            {
                "network": self.network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self._total_steps,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self._total_steps = checkpoint["total_steps"]


# ACTOR CRITIC NETWORK
# Shared-trunk MLP with separate actor (softmax policy) and critic (scalar value) heads.
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_size: int, hidden_layers: list):
        super().__init__()
        assert len(hidden_layers) == 2, (
            "ActorCriticNetwork requires exactly 2 hidden layer sizes"
        )

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_layers[0]),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], DECK_SIZE),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1),
        )

    def forward(self, obs: torch.Tensor, hand_mask: torch.Tensor) -> tuple:
        features = self.shared(obs)
        actor_logits = self.actor(features)  # (batch, 32)
        state_value = self.critic(features)  # (batch, 1)

        # Apply hand mask and softmax
        masked_logits = actor_logits + hand_mask
        action_probs = torch.softmax(masked_logits, dim=-1)

        return action_probs, state_value

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.shared(obs)
        return self.actor(features)


# A2C AGENT
# Advantage Actor-Critic agent with Monte Carlo returns, BC warm-start, and critic warmup.
class A2CAgent:
    def __init__(
        self,
        hidden_layers=(128, 64),
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        gamma=0.95,
        entropy_coeff=0.01,
        critic_coeff=0.5,
        gradient_clip=0.5,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.gradient_clip = gradient_clip
        self.hidden_layers = tuple(hidden_layers)

        self.network = ActorCriticNetwork(OBS_SIZE, list(hidden_layers)).to(self.device)

        # Separate optimizers: actor gets gentler learning rate
        self.actor_optimizer = optim.Adam(
            list(self.network.actor.parameters()), lr=actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            list(self.network.shared.parameters())
            + list(self.network.critic.parameters()),
            lr=critic_learning_rate,
        )

    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple:
        hand = parse_hand(obs)
        hand_mask = torch.full((DECK_SIZE,), -1e9, device=self.device)
        for c in hand:
            hand_mask[c] = 0.0

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        probs, value = self.network(obs_t, hand_mask.unsqueeze(0))

        if greedy:
            action = int(probs.argmax(dim=-1).item())
            return action, None, None

        dist = torch.distributions.Categorical(probs.squeeze(0))
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), value.squeeze()

    def update(self, log_probs, values, returns, actor_frozen: bool = False) -> dict:
        if len(log_probs) == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
            }

        # Ensure all tensors are on correct device
        log_probs = log_probs.to(self.device)
        values = values.to(self.device)
        returns = returns.to(self.device)

        # Critic loss: predict the returns
        critic_loss = nn.functional.mse_loss(values, returns)

        if actor_frozen:
            # Only update critic — actor weights are frozen
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
            self.critic_optimizer.step()

            return {
                "actor_loss": 0.0,
                "critic_loss": critic_loss.item(),
                "entropy": 0.0,
                "total_loss": self.critic_coeff * critic_loss.item(),
            }

        # Full update: both actor and critic
        advantages = returns - values.detach()

        # Advantage normalization
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss: policy gradient weighted by advantage
        actor_loss = -(log_probs * advantages).mean()

        # Entropy bonus: encourage exploration
        entropy = -(log_probs.exp() * log_probs).mean()

        total_loss = (
            actor_loss + self.critic_coeff * critic_loss - self.entropy_coeff * entropy
        )

        # Zero both optimizers
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }

    def bc_update(self, obs_batch: np.ndarray, action_batch: np.ndarray) -> float:
        obs_t = torch.from_numpy(obs_batch).float().to(self.device)
        action_t = torch.from_numpy(action_batch).long().to(self.device)

        # Get raw logits (no mask — BC data only contains legal actions anyway)
        logits = self.network.get_logits(obs_t)

        # Cross-entropy loss — only over cards in hand
        hand_masks = torch.full_like(logits, -1e9, device=self.device)
        for i in range(obs_t.shape[0]):
            hand = parse_hand(obs_batch[i])
            for c in hand:
                hand_masks[i, c] = 0.0
        masked_logits = logits + hand_masks

        loss = nn.functional.cross_entropy(masked_logits, action_t)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.actor_optimizer.step()
        self.critic_optimizer.step()  # also trains shared layers

        return loss.item()

    def save(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(
            {
                "network": self.network.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "gamma": self.gamma,
                "entropy_coeff": self.entropy_coeff,
                "critic_coeff": self.critic_coeff,
                "gradient_clip": self.gradient_clip,
                "hidden_layers": self.hidden_layers,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])

        # Handle backward compatibility: old checkpoints had a single "optimizer" key
        if "actor_optimizer" in checkpoint and "critic_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        # else: old checkpoint or no optimizer state — keep current optimizer
        # instances as-is (already initialized in __init__). Network weights are
        # what matter (actor pretrained from BC).

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.entropy_coeff = checkpoint.get("entropy_coeff", self.entropy_coeff)
        self.critic_coeff = checkpoint.get("critic_coeff", self.critic_coeff)
        self.gradient_clip = checkpoint.get("gradient_clip", self.gradient_clip)
        self.hidden_layers = tuple(checkpoint.get("hidden_layers", self.hidden_layers))

    def clone(self) -> "A2CAgent":
        c = A2CAgent(
            hidden_layers=self.hidden_layers,
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            gamma=self.gamma,
            entropy_coeff=self.entropy_coeff,
            critic_coeff=self.critic_coeff,
            gradient_clip=self.gradient_clip,
            device=self.device,
        )
        c.network.load_state_dict(self.network.state_dict())
        return c


# TRUMP PREDICTOR

# Input size: team tricks (8) + 4 history slots (224) + current trick (56) = 288 bits
TRUMP_PREDICTOR_INPUT_SIZE = (
    HISTORY_START - TEAM_TRICKS_START
) + 5 * TRICK_SLOT_SIZE  # 288 bits


class TrumpPredictorNetwork(nn.Module):
    def __init__(
        self, input_size: int = TRUMP_PREDICTOR_INPUT_SIZE, hidden_layers: list = None
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64]

        layers = []
        prev_size = input_size
        for hsize in hidden_layers:
            layers.append(nn.Linear(prev_size, hsize))
            layers.append(nn.ReLU())
            prev_size = hsize
        layers.append(nn.Linear(prev_size, DECK_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrumpPredictor:
    def __init__(
        self, hidden_layers: list = None, learning_rate: float = 3e-4, device=None
    ):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.network = TrumpPredictorNetwork(
            input_size=TRUMP_PREDICTOR_INPUT_SIZE,
            hidden_layers=hidden_layers or [128, 64],
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def predict(self, obs: np.ndarray, return_probs: bool = False):
        # Extract trick-history input slice
        trick_history_input = obs[
            TEAM_TRICKS_START : CURRENT_TRICK_START + TRICK_SLOT_SIZE
        ]  # 288 bits

        obs_t = (
            torch.from_numpy(trick_history_input).float().unsqueeze(0).to(self.device)
        )
        logits = self.network(obs_t)  # (1, 32)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (32,)

        predicted_card = int(probs.argmax().item())
        if return_probs:
            return predicted_card, probs.cpu().detach().numpy()
        return predicted_card, None

    def predict_from_distribution(self, obs: np.ndarray) -> np.ndarray:
        trick_history_input = obs[
            TEAM_TRICKS_START : CURRENT_TRICK_START + TRICK_SLOT_SIZE
        ]
        obs_t = (
            torch.from_numpy(trick_history_input).float().unsqueeze(0).to(self.device)
        )
        logits = self.network(obs_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.cpu().detach().numpy()

    def update(self, obs_batch: np.ndarray, trump_card_batch: np.ndarray) -> float:
        # Extract trick-history slices
        obs_trick = (
            torch.from_numpy(
                obs_batch[:, TEAM_TRICKS_START : CURRENT_TRICK_START + TRICK_SLOT_SIZE]
            )
            .float()
            .to(self.device)
        )
        trump_t = torch.from_numpy(trump_card_batch).long().to(self.device)

        logits = self.network(obs_trick)  # (batch, 32)
        loss = nn.functional.cross_entropy(logits, trump_t)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# FUNCTION DEFINITIONS


# GET PREDICTED TRUMP ENCODING
# Returns (suit_onehot, rank_onehot) arrays from the predictor's top-1 prediction.
def get_predicted_trump_encoding(trump_predictor: "TrumpPredictor", obs: np.ndarray):
    pred_card, _ = trump_predictor.predict(obs, return_probs=True)
    pred_suit = pred_card // 8
    pred_rank = pred_card % 8
    suit_oh = np.zeros(4, dtype=np.float32)
    rank_oh = np.zeros(8, dtype=np.float32)
    suit_oh[pred_suit] = 1.0
    rank_oh[pred_rank] = 1.0
    return suit_oh, rank_oh


# MAIN
if __name__ == "__main__":
    from replay_buffer import ReplayBuffer

    print(f"OBS_SIZE: {OBS_SIZE}")

    # Create agent
    buf = ReplayBuffer(capacity=10000, obs_size=OBS_SIZE)
    agent = DQNAgent(replay_buffer=buf)
    print(f"Network: {agent.network}")
    print(f"Epsilon: {agent.epsilon}")

    # Test with dummy observation
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for c in [0, 5, 10, 15, 20]:
        obs[HAND_START + c] = 1.0
    obs[SEAT_START] = 1.0

    action = agent.act(obs)
    print(f"Action: {action} (should be in [0, 5, 10, 15, 20])")

    q_vals = agent.get_q_values(obs)
    print(f"Q-values shape: {q_vals.shape}")
    print(f"Q-values range: [{q_vals.min():.4f}, {q_vals.max():.4f}]")

    # Test clone
    teammate = agent.clone()
    teammate_action = teammate.act(obs)
    print(f"Clone action: {teammate_action}")

    # Verify clone Q-values match
    assert np.allclose(agent.get_q_values(obs), teammate.get_q_values(obs)), (
        "Clone Q-values should match original"
    )
    print("Clone Q-values match original.")
