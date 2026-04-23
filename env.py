# CS5180 Final Project
# Max Huber

# This file implements the Watten 4-player trick-taking card game as a Gymnasium environment.

# IMPORTS
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from cards import (
    DECK_SIZE,
    NUM_RANKS,
    NUM_SUITS,
    RANK_NAMES,
    SUIT_NAMES,
    card_id_to_observation,
    card_id_to_rank,
    card_id_to_string,
    card_id_to_suit,
    trick_card_strength,
)

# CONSTANTS
CARDS_PER_PLAYER = 5
NUM_PLAYERS = 4
MAX_TRICKS = 5
TRICKS_TO_WIN = 3

# Team assignments: seats 0 and 2 are team 0, seats 1 and 3 are team 1
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}

# CLASS DEFINITIONS


# WATTEN ENV
# Gymnasium environment for the Watten 4-player trick-taking card game.
class WattenEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Observation size: 12 + 32 + 4 + 8 + 4*56 + 56 = 336 bits
    OBS_SIZE = 336

    def __init__(self, render_mode: Optional[str] = None, blind_mode: bool = True):
        super().__init__()
        self.render_mode = render_mode
        self.blind_mode = blind_mode

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(WattenEnv.OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(DECK_SIZE)

        self._dealer = 0
        self._forehand = 1

        self.hands: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.trump_suit: int = -1
        self.trump_rank: int = -1
        self.dealer_shown_card: int = -1
        self.forehand_shown_card: int = -1
        self.current_player: int = -1
        self.trick_leader: int = -1
        self.current_trick: List[Tuple[int, int]] = []
        self.tricks_won: List[int] = [0] * NUM_PLAYERS
        self.team_tricks: List[int] = [0, 0]
        self.tricks_played: int = 0
        self.done: bool = False
        self.trick_history: List[List[Tuple[int, int]]] = []

    # RESET
    # Deals a new round, assigns dealer/forehand, sets trump, and returns the initial observation.
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Choose dealer randomly (but deterministically from seed)
        self._dealer = int(self.np_random.integers(NUM_PLAYERS))
        self._forehand = (self._dealer + 1) % NUM_PLAYERS

        deck = list(range(DECK_SIZE))
        self.np_random.shuffle(deck)

        self.hands = [[] for _ in range(NUM_PLAYERS)]
        for i in range(NUM_PLAYERS):
            start = i * CARDS_PER_PLAYER
            self.hands[i] = sorted(deck[start : start + CARDS_PER_PLAYER])

        self.dealer_shown_card = self.hands[self._dealer][
            self.np_random.integers(CARDS_PER_PLAYER)
        ]
        self.forehand_shown_card = self.hands[self._forehand][
            self.np_random.integers(CARDS_PER_PLAYER)
        ]
        self.trump_suit = card_id_to_suit(self.dealer_shown_card)
        self.trump_rank = card_id_to_rank(self.forehand_shown_card)

        self.current_player = self._forehand
        self.trick_leader = self._forehand
        self.current_trick = []
        self.tricks_won = [0] * NUM_PLAYERS
        self.team_tricks = [0, 0]
        self.tricks_played = 0
        self.done = False
        self.trick_history = []

        return self._get_observation(self.current_player), self._get_info()

    # STEP
    # Plays a card, resolves the trick if complete, and returns (obs, reward, terminated, truncated, info).
    def step(self, action: int):
        assert not self.done, "Round is over. Call reset()."
        assert action in self.hands[self.current_player], (
            f"Card {action} not in player {self.current_player}'s hand."
        )

        self.hands[self.current_player].remove(action)
        self.current_trick.append((self.current_player, action))
        self.current_player = (self.current_player + 1) % NUM_PLAYERS

        reward = 0.0

        if len(self.current_trick) == NUM_PLAYERS:
            winner_seat = self._resolve_trick()
            self.tricks_won[winner_seat] += 1
            winner_team = TEAMS[winner_seat]
            self.team_tricks[winner_team] += 1
            self.tricks_played += 1
            self.trick_history.append(self.current_trick)
            self.current_trick = []

            trick_reward = 0.3 if winner_team == 0 else -0.3

            if self.team_tricks[winner_team] >= TRICKS_TO_WIN:
                self.done = True
                round_reward = 1.0 if winner_team == 0 else -1.0
                reward = trick_reward + round_reward
                return (
                    self._get_observation(self.current_player),
                    reward,
                    True,
                    False,
                    self._get_info(),
                )

            reward = trick_reward
            self.current_player = winner_seat
            self.trick_leader = winner_seat
            return (
                self._get_observation(self.current_player),
                reward,
                False,
                False,
                self._get_info(),
            )

        return (
            self._get_observation(self.current_player),
            reward,
            False,
            False,
            self._get_info(),
        )

    # RESOLVE TRICK
    # Determines the winning seat of the current completed trick.
    def _resolve_trick(self) -> int:
        led_suit = card_id_to_suit(self.current_trick[0][1])
        best_seat = self.current_trick[0][0]
        best_strength = trick_card_strength(
            self.current_trick[0][1], self.trump_suit, self.trump_rank, led_suit
        )

        for seat, card in self.current_trick[1:]:
            strength = trick_card_strength(
                card, self.trump_suit, self.trump_rank, led_suit
            )
            if strength > best_strength:
                best_seat = seat
                best_strength = strength

        return best_seat

    # ENCODE TRUMP
    # Encodes trump as 12 bits; returns zeros for blind players who don't know trump.
    def encode_trump(self, player: int) -> np.ndarray:
        if self.blind_mode:
            if player in (self._dealer, self._forehand):
                suit_oh = np.zeros(NUM_SUITS, dtype=np.float32)
                suit_oh[self.trump_suit] = 1.0
                rank_oh = np.zeros(NUM_RANKS, dtype=np.float32)
                rank_oh[self.trump_rank] = 1.0
                return np.concatenate([suit_oh, rank_oh])
            else:
                return np.zeros(NUM_SUITS + NUM_RANKS, dtype=np.float32)
        else:
            suit_oh = np.zeros(NUM_SUITS, dtype=np.float32)
            suit_oh[self.trump_suit] = 1.0
            rank_oh = np.zeros(NUM_RANKS, dtype=np.float32)
            rank_oh[self.trump_rank] = 1.0
            return np.concatenate([suit_oh, rank_oh])

    # ENCODE CARD 12BIT
    # Encodes a played card as 12 bits: 4-bit suit one-hot + 8-bit rank one-hot.
    def _encode_card_12bit(self, card_id: int) -> np.ndarray:
        bits = np.zeros(NUM_SUITS + NUM_RANKS, dtype=np.float32)
        bits[card_id_to_suit(card_id)] = 1.0
        bits[NUM_SUITS + card_id_to_rank(card_id)] = 1.0
        return bits

    # ENCODE TRICK SLOT
    # Encodes a trick as 56 bits: 4 cards (12 each) + leader (4) + winner (4).
    def _encode_trick_slot(
        self, trick: List[Tuple[int, int]], winner_seat: int = -1
    ) -> np.ndarray:
        slot = np.zeros(56, dtype=np.float32)

        for i, (seat, card_id) in enumerate(trick):
            slot[i * 12 : (i + 1) * 12] = self._encode_card_12bit(card_id)

        if trick:
            leader_seat = trick[0][0]
            slot[48 + leader_seat] = 1.0

        if winner_seat >= 0:
            slot[52 + winner_seat] = 1.0

        return slot

    # GET TRICK WINNER
    # Resolves and returns the winner seat of a completed trick from trick history.
    def _get_trick_winner(self, trick: List[Tuple[int, int]]) -> int:
        led_suit = card_id_to_suit(trick[0][1])
        best_seat = trick[0][0]
        best_strength = trick_card_strength(
            trick[0][1], self.trump_suit, self.trump_rank, led_suit
        )
        for seat, card in trick[1:]:
            s = trick_card_strength(card, self.trump_suit, self.trump_rank, led_suit)
            if s > best_strength:
                best_seat = seat
                best_strength = s
        return best_seat

    # GET OBSERVATION
    # Builds the full 336-bit observation vector for the given player.
    def _get_observation(self, player: int) -> np.ndarray:
        parts = []

        # Trump encoding (12 bits)
        parts.append(self.encode_trump(player))

        # Hand (32 bits)
        hand_obs = np.zeros(DECK_SIZE, dtype=np.float32)
        for card_id in self.hands[player]:
            hand_obs[card_id] = 1.0
        parts.append(hand_obs)

        # Player seat (4 bits)
        seat_obs = np.zeros(NUM_PLAYERS, dtype=np.float32)
        seat_obs[player] = 1.0
        parts.append(seat_obs)

        # Team tricks (4 + 4 = 8 bits)
        for team in range(2):
            tricks_obs = np.zeros(4, dtype=np.float32)
            tricks_obs[self.team_tricks[team]] = 1.0
            parts.append(tricks_obs)

        # Completed tricks (4 slots × 56 bits = 224 bits)
        for i in range(4):
            if i < len(self.trick_history):
                trick = self.trick_history[i]
                winner = self._get_trick_winner(trick)
                parts.append(self._encode_trick_slot(trick, winner_seat=winner))
            else:
                parts.append(np.zeros(56, dtype=np.float32))

        # Current trick (56 bits, winner zeros)
        parts.append(self._encode_trick_slot(self.current_trick, winner_seat=-1))

        obs = np.concatenate(parts)
        assert len(obs) == self.OBS_SIZE, f"Expected {self.OBS_SIZE}, got {len(obs)}"
        return obs

    # GET INFO
    # Returns a dict of game state metadata.
    def _get_info(self) -> dict:
        return {
            "dealer": self._dealer,
            "forehand": self._forehand,
            "trump_suit": self.trump_suit,
            "trump_rank": self.trump_rank,
            "dealer_shown_card": self.dealer_shown_card,
            "forehand_shown_card": self.forehand_shown_card,
            "hands": [list(h) for h in self.hands],
            "current_player": self.current_player,
            "team_tricks": list(self.team_tricks),
            "tricks_played": self.tricks_played,
            "done": self.done,
        }

    # RENDER
    # Prints the current game state to stdout in human-readable form.
    def render(self):
        if self.render_mode != "human":
            return

        print("TRUMP INFO:")
        print(f"Dealer: Player {self._dealer}")
        print(f"Dealer showed: {card_id_to_string(self.dealer_shown_card)}")
        print(f"Forehand: Player {self._forehand}")
        print(f"Forehand showed: {card_id_to_string(self.forehand_shown_card)}")
        print(
            f"Trump suit: {SUIT_NAMES[self.trump_suit]} | Trump rank: {RANK_NAMES[self.trump_rank]}"
        )

        print("\nGAME STATE:")
        print(f"Current player: Player {self.current_player}")
        print(
            f"Team tricks: Team 0 = {self.team_tricks[0]}, Team 1 = {self.team_tricks[1]}"
        )
        print(f"Tricks played: {self.tricks_played}")

        print("\nPLAYER HANDS:")
        for i in range(NUM_PLAYERS):
            team = TEAMS[i]
            hand_str = ", ".join(card_id_to_string(c) for c in self.hands[i])
            role = ""
            if i == self._dealer:
                role = " (Dealer)"
            elif i == self._forehand:
                role = " (Forehand)"
            print(f"Player {i} [Team {team}]{role}: {hand_str}")

        if self.current_trick:
            trick_str = ", ".join(
                f"P{seat}: {card_id_to_string(c)}" for seat, c in self.current_trick
            )
            print(f"\n  Current trick: {trick_str}")

        if self.trick_history:
            print("\nTRICK HISTORY:")
            for i, trick in enumerate(self.trick_history):
                winner = self._get_trick_winner(trick)
                trick_str = ", ".join(
                    f"P{seat}: {card_id_to_string(c)}" for seat, c in trick
                )
                print(f"Trick {i + 1}: {trick_str} -> Winner: P{winner}")

        print(f"{'=' * 50}")

    def close(self):
        pass


# MAIN
if __name__ == "__main__":
    env = WattenEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    env.render()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation (first 12 bits, trump): {obs[:12]}")
    print(f"Observation (bits 12-44, hand):     {obs[12:44]}")

    print("\nPlaying round (first card from each hand)")
    while not env.done:
        p = env.current_player
        card = env.hands[p][0]
        print(f"Player {p} plays {card_id_to_string(card)}")
        obs, reward, terminated, truncated, info = env.step(card)
    print(f"\nRound over! Reward: {reward}")
    env.render()
