# CS5180 Final Project
# Max Huber

# This file defines card IDs, suit/rank encoding, observation encoding, and card strength calculations.

# IMPORTS
import numpy as np

# CONSTANTS
NUM_SUITS = 4
NUM_RANKS = 8
DECK_SIZE = NUM_SUITS * NUM_RANKS
CARD_OBS_SIZE = NUM_SUITS + 1 + NUM_RANKS + 1  # 14

RANK_NAMES = ["7", "8", "9", "10", "Unter", "Ober", "King", "Ace"]
SUIT_NAMES = ["Hearts", "Bells", "Acorns", "Leaves"]

# Observation layout: indices within a 14-bit card observation
SUIT_START = 0
SUIT_END = NUM_SUITS  # 4
UNKNOWN_SUIT = NUM_SUITS  # index 4
RANK_START = NUM_SUITS + 1  # 5
RANK_END = RANK_START + NUM_RANKS  # 13
UNKNOWN_RANK = RANK_END  # index 13

# FUNCTION DEFINITIONS


# CARD ID TO SUIT
# Returns the suit index (0-3) for a given card ID.
def card_id_to_suit(card_id: int) -> int:
    return card_id // NUM_RANKS


# CARD ID TO RANK
# Returns the rank index (0-7) for a given card ID.
def card_id_to_rank(card_id: int) -> int:
    return card_id % NUM_RANKS


# GET CARD ID
# Returns the card ID for a given suit and rank index.
def get_card_id(suit: int, rank: int) -> int:
    return suit * NUM_RANKS + rank


# CARD ID TO STRING
# Returns a human-readable name string for a given card ID.
def card_id_to_string(card_id: int) -> str:
    return f"{RANK_NAMES[card_id_to_rank(card_id)]} of {SUIT_NAMES[card_id_to_suit(card_id)]}"


# CARD ID TO OBSERVATION
# Encodes a card as a 14-bit vector: suit (4) + unknown_suit (1) + rank (8) + unknown_rank (1).
def card_id_to_observation(
    card_id: int, hide_suit: bool = False, hide_rank: bool = False
) -> np.ndarray:
    obs = np.zeros(CARD_OBS_SIZE, dtype=np.float32)
    if hide_suit:
        obs[UNKNOWN_SUIT] = 1.0
    else:
        obs[SUIT_START + card_id_to_suit(card_id)] = 1.0
    if hide_rank:
        obs[UNKNOWN_RANK] = 1.0
    else:
        obs[RANK_START + card_id_to_rank(card_id)] = 1.0
    return obs


# POWER SCORE
# Returns the inherent power score of a card, independent of led suit.
def _power_score(card_id: int, trump_suit: int, trump_rank: int) -> int:
    # Tiers: Rechte=1000, Linke=500, trump-suited=200+rank, off-suit=rank
    suit = card_id_to_suit(card_id)
    rank = card_id_to_rank(card_id)
    if suit == trump_suit and rank == trump_rank:
        return 1000
    if rank == trump_rank:
        return 500
    if suit == trump_suit:
        return 200 + rank
    return rank


# TRICK SCORE
# Returns the trick-specific power score of a card given the led suit.
def _trick_score(card_id: int, trump_suit: int, trump_rank: int, led_suit: int) -> int:
    # Tiers: Rechte=1000, Linke=500, trump-suited=200+rank, led-suit=100+rank, off-suit=0
    suit = card_id_to_suit(card_id)
    rank = card_id_to_rank(card_id)
    if suit == trump_suit and rank == trump_rank:
        return 1000
    if rank == trump_rank:
        return 500
    if suit == trump_suit:
        return 200 + rank
    if suit == led_suit:
        return 100 + rank
    return 0


# GENERAL CARD STRENGTH
# Returns how many of the 32 deck cards this card strictly beats, independent of led suit.
def general_card_strength(card_id: int, trump_suit: int, trump_rank: int) -> int:
    my_score = _power_score(card_id, trump_suit, trump_rank)
    count = 0
    for other_id in range(DECK_SIZE):
        if other_id == card_id:
            continue
        if _power_score(other_id, trump_suit, trump_rank) < my_score:
            count += 1
    return count


# TRICK CARD STRENGTH
# Returns how many of the 32 deck cards this card strictly beats in a trick with the given led suit.
def trick_card_strength(
    card_id: int, trump_suit: int, trump_rank: int, led_suit: int
) -> int:
    my_score = _trick_score(card_id, trump_suit, trump_rank, led_suit)
    count = 0
    for other_id in range(DECK_SIZE):
        if other_id == card_id:
            continue
        if _trick_score(other_id, trump_suit, trump_rank, led_suit) < my_score:
            count += 1
    return count


# HAND STRENGTH
# Returns the sum of general card strengths for all cards in a hand.
def hand_strength(hand: list, trump_suit: int, trump_rank: int) -> int:
    return sum(general_card_strength(c, trump_suit, trump_rank) for c in hand)


# MAIN
if __name__ == "__main__":
    ts, tr = 0, 5
    rechte = get_card_id(ts, tr)
    print(
        f"Rechte: {card_id_to_string(rechte)}, general strength: {general_card_strength(rechte, ts, tr)}"
    )
    for s in range(NUM_SUITS):
        if s == ts:
            continue
        linke = get_card_id(s, tr)
        print(
            f"Linke:  {card_id_to_string(linke)}, general strength: {general_card_strength(linke, ts, tr)}"
        )
    trump_ace = get_card_id(ts, 7)
    print(
        f"Trump:  {card_id_to_string(trump_ace)}, general strength: {general_card_strength(trump_ace, ts, tr)}"
    )
