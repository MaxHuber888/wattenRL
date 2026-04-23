# CS5180 Final Project
# Max Huber

# This file implements a circular experience replay buffer for DQN training.

# IMPORTS
import numpy as np

# CLASS DEFINITIONS


# REPLAY BUFFER
# Circular buffer storing (obs, action, reward, next_obs, done) transitions in pre-allocated arrays.
class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int):
        assert capacity > 0, f"capacity must be positive, got {capacity}"
        self.capacity = capacity
        self.obs_size = obs_size
        self.ptr = 0
        self.size = 0

        self.obs_buffer = np.zeros((capacity, obs_size), dtype=np.float32)
        self.action_buffer = np.zeros(capacity, dtype=np.int32)
        self.reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.next_obs_buffer = np.zeros((capacity, obs_size), dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.float32)

    # ADD
    # Stores a single transition in the buffer, overwriting the oldest if full.
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # SAMPLE
    # Returns a random batch of transitions as a dict of numpy arrays.
    def sample(self, batch_size: int) -> dict:
        if self.size < batch_size:
            raise RuntimeError(
                f"Cannot sample {batch_size} transitions: only {self.size} stored"
            )

        indices = np.random.choice(self.size, size=batch_size, replace=False)

        return {
            "obs": self.obs_buffer[indices].copy(),
            "action": self.action_buffer[indices].copy(),
            "reward": self.reward_buffer[indices].copy(),
            "next_obs": self.next_obs_buffer[indices].copy(),
            "done": self.done_buffer[indices].copy(),
        }

    def __len__(self):
        return self.size

    # CLEAR
    # Resets the buffer to empty.
    def clear(self):
        self.ptr = 0
        self.size = 0
