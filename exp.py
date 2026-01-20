import numpy as np
import random

# -------------------------------
# RTS GAME ENVIRONMENT
# -------------------------------
class RTSEnvironment:
    def reset(self):
        # state = [resources, units, enemy_strength]
        self.state = np.array([50.0, 5.0, 40.0])
        return self.state

    def step(self, action):
        gather, build, attack = action

        # Apply actions
        self.state[0] += gather * 5      # gather resources
        self.state[1] += build * 2       # build units
        self.state[2] -= attack * self.state[1]  # attack enemy

        # Reward function
        reward = gather + build + (attack * 5)

        # Check if enemy defeated
        done = self.state[2] <= 0

        return self.state, reward, done


# -------------------------------
# ACTOR (Deterministic Policy)
# -------------------------------
class Actor:
    def get_action(self, state):
        gather = min(1.0, state[0] / 100)
        build  = min(1.0, state[1] / 20)
        attack = min(1.0, state[1] / max(state[2], 1))
        return np.array([gather, build, attack])


# -------------------------------
# REPLAY BUFFER
# -------------------------------
class ReplayBuffer:
    def __init__(self, max_size=500):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)


# -------------------------------
# DDPG TRAINING LOOP
# -------------------------------
env = RTSEnvironment()
actor = Actor()
memory = ReplayBuffer()

EPISODES = 10
STEPS = 20

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(STEPS):
        action = actor.get_action(state)
        next_state, reward, done = env.step(action)

        memory.add((state, action, reward, next_state))
        state = next_state
        total_reward += reward

        if done:
            break

    print("Episode", episode + 1, "| Total Reward =", round(total_reward, 2))

print("\nDDPG training completed successfully.")
_
