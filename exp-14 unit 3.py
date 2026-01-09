import numpy as np
import random

# -----------------------------
# RTS Environment
# -----------------------------
class RTSEnvironment:
    def reset(self):
        self.state = np.array([50.0, 5.0, 40.0])  # resources, units, enemy
        return self.state

    def step(self, action):
        gather, build, attack = action

        self.state[0] += gather * 4
        self.state[1] += build * 2
        self.state[2] -= attack * self.state[1]

        reward = gather + build + attack * 5
        done = self.state[2] <= 0

        return self.state, reward, done


# -----------------------------
# Actor Network (Deterministic Policy)
# -----------------------------
class Actor:
    def predict(self, state):
        gather = min(1.0, state[0] / 100)
        build = min(1.0, state[1] / 20)
        attack = min(1.0, state[1] / max(state[2], 1))
        return np.array([gather, build, attack])


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, size=500):
        self.buffer = []
        self.size = size

    def add(self, data):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(data)


# -----------------------------
# Training Loop
# -----------------------------
env = RTSEnvironment()
actor = Actor()
memory = ReplayBuffer()

EPISODES = 10

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(20):
        action = actor.predict(state)
        next_state, reward, done = env.step(action)

        memory.add((state, action, reward, next_state))
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {ep+1} : Total Reward = {total_reward:.2f}")

print("\nTraining completed successfully.")
