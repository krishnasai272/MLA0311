import numpy as np
import random

# -------------------------------
# MULTI-DRONE ENVIRONMENT
# -------------------------------
class DroneEnvironment:
    def __init__(self, num_drones=3):
        self.num_drones = num_drones

    def reset(self):
        # Random initial positions for drones
        self.positions = np.random.rand(self.num_drones, 2) * 10
        return self.positions

    def step(self, actions):
        reward = 0

        # Move drones
        self.positions += actions

        # Collision avoidance penalty
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                distance = np.linalg.norm(self.positions[i] - self.positions[j])
                if distance < 1.0:
                    reward -= 5

        # Cooperative coverage reward
        reward += self.num_drones * 2

        done = False
        return self.positions, reward, done


# -------------------------------
# DRONE AGENT (Deterministic Policy)
# -------------------------------
class DroneAgent:
    def get_action(self, state):
        dx = random.uniform(-0.5, 0.5)
        dy = random.uniform(-0.5, 0.5)
        return np.array([dx, dy])


# -------------------------------
# MULTI-AGENT TRAINING LOOP
# -------------------------------
NUM_DRONES = 3
EPISODES = 10
STEPS = 15

env = DroneEnvironment(NUM_DRONES)
agents = [DroneAgent() for _ in range(NUM_DRONES)]

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(STEPS):
        actions = []

        for i in range(NUM_DRONES):
            action = agents[i].get_action(state[i])
            actions.append(action)

        actions = np.array(actions)
        state, reward, done = env.step(actions)
        total_reward += reward

    print("Episode", episode + 1, "| Collective Reward =", total_reward)

print("\nMulti-Agent Training Completed Successfully.")
