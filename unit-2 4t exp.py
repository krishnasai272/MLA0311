import numpy as np
import random

# ----- Warehouse Grid Environment -----
# 0 = free cell
# 1 = obstacle / wall
# -1 = danger cell (unsafe zone)
# 9 = goal / delivery destination

warehouse = np.array([
    [0,  0,  0,  0, 9],
    [0,  1,  1,  0, 0],
    [0, -1,  0,  0, 0],
    [0,  0,  0, -1, 0],
    [0,  0,  0,  0, 0]
])

ROWS, COLS = warehouse.shape

start_state = (4, 0)   # Robot starting point
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# Q-table
Q = np.zeros((ROWS, COLS, len(actions)))

# Hyperparameters
alpha = 0.7      # learning rate
gamma = 0.9      # discount factor
epsilon = 1.0    # exploration
epsilon_decay = 0.995
min_epsilon = 0.05
episodes = 500


# ----- Utility Functions -----
def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and warehouse[r][c] != 1


def step(state, action):
    r, c = state

    if action == "UP":    r -= 1
    elif action == "DOWN": r += 1
    elif action == "LEFT": c -= 1
    elif action == "RIGHT": c += 1

    # Invalid move → stay
    if not is_valid(r, c):
        return state, -5   # penalty for hitting wall

    cell = warehouse[r][c]

    if cell == 9:
        return (r, c), 20    # reward for reaching goal
    if cell == -1:
        return (r, c), -10   # danger zone penalty

    return (r, c), -1        # small step cost


# ----- Q-Learning Training -----
for ep in range(episodes):
    state = start_state

    for step_count in range(100):
        r, c = state

        # ε-greedy exploration
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, 3)
        else:
            action_index = np.argmax(Q[r, c])

        action = actions[action_index]
        next_state, reward = step(state, action)

        nr, nc = next_state

        # Q-Update Rule
        Q[r, c, action_index] += alpha * (
            reward + gamma * np.max(Q[nr, nc]) - Q[r, c, action_index]
        )

        state = next_state

        if warehouse[nr][nc] == 9:
            break

    # decay exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


# ----- Extract Optimal Path After Training -----
path = [start_state]
state = start_state

for _ in range(30):
    r, c = state
    action_index = np.argmax(Q[r, c])
    action = actions[action_index]

    next_state, _ = step(state, action)
    path.append(next_state)

    if warehouse[next_state] == 9:
            break

    state = next_state

print("\nOptimal Learned Path:")
print(path)

print("\nFinal Q-Table (values rounded):")
print(np.round(Q, 2))
