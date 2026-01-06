import numpy as np

# Grid size
ROWS, COLS = 5, 5

# Actions: Up, Down, Left, Right
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = ['U', 'D', 'L', 'R']

# Discount factor
gamma = 0.9

# Warehouse (start) and delivery point (goal)
START = (0, 0)
GOAL = (4, 4)

# Initialize value function
V = np.zeros((ROWS, COLS))

# Initialize random policy
policy = np.random.choice(ACTIONS, size=(ROWS, COLS))

# Reward function
def reward(state):
    if state == GOAL:
        return 0
    return -1  # step cost

# Transition function
def next_state(state, action):
    x, y = state

    if action == 0 and x > 0:        # Up
        x -= 1
    elif action == 1 and x < ROWS-1: # Down
        x += 1
    elif action == 2 and y > 0:      # Left
        y -= 1
    elif action == 3 and y < COLS-1: # Right
        y += 1

    return (x, y)

# Policy Evaluation
def policy_evaluation():
    global V
    while True:
        delta = 0
        for i in range(ROWS):
            for j in range(COLS):
                state = (i, j)
                if state == GOAL:
                    continue

                v = V[i, j]
                action = policy[i, j]
                next_s = next_state(state, action)
                V[i, j] = reward(state) + gamma * V[next_s]
                delta = max(delta, abs(v - V[i, j]))

        if delta < 0.01:
            break

# Policy Improvement
def policy_improvement():
    global policy
    policy_stable = True

    for i in range(ROWS):
        for j in range(COLS):
            state = (i, j)
            if state == GOAL:
                continue

            old_action = policy[i, j]
            action_values = []

            for action in ACTIONS:
                next_s = next_state(state, action)
                action_value = reward(state) + gamma * V[next_s]
                action_values.append(action_value)

            best_action = np.argmax(action_values)
            policy[i, j] = best_action

            if old_action != best_action:
                policy_stable = False

    return policy_stable

# Policy Iteration Algorithm
while True:
    policy_evaluation()
    if policy_improvement():
        break

# Display Optimal Policy
print("\nOptimal Policy (U=Up, D=Down, L=Left, R=Right):\n")
for i in range(ROWS):
    for j in range(COLS):
        if (i, j) == GOAL:
            print(" G ", end="")
        else:
            print(" " + ACTION_NAMES[policy[i, j]] + " ", end="")
    print()
