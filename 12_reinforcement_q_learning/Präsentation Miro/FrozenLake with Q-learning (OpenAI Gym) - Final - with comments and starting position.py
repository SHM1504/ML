# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 08:38:40 2025

@author: miroz
"""



import gym                        # Import the OpenAI Gym library (for the FrozenLake environment)
import numpy as np                # Import NumPy for numerical operations
import matplotlib.pyplot as plt   # Import matplotlib for plotting
import seaborn as sns             # Import seaborn for nicer-looking heatmaps

# --- Setup environment ---
env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")  # Create the 4x4 FrozenLake (non-slippery = deterministic)
n_states = env.observation_space.n      # Number of states (should be 16 for 4x4 grid)
n_actions = env.action_space.n          # Number of actions (should be 4: left, down, right, up)

# --- Q-learning Parameters ---
Q = np.zeros((n_states, n_actions))     # Initialize Q-table with zeros (rows = states, columns = actions)
learning_rate = 0.8                     # How quickly the agent updates its knowledge (alpha)
gamma = 0.95                            # Discount factor: how much future rewards are valued
epsilon = 1.0                           # Initial exploration rate (100% random at first)
epsilon_decay = 0.995                   # Decay rate for epsilon after each episode
epsilon_min = 0.01                      # Minimum value for epsilon (stops exploring completely)
episodes = 2000                         # Number of episodes to train for
rewards = []                            # Track rewards per episode

# --- Visualize Starting Lake Map ---
lake_map = env.desc.astype(str)  # Convert byte strings to regular strings for display

plt.figure(figsize=(4, 4))
sns.heatmap([[1]*4]*4, cbar=False, linewidths=0.5, linecolor='gray', square=True,
            annot=lake_map, fmt='s', cmap='Blues', annot_kws={"size": 16, "weight": "bold"})
plt.title("FrozenLake Map (S=start, F=frozen, H=hole, G=goal)")
plt.xticks([]); plt.yticks([])  # Hide axis ticks
plt.show()




# --- Training loop ---
for episode in range(episodes):
    state, _ = env.reset()              # Reset environment at the start of each episode
    done = False                        # Flag for whether episode has ended
    total_reward = 0                    # Total reward collected in this episode

    while not done:
        # Epsilon-greedy action selection: explore with probability epsilon, else exploit best known action
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])

        # Take the action and observe what happens
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # End the episode if either done condition is met

        # Q-learning update rule
        best_next = np.max(Q[next_state])  # Estimate of optimal future value
        Q[state, action] += learning_rate * (reward + gamma * best_next - Q[state, action])  # Update Q-value

        # Move to the next state
        state = next_state
        total_reward += reward            # Accumulate reward

    # Decay epsilon (less exploration over time)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Save total reward for plotting later
    rewards.append(total_reward)

# --- Plot: Average reward per 100 episodes ---
window = 100
rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')  # Moving average of rewards

plt.figure(figsize=(10, 4))
plt.plot(rolling_avg)
plt.title("Average Reward per 100 Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid()
plt.show()

# --- Plot: Q-table as a heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(Q, annot=True, cmap="coolwarm", fmt=".2f")  # Show Q-values for each state-action pair
plt.title("Q-table (rows = states, cols = actions)")
plt.xlabel("Actions (0=←, 1=↓, 2=→, 3=↑)")
plt.ylabel("States")
plt.show()

# --- Visualize the learned policy ---
arrows = ['←', '↓', '→', '↑']  # Arrow symbols for directions
policy = np.array([arrows[np.argmax(Q[s])] for s in range(n_states)])  # Choose best action for each state

print("\nLearned Policy (4x4 grid):\n")
for i in range(0, n_states, 4):
    print(" ".join(policy[i:i + 4]))  # Print policy as 4x4 grid


# --- Overlay learned policy on FrozenLake map ---

# Get the actual map layout from the environment
lake_map = env.unwrapped.desc.astype(str).tolist()

# Convert the 1-letter tile codes to symbols
tile_symbols = {
    'S': 'S',  # Start
    'F': '·',  # Frozen (safe)
    'H': 'H',  # Hole
    'G': 'G'   # Goal
}

print("\nVisualized Learned Policy with Lake Map:\n")
for i in range(4):
    row = ""
    for j in range(4):
        state_idx = i * 4 + j
        tile = lake_map[i][j]
        if tile == 'H' or tile == 'G':
            row += f" {tile_symbols[tile]} "  # Show holes/goals directly
        else:
            action = np.argmax(Q[state_idx])
            row += f" {arrows[action]} "      # Show arrow on walkable tiles
    print(row)

