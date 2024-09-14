# deeplizard example 1 : Frozen Lake

# İmport Libraries 

import numpy as np \
import gym \
import random \
import time \
from IPython.display import clear_output 

# Initialize the environment
env = gym.make("FrozenLake-v1")

# Define action and state space sizes
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Initialize the Q-table
q_table = np.zeros((state_space_size, action_space_size))

# Hyperparameters
num_episodes = 1000 \
max_steps_per_episode = 100 \
learning_rate = 0.1 \
discount_rate = 0.99 

# Exploration parameters
exploration_rate = 1 \
max_exploration_rate = 1 \
min_exploration_rate = 0.01 \
exploration_decay_rate = 0.01 

# List to hold rewards for each episode
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes): \
    state = env.reset()[0]  # env.reset() now returns 2 values, state and info 

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Epsilon-greedy action selection
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        # Take action and get new state, reward, done, truncated, and info
        new_state, reward, done, truncated, info = env.step(action)

        # Update Q(s, a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        # End episode if done or truncated
        if done or truncated:
            break

    # Decay exploration rate
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000) \
count = 1000 \
print("******Average reward per thousand episodes***********\n") \
for r in rewards_per_thousand_episodes: \
    print(count, ": ", str(sum(r) / 1000)) \
    count += 1000 

# Print updated Q-table
print("\n\n******* Q-table *********\n") \
print(q_table) 
