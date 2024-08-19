import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

def run(episodes, planning_steps=10, exploration_bonus=0.01, is_training=True, render=False):
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
        model = defaultdict(lambda: (0, 0))  # Model for storing transitions (new_state, reward)
        visit_count = np.zeros((env.observation_space.n, env.action_space.n))  # Visit counts for exploration bonus
    else:
        with open("Lab6/q_learning_results/frozen_lake4x4_dynaq.pkl", "rb") as f:
            q = pickle.load(f)

    learning_rate_a = 0.9       # Alpha or learning state
    discount_factor_g = 0.9     # Gamma or discount factor

    epsilon = 1 if is_training else 0  # No exploration during evaluation
    epsilon_decay_rate = 0.0001     # Epsilon decay rate. 1/0.0001 = 10,000 episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]      # Initial state
        terminated = False          # True when fall in hole or reached goal
        truncated = False           # True when actions > 200

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                if is_training:
                    action = np.argmax(q[state, :] + exploration_bonus / (visit_count[state, :] + 1))  # Adding exploration bonus
                else:
                    action = np.argmax(q[state, :])  # No exploration bonus during evaluation

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                visit_count[state, action] += 1
                # Q-learning update for the real experience
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

                # Update the model with the observed transition
                model[(state, action)] = (new_state, reward)

                # Planning step: Simulate experiences from the model
                for _ in range(planning_steps):
                    sim_state, sim_action = rng.choice(list(model.keys()))  # Randomly sample from the model
                    sim_new_state, sim_reward = model[(sim_state, sim_action)]
                    q[sim_state, sim_action] += learning_rate_a * (
                        sim_reward + discount_factor_g * np.max(q[sim_new_state, :]) - q[sim_state, sim_action]
                    )

            state = new_state
    
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, 0)

            if epsilon == 0:
                learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.title("Dyna-Q+ episodes rating learning")
    plt.xlabel("Number of episodes")
    plt.ylabel("Number of rewards")
    plt.savefig('Lab6/q_learning_results/frozen_lake4x4_dynaq.png')

    if is_training:
        with open("Lab6/q_learning_results/frozen_lake4x4_dynaq.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(15000)  # Training
    # run(1, is_training=False, render=True)   # Evaluation
