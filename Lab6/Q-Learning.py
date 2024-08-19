import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("Lab6\q_learning_results\\frozen_lake4x4.pkl", "rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9       # Alpha or learning state
    discount_factor_g = 0.9     # Gamma or discount factor

    epsilon = 1
    epsilon_decay_rate = 0.0001     #Epsilon decay rate. 1/0.0001 = 10,000  episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]      #States
        terminated = False          #True when fall in hole or reached goal
        truncated = False           #True when actions > 200


        while (not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action]
                )

            state = new_state
    
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if (epsilon == 0):
            learning_rate_a = 0.0001

        if (reward == 1):
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.title("Q-learning episodes rating learning")
    plt.xlabel("Number of episodes")
    plt.ylabel("Number of rewards")
    plt.savefig('Lab6\q_learning_results\\frozen_lake4x4.png')

    if is_training:
        f = open("Lab6\q_learning_results\\frozen_lake4x4.pkl", "wb")
        pickle.dump(q,f)
        f.close()

if __name__ == '__main__':
    run(15000)      # If is training
    # run(1, is_training=False, render=True)   #If is not trainig