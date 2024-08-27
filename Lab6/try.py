import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.reward = 0.0
    
    def is_fully_expanded(self, env):
        return len(self.children) == env.action_space.n

    def best_child(self, c_param=np.sqrt(2.0)):
        best_value = float('-inf')
        best_child = None

        for child in self.children.values():
            if child.visits == 0:
                return random.choice(list(self.children.values()))

            else:
                exploitation_term = child.reward / child.visits
                exploration_term = c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
                child_value = exploitation_term + exploration_term

                if child_value > best_value:
                    best_value = child_value
                    best_child = child

        return best_child

def select(node, env):
    while node.is_fully_expanded(env):
        node = node.best_child()
    return node

def expand(node, env):
    if not node.is_fully_expanded(env):
        action = random.choice([a for a in range(env.action_space.n) if a not in node.children])
        next_state = env.step(action)[0]

        child_node = Node(next_state, node)
        node.children[action] = child_node
        return child_node
    return node

def simulate(env, node):
    steps = 0
    total_reward = 0
    current_state = node.state
    env.unwrapped.s = current_state  # Reset the environment to the current node state
    done = False

    while not done:
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if done and reward == 1: 
            return total_reward, steps

    return total_reward, steps

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent



def mcts(env, root, num_simulations):
    success_list = []
    successful_episodes = 0
    
    steps_to_goal = []

    rewards = []
    cumulative_reward = 0.0

    for i in range(1, num_simulations+1):
        env.reset()  # Reset the environment for each simulation
        node = select(root, env)
        node = expand(node, env)
        reward, steps = simulate(env, node)

        cumulative_reward += reward
        average_reward = cumulative_reward / i
        rewards.append(average_reward)

        if reward == 1: 
            successful_episodes += 1
            steps_to_goal.append(steps)

        success_rate = successful_episodes / i
        success_list.append(success_rate)

        backpropagate(node, reward)

    plt.plot(success_list)
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Episodes Success Rate - MCTS')
    plt.grid(True)
    plt.show()

    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Per Episode - MCTS')
    plt.grid(True)
    plt.show()

    plt.plot(steps_to_goal)
    plt.xlabel('Successful Episodes')
    plt.ylabel('Number of Steps to Reach the Goal')
    plt.title('Converge Rate - MCTS')
    plt.grid(True)
    plt.show()
    
    best_action = max(root.children, key=lambda action: root.children[action].visits)
    return best_action

env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset()

root = Node(env.unwrapped.s)

best_action = mcts(env, root, num_simulations=50000)

print("Best action selected by MCTS:", best_action)
