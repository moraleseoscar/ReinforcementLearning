import gymnasium as gym
import numpy as np
from collections import defaultdict
import math
import random
import matplotlib.pyplot as plt 

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == env.action_space.n

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / (child.visits + 1e-4)) + c_param * math.sqrt((2 * math.log(self.visits + 1)) / (child.visits + 1e-4))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self):
        action = random.choice([a for a in range(env.action_space.n) if a not in self.children])
        next_state, reward, terminated, truncated, _ = env.step(action)
        child_node = MCTSNode(state=next_state, parent=self)
        self.children[action] = child_node
        return child_node, reward, terminated, truncated

class MCTS:
    def __init__(self, env, iterations=1000):
        self.env = env
        self.iterations = iterations

    def run(self, initial_state):
        root = MCTSNode(state=initial_state)
        for _ in range(self.iterations):
            node = root
            env.reset()

            # Selección y Expansión
            while not node.is_fully_expanded():
                node, reward, terminated, truncated = node.expand()
                if terminated or truncated:
                    break

            # Simulación
            reward_sum = self.rollout(node.state)

            # Backpropagation
            self.backpropagate(node, reward_sum)

        return root.best_child(c_param=0.0).state

    def rollout(self, state):
        total_reward = 0
        observation = state  # Usar el estado actual como punto de partida

        while True:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

env = gym.make('FrozenLake-v1', render_mode="human")
mcts = MCTS(env=env, iterations=1000)
initial_state, info = env.reset()
best_action_sequence = []
rewards_per_episode = []

for episode in range(100):
    best_state = mcts.run(initial_state)
    best_action_sequence.append(best_state)
    observation, reward, terminated, truncated, info = env.step(best_state)

    rewards_per_episode.append(reward)  # Guarda la recompensa por episodio

    if terminated or truncated:
        print("Goal reached!" if reward > 0 else "Fell into a hole!")
        break

env.close()