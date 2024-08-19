import numpy as np
import random
import math
import matplotlib.pyplot as plt
import gymnasium as gym

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.reset()
        
    def reset(self):
        self.agent_pos = 0
        self.terminated = False
        return self.agent_pos
    
    def step(self, action):
        if self.terminated:
            raise RuntimeError("Step called after termination")
        
        row, col = divmod(self.agent_pos, self.size)
        
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
        
        self.agent_pos = row * self.size + col
        
        # Check for termination
        if self.agent_pos == self.size * self.size - 1:  # Goal state
            reward = 1
            self.terminated = True
        else:
            reward = -0.1  # Penalty for each step to encourage faster solutions
        
        return self.agent_pos, reward, self.terminated
    
    def action_space(self):
        return 4  # Number of actions: Up, Right, Down, Left

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == 4  # Number of possible actions

    def best_child(self, c_param=1.4):
        if self.visits == 0:
            return random.choice(list(self.children.values()))
        choices_weights = [
            (child.value / (child.visits + 1e-4)) + c_param * math.sqrt((2 * math.log(self.visits + 1)) / (child.visits + 1e-4))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self, env):
        actions = [a for a in range(4) if a not in self.children]
        if not actions:
            return self, 0, True  # No actions left to expand

        action = random.choice(actions)
        next_state, reward, terminated = env.step(action)
        child_node = MCTSNode(state=next_state, parent=self)
        self.children[action] = child_node
        return child_node, reward, terminated

class MCTS:
    def __init__(self, env, iterations=1000):
        self.env = env
        self.iterations = iterations
        self.rewards_per_episode = []

    def run(self, initial_state):
        root = MCTSNode(state=initial_state)
        for _ in range(self.iterations):
            node = root
            self.env.reset()

            # Selection and Expansion
            while not node.is_fully_expanded():
                node, reward, terminated = node.expand(self.env)
                if terminated:
                    break

            # Simulation
            reward_sum = self.rollout(node.state)
            self.rewards_per_episode.append(reward_sum)

            # Backpropagation
            self.backpropagate(node, reward_sum)

        # Return the best action from the root node
        best_action = root.best_child(c_param=0.0)
        return best_action

    def rollout(self, state):
        total_reward = 0
        self.env.reset()
        self.env.agent_pos = state

        while True:
            action = random.choice(range(4))  # Random action in grid world
            next_state, reward, terminated = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

# Main script
env = GridWorld(size=4)
mcts = MCTS(env=env, iterations=1000)
initial_state = env.reset()

# Running MCTS in the simulated grid world
best_action_node = mcts.run(initial_state)
best_action = list(best_action_node.children.keys())[0]  # Get the best action

# Plotting rewards per episode after simulation
plt.plot(mcts.rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode in GridWorld Simulation')
plt.show()

frozenlake_env = gym.make('FrozenLake-v1', render_mode="human")
state, info = frozenlake_env.reset()
done = False

while not done:
    action = best_action  # Use the best action from MCTS
    print(action)
    state, reward, done, truncated, info = frozenlake_env.step(action)
    frozenlake_env.render()

frozenlake_env.close()