import gymnasium as gym
import numpy as np
import math
import time
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.node_value = 0.0
        self.penalized_states = set()

def uct_select(node, exploration_weight=3.0):
    log_n_visits = math.log(node.visits + 1)
    
    def uct(n):
        if n.visits == 0:
            return float('inf')
        return n.value / n.visits + exploration_weight * math.sqrt(log_n_visits / (n.visits + 1))
    
    return max(node.children.items(), key=lambda act_node: uct(act_node[1]))

def get_manhattan_distance(state, goal_state):
    state_x, state_y = state % 4, state // 4
    goal_x, goal_y = goal_state % 4, goal_state // 4
    return abs(state_x - goal_x) + abs(state_y - goal_y)

def heuristic_rollout(env, state, goal_state, rollout_depth=50, penalized_states=set()):
    total_reward = 0
    for _ in range(rollout_depth):
        available_actions = list(range(env.action_space.n))
        best_action = None
        best_distance = float('inf')

        # Favor actions that reduce the Manhattan distance to the goal
        for action in available_actions:
            env.unwrapped.s = state
            next_state, _, done, _, _ = env.step(action)
            distance = get_manhattan_distance(next_state, goal_state)
            if distance < best_distance and next_state not in penalized_states:
                best_distance = distance
                best_action = action

        env.unwrapped.s = state
        state, reward, done, _, _ = env.step(best_action)
        total_reward += reward

        if done:
            break

    return total_reward

def mcts_search(env, root_state, num_simulations=10000, rollout_depth=50, penalized_states=set()):
    root = MCTSNode(root_state)
    goal_state = 15 

    for _ in range(num_simulations):
        node = root
        state = root_state
        done = False
        path = []
        depth = 0

        # Selection and Expansion
        while not done and depth < 20:  # Limit depth to avoid too long paths
            if not node.children:
                available_actions = range(env.action_space.n)
                for action in available_actions:
                    env.unwrapped.s = state  # Restore state before action
                    new_state, _, done, _, _ = env.step(action)
                    node.children[action] = MCTSNode(new_state, parent=node)
                    if done:  # Stop expansion if we reach a terminal state
                        break
            
            exploration_weight = 3.0 / (depth + 1)  # More exploration at the beginning
            action, node = uct_select(node, exploration_weight)
            path.append((node, action))
            prev_state = state
            state, reward, done, _, _ = env.step(action)
            
            # Memorize penalized states
            if state in penalized_states or state == prev_state:
                reward -= 5.0  # Heavy penalty for revisiting penalized states or staying stuck
                node.penalized_states.add(state)
            
            depth += 1
            if done:
                break

        # Evaluation using heuristic rollout
        reward = heuristic_rollout(env, state, goal_state, rollout_depth, penalized_states)

        # Backpropagation with discounting
        discount_factor = 0.95
        for node, _ in reversed(path):
            node.visits += 1
            node.value += reward
            reward *= discount_factor

    return max(root.children.items(), key=lambda act_node: act_node[1].visits)[0]

def mcts_agent(env, num_episodes=10000, num_simulations=10000, rollout_depth=100):
    best_reward = float('-inf')
    best_actions = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        actions = []
        penalized_states = set()

        for step in range(100):  # Limit episode length
            action = mcts_search(env, state, num_simulations, rollout_depth, penalized_states)
            next_state, reward, done, _, _ = env.step(action)
            
            distance_before = get_manhattan_distance(state, 15)
            distance_after = get_manhattan_distance(next_state, 15)
            
            # Encourage moving closer to the goal, discourage moving away
            if distance_after < distance_before:
                reward += 2.0  # Stronger reward for moving closer
            elif distance_after > distance_before:
                reward -= 2.0  # Stronger penalty for moving away
            if next_state == 15:
                reward += 15.0  # Additional reward for reaching the goal
            
            total_reward += reward
            actions.append(action)
            state = next_state
            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {len(actions)}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_actions = actions

    return best_reward, best_actions

def render_best_episode(env, best_actions):
    state, _ = env.reset()
    env.render()
    time.sleep(1)

    for action in best_actions:
        state, reward, done, _, _ = env.step(action)
        env.render()
        time.sleep(0.5)
        if done:
            break

    env.close()

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    best_reward, best_actions = mcts_agent(env)
    
    print(f"\nBest episode found with reward: {best_reward}")
    print("Rendering best episode...")
    render_best_episode(env, best_actions)
