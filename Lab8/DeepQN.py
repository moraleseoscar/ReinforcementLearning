import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Definir la arquitectura de la red neuronal
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity=10000, batch_size=64, gamma=0.99, lr=0.001, tau=0.01):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Red principal (Q-Network)
        self.q_net = DQN(state_dim, action_dim)
        self.q_target_net = DQN(state_dim, action_dim)
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.q_target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.q_target_net.to(self.device)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
            return q_values.argmax().item()

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample del buffer de experiencia
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calcular los valores Q para las acciones seleccionadas
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_values = self.q_target_net(next_states).max(1)[0]
            q_target = rewards + self.gamma * q_next_values * (1 - dones)

        loss = self.loss_fn(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar la red de destino (soft update)
        for target_param, param in zip(self.q_target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

episodes = 1000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
total_rewards = []
average_rewards = []

# Entrenamiento del agente
for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        total_rewards.append(episode_reward)
        agent.train()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    avg_reward = sum(total_rewards) / len(total_rewards)
    average_rewards.append(avg_reward)
    print(f"Episode {episode}, Average Reward: {avg_reward} Epsilon: {epsilon}")

env.close()

plt.plot(average_rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward Per Episode')
plt.grid(True)
plt.show()

# ------------------------------------------
# Ejecutar el entorno con el agente entrenado
# ------------------------------------------
env = gym.make('CartPole-v1', render_mode='human')  # Habilitar el renderizado

state, _ = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    env.render()  # Renderizar la simulación
    action = agent.select_action(state, epsilon=0.0)  # Acción sin exploración (exploitation puro)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    total_reward += reward
    steps += 1

print(f"Total reward after training: {total_reward}")
env.close()
