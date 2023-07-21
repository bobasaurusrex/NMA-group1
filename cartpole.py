!pip install swig
!pip install gymnasium['all']

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return np.stack(states), actions, np.stack(next_states), rewards, dones

    def __len__(self):
        return len(self.memory)

def train(model, memory, optimizer, criterion, batch_size, gamma):

    if len(memory) < batch_size:
        return

    states, actions, next_states, rewards, dones = memory.sample(batch_size)
    states = Variable(torch.FloatTensor(states))
    actions = Variable(torch.LongTensor(actions))
    next_states = Variable(torch.FloatTensor(next_states))
    rewards = Variable(torch.FloatTensor(rewards))
    dones = Variable(torch.FloatTensor(dones))

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = model(next_states).max(1)[0]

    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(q_values, target_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def exec_training(gamma=0.99, num_episodes=100):

    # Create an instance of the DQN model
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    model = DQN(input_size, output_size)

    # Create an instance of the replay memory
    capacity = 1000
    memory = ReplayMemory(capacity)

    # Set hyperparameters
    batch_size = 64
    lr = 0.001

    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    rewards_list = []
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0

        step_count = 1
        while not done:
            # Select an action using epsilon-greedy policy
            epsilon = max(0.01, 0.08 - 0.01 * episode)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            # Take the selected action and observe the next state and reward
            next_state, reward, done, terminated, truncated = env.step(action)

            # Store the transition in the replay memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state
            total_reward += reward

            # Train the model
            train(model, memory, optimizer, criterion, batch_size, gamma)
            step_count += 1

        rewards_list.append(total_reward)

        # Print the total reward for the episode
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return rewards_list


# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

re1 = exec_training(gamma=0.0, num_episodes = 100)

