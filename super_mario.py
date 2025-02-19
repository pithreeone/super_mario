import cv2
import gym
import pygame
import random
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import deque, namedtuple
from tqdm import tqdm

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        channels_dim = [input_dim, 32, 64, 128]

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_dim[i], channels_dim[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(channels_dim[i+1]),
            )
            for i in range(len(channels_dim)-1)
        ])

        self.flatten = nn.Flatten()
        
        # (240, 256, 3) -> 
        self.linear = nn.Sequential(
            nn.Linear(128*30*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
        )

    def forward(self, obs):
        out = obs
        for conv in self.conv:
            out = conv(out)

        out = self.flatten(out)
        out = self.linear(out)
        return out


class Agent():
    def __init__(self):
        self.action_space = env.action_space
        self.replay_memory = deque([], maxlen = 1000)
        self.policy_model = DQN(input_dim = 3, output_dim = env.action_space.n)
        self.target_model = DQN(input_dim = 3, output_dim = env.action_space.n)
        self.policy_model.to(device)
        self.target_model.to(device)

        self.training = True
        self.gamma = 0.99
        self.batch_size = 8
        self.TAU = 0.01
        self.optimzer = optim.AdamW(self.policy_model.parameters(), lr = 1e-4, amsgrad = True)

    def act(self, obs, epsilon):
        if np.random.random() < epsilon and self.training:
            return self.action_space.sample()
        else:
            action_prob = self.policy_model(obs.to(device))
            return torch.argmax(action_prob).item()
    
    def learn(self):
        random_samples = self.sample() # list of Transition
        batch = Transition(*zip(*random_samples)) # Transition of list
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        state_action_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            next_state_value = self.target_model(next_state_batch).max(dim=1).values

        expected_state_action_values = reward_batch + self.gamma * next_state_value

        criterion = nn.SmoothL1Loss()
        
        loss = criterion(state_action_values, expected_state_action_values)
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()

        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_model.load_state_dict(target_net_state_dict)

        return loss.item()

    def sample(self):
        return random.sample(self.replay_memory, self.batch_size)


def train(agent, env, episodes):
    for episode in range(1, episodes + 1):
        ret = 0
        done = False
        obs, _ = env.reset()
        obs = torch.tensor(obs.transpose(2, 0, 1).copy()).to(torch.float).unsqueeze(0) / 255

        count = 0
        learn_count = 0
        while not done:
            action = agent.act(obs, epsilon=0.1)
            next_obs, reward, terminanted, truncated, _ = env.step(action)
            done = terminanted or truncated
            ret += reward

            # print(f"Action: {action}")

            action = torch.tensor(action).unsqueeze(0)
            reward = torch.tensor(reward).unsqueeze(0)
            next_obs = torch.tensor(next_obs.transpose(2, 0, 1).copy()).to(torch.float).unsqueeze(0) / 255

            # if count % 100 == 0:       
            agent.replay_memory.append(Transition(obs, action, reward, next_obs))
            obs = next_obs
            
            if len(agent.replay_memory) < agent.batch_size:
                continue
            else:
                # optimize the model
                loss = agent.learn()
                learn_count += 1
                if learn_count % 100 == 0:
                    print(loss)

        print(f"Return: {ret}")

if __name__ == '__main__':
    # Initialize environment
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # Initialize pygame
    pygame.init()
    obs, _ = env.reset()

    obs_rgb = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    cv2.imshow("Mario Observation", obs_rgb)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    agent = Agent()
    print("Observation shape:", obs.shape, type(obs))  # (240, 256, 3)  # <'numpy.ndarray'>
    print("Available actions and their corresponding indices:")
    for i, action in enumerate(COMPLEX_MOVEMENT):
        print(f"{i}: {action}")
    '''
    0: ['NOOP']
    1: ['right']
    2: ['right', 'A']
    3: ['right', 'B']
    4: ['right', 'A', 'B']
    5: ['A']
    6: ['left']
    7: ['left', 'A']
    8: ['left', 'B']
    9: ['left', 'A', 'B']
    10: ['down']
    11: ['up']
    '''

    episodes = 1000
    train(agent, env, episodes)

    env.close()