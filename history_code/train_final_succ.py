import cv2
import pygame
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym.wrappers.frame_stack import LazyFrames
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from nes_py.wrappers import JoypadSpace
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# define the replay memory save type
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0') # observation shape: (240, 256, 3)
    # env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=4)
    
    return env

class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DuelingDQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQNSolver, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Shared fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        # Separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(512, 1)  # Outputs V(s)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, n_actions)  # Outputs A(s, a)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        features = self.conv(x).view(x.size(0), -1)  # Extract features
        fc_out = self.fc(features)

        value = self.value_stream(fc_out)  # V(s)
        advantage = self.advantage_stream(fc_out)  # A(s, a)

        # Combine streams using mean subtraction
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class Agent():
    def __init__(self, input_shape):
        self.action_space = env.action_space
        self.replay_memory = deque([], maxlen = 50000)

        # model
        self.policy_model = DuelingDQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        self.target_model = DuelingDQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        # self.policy_model = torch.jit.load("policy_model_latest.pth")
        # self.target_model = torch.jit.load("policy_model_latest.pth")
        for param in self.target_model.parameters():
            param.requires_grad = False
        # self.target_model.load_state_dict(self.policy_model.state_dict())

        # training
        self.training = True
        self.gamma = 0.9
        self.batch_size = 32
        self.step_count = 0
        self.network_sync_rate = 10000
        self.optimzer = optim.Adam(self.policy_model.parameters(), lr = 0.00025) #lr=0.00025
        self.criterion = nn.SmoothL1Loss()

        # exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.exploration_fraction = 0.2
        self.total_timestep = 10000000

        # information buffer
        self.losses = []
        self.rewards = []
        self.qvalues = []
        self.reward_mean = []
        self.qvalues_mean = []
        self.returns = []

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        # define the percentatge of exploration
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def act(self, state, epsilon):
        state = state.to(device)
        qvalues = self.policy_model(state)
        action = torch.argmax(qvalues).item()

        # only for plot
        self.qvalues.append(torch.max(qvalues).item())

        # epsilon-greedy
        if np.random.rand() < epsilon and self.training:
            return self.action_space.sample()
        else:
            return action

    def cache(self, state, next_state, action, reward, done):
        # save experience into replay memory
        action = torch.tensor(action).unsqueeze(0)
        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)

        self.replay_memory.append(Transition(state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.replay_memory, self.batch_size)
    
    def recall(self):
        random_samples = self.sample() # list of Transition
        batch = Transition(*zip(*random_samples)) # Transition of list

        # Normalize states
        state_batch = (torch.tensor(np.array(batch.state), dtype=torch.float) / 255.0).to(device)
        
        # Normalize next_states
        next_state_batch = (torch.tensor(np.array(batch.next_state), dtype=torch.float) / 255.0).to(device)
        
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch
    def learn(self):
        
        # sample the experience from replay memory
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.recall()

        # TD estimate
        state_action_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))

        # TD target
        with torch.no_grad():
            next_state_value, _ = self.target_model(next_state_batch).max(dim=1)

        # expected_state_action_values = reward_batch + (1-done_batch.float()) * self.gamma * next_state_value
        expected_state_action_values = reward_batch + self.gamma * next_state_value

        # loss function
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # back-propagation
        self.optimzer.zero_grad()
        loss.backward()

        # clamp gradient
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        # update weight
        self.optimzer.step()

        self.step_count += 1
        
        if self.step_count >= self.network_sync_rate:
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.step_count=0

        return loss.item()

    def clearbuffer(self):
        self.losses = []
        self.rewards = []
        self.qvalues = []
    
    def save_model(self, episode=0, file_name="policy_model_best.pth"):
        scripted_model = torch.jit.script(self.policy_model)
        torch.jit.save(scripted_model, file_name)


def train(env, episodes):
    log_data = []
    max_ret = 0
    obs = env.reset()

    agent = Agent(obs.shape)

    # The global timestep and progress_bar is for epsilon scheduling and progress visualization
    global_timestep = 0
    progress_bar = tqdm(total=agent.total_timestep, desc="Training Progress")

    for episode in range(1, episodes + 1):

        ret = 0
        learn_count = 0
        done = False
        
        obs_input = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0) / 255

        while True:
            # get epsilon from epsilon-scheduler, depends on the curent global-timestep
            epsilon = agent.linear_schedule(agent.epsilon_start, agent.epsilon_end, agent.exploration_fraction * agent.total_timestep, global_timestep)
            
            # choose action by epsilon-greedy
            action = agent.act(obs_input, epsilon)
            
            # apply action to environment and get r and s'
            next_obs, reward, done, info = env.step(action)

            next_obs_input = torch.tensor(np.array(next_obs), dtype=torch.float32).unsqueeze(0) / 255

            # update return
            ret += reward

            # save experience into buffer
            agent.cache(obs, next_obs, action, reward, done)
            
            obs_input = next_obs_input
            obs = next_obs
            
            # env.render()
            
            if len(agent.replay_memory) < agent.batch_size:
                continue

            # optimize the model
            loss = agent.learn()
            
            learn_count += 1
            global_timestep += 1

            # for plot
            agent.losses.append(loss)
            agent.rewards.append(reward)

            # log info
            # if learn_count % 1000 == 0:
            #     tqdm.write(f"Episode {episode}, Step {learn_count}, Loss: {loss:.4f}, Epsilon: {epsilon}")

            # Update tqdm bar manually
            progress_bar.update(1)

            # Check if end of game
            if done or info["flag_get"]:
                break

        agent.reward_mean.append(np.mean(agent.rewards))
        agent.qvalues_mean.append(np.mean(agent.qvalues))
        agent.returns.append(ret)

        tqdm.write(f"Episode {episode} Return: {ret}, Epsilon: {epsilon}")
        log_data.append({"episode": episode, "return": ret})
        
        # save the model with higtest return
        if ret > max_ret:
            agent.save_model(episode, file_name='policy_model_best.pth')
            max_ret = ret

        # Save model every 20 episodes
        if episode % 20 == 0:
            agent.save_model(episode, file_name='policy_model_latest.pth')
            tqdm.write("[INFO]: Save model!")

            with open("training_log.json", "w") as log_file:
                json.dump(log_data, log_file, indent=4)
            tqdm.write(f"Training log saved at episode {episode}")
            save_plot(episode, agent.losses, agent.reward_mean, agent.qvalues_mean, agent.returns)
        
        agent.clearbuffer()
        obs = env.reset()

    progress_bar.close()

def moving_average(values, n):
    offset = (n - 1) // 2
    v = [values[0]] * offset + values + [values[-1]] * offset
    return [sum(v[i - offset : i + offset + 1]) / n for i in range(offset, len(v) - offset)]

def save_plot(episode, losses, rewards, qvalues, returns):
    fig, axis = plt.subplots(2, 3, figsize=(16, 5))
    axis = axis.flatten()

    # plot loss curve
    axis[0].plot(range(len(losses)), losses)
    axis[0].set_ylabel('Loss per optimization')
    # plot average reward per epsiode
    axis[1].plot(range(len(rewards)), rewards)
    axis[1].set_ylabel('Average of the reward per episode')
    # plot average max Q-value per epsiode
    axis[2].plot(range(len(qvalues)), qvalues)
    axis[2].set_ylabel('Average of the max predicted Q value')
    # plot return per epsiode
    axis[3].plot(range(len(returns)), returns)
    axis[3].set_ylabel('Return per episode')
    # plot the moving average of return 
    returns_movavg = moving_average(returns, 60)
    axis[3].plot(range(len(returns_movavg)), returns_movavg, color='red')

    fig.suptitle(f"Episode {episode}")
    fig.tight_layout()
    plt.savefig(f"plot/training6/episode-{episode}.png")
    tqdm.write(f"Figure \"episode-{episode}.png\" saved.")
    for axis in axis:
        axis.cla()
    plt.close(fig)

if __name__ == '__main__':
    # Initialize environment
    env = make_env()
    # Initialize pygame
    pygame.init()

    episodes = 20000
    train(env, episodes)

    env.close()

'''
Full Action Space:
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