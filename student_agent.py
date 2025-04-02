import gym
import torch
import torch.nn as nn
from collections import deque
import numpy as np
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frames = deque([], maxlen=4)
        self.frames0 = deque([], maxlen=4)
        self.frames1 = deque([], maxlen=4)
        self.frames2 = deque([], maxlen=4)
        self.frames3 = deque([], maxlen=4)
        self.frames_list = [self.frames0, self.frames1, self.frames2, self.frames3]

        self.model = torch.jit.load('policy_model_latest.pth').to(device)
        self.model.eval()  # Set to evaluation mode
        self.prev_action = 0
        self.count = 0

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((84, 84))
        ])

    def act(self, observation):
        mod = self.count % 4
        # frames = self.frames_list[mod]
        # obs = torch.tensor(observation.copy()).permute(2, 0, 1)
        # frame = self.transform(obs)
        # frame = np.array(frame)

        # frame = np.expand_dims(observation, 0)

        # self.frames_list[mod].append(frame)
        # while len(self.frames_list[mod]) < 4:
        #     self.frames_list[mod].append(frame) 

        # input = np.concatenate(list(self.frames_list[mod]), axis=0)
        # input = torch.tensor(input, dtype=torch.float32).unsqueeze(0) / 255.0
        # action_values = self.model(input.to(device))
        # action = torch.argmax(action_values).item()           
        # print(torch.tensor(np.array(observation), dtype=torch.float32).shape)
        # input = torch.tensor(np.array(observation), dtype=torch.float32).to("cuda").unsqueeze(0)
        # action_values = self.model(input)
        # action = torch.argmax(action_values).item()   

        if self.count % 4 == 0:
            obs = torch.tensor(observation.copy()).permute(2, 0, 1)
            frame = self.transform(obs)
            frame = np.array(frame)

            self.frames.append(frame)
            while len(self.frames) < 4:
                self.frames.append(frame)

            input = np.concatenate(list(self.frames), axis=0)
            input = torch.tensor(input, dtype=torch.float32).unsqueeze(0) / 255.0
        
            action_values = self.model(input.to(device))
            action = torch.argmax(action_values).item()
            self.prev_action = action
        else:
            action = self.prev_action

        self.count += 1

        return action
        # return self.action_space.sample()