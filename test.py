import gym
import torch
import torch.nn as nn
from collections import deque
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frames = deque([], maxlen=4)

        self.model = torch.jit.load('policy_model_latest.pth').to(device)
        self.model.eval()  # Set to evaluation mode
        self.prev_action = 0
        self.count = 0

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((84, 84))
        ])

    def act(self, observation):
        # select action only at no skipping frames        
        if self.count % 4 == 0:
            obs = torch.tensor(observation.copy()).permute(2, 0, 1)
            frame = self.transform(obs)
            
            self.frames.append(frame)
            while len(self.frames) < 4:
                self.frames.append(frame)

            input_tensor = torch.stack(list(self.frames), dim=0).float().unsqueeze(0) / 255.0
            
            action_values = self.model(input_tensor.to(device))
            action = torch.argmax(action_values).item()
            self.prev_action = action
        else:
            action = self.prev_action

        self.count += 1
        return action
