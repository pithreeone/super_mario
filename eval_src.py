from xml.etree import ElementTree as ET
import importlib.util
import sys
import os
import requests
import argparse
import numpy as np
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation

env = gym_super_mario_bros.make('SuperMarioBros-v0') #, apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

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
    
env = SkipFrame(env, skip=4)

env = ResizeObservation(env, shape=(84, 84))  # Resize to 84×84
env = GrayScaleObservation(env, keep_dim=False)  # Convert to grayscale and remove the color channel
env = FrameStack(env, num_stack=4)  # Stack last 4 frames
# env = JoypadSpace(env, [["right"], ["right", "A"]])
# env = JoypadSpace(env, COMPLEX_MOVEMENT)

# initializing agent
# sub_name = ""
# agent_path = sub_name + "_hw2_test.py"
agent_path = "test.py"
module_name = agent_path.replace('/', '.').replace('.py', '')
print(module_name, agent_path)
spec = importlib.util.spec_from_file_location(module_name, agent_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module  # makesure the module is set
spec.loader.exec_module(module)
Agent = getattr(module, 'Agent')

os.environ["SDL_AUDIODRIVER"] = "dummy"

# evaluating
import time
from tqdm import tqdm

total_reward = 0
total_time = 0
agent = Agent()
time_limit = 120

for episode in tqdm(range(2), desc="Evaluating"):
    obs = env.reset()
    start_time = time.time()
    episode_reward = 0
    
    i=0
    while True:
        action = agent.act(obs)

        # obs, reward, terminated, truncated, info = env.step(action)
        # done = terminated or truncated

        next_obs, reward, done, info = env.step(action)
        episode_reward += reward

        # next_obs_np = np.array(next_obs)
        # obs_np = np.array(obs)
        # print(np.linalg.norm(next_obs_np - obs_np))

        obs = next_obs

        time.sleep(0.05)

        if time.time() - start_time > time_limit:
            print(f"Time limit reached for episode {episode}")
            break

        if done:
            break

        # Render the environment
        env.render()

    end_time = time.time()
    total_reward += episode_reward
    total_time += (end_time - start_time)

env.close()

score = total_reward / 2
print(f"Final Score: {score}")