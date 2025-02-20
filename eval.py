from xml.etree import ElementTree as ET
import importlib.util
import sys
import os
import requests
import argparse

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

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
    
    while True:
        action = agent.act(obs) 

        obs, reward, done, info = env.step(action)
        # done = terminated or truncated
        episode_reward += reward

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

score = total_reward / 50
print(f"Final Score: {score}")