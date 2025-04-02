from xml.etree import ElementTree as ET
import importlib.util
import sys
import os
import requests
import argparse
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation
import gym

# evaluating
import time
from tqdm import tqdm

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
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default="", type=str)
    args = parser.parse_args()
    return args

def reward_shaping(state, reward):
    if reward == -15:
        return -100
    elif reward == 4:
        return 200
    else:
        return reward

def run_agent(episode_num=50, time_limit=180, render=False):
    # initializing agent
    agent_path = "student_agent.py"
    module_name = agent_path.replace('.py', '')

    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # makesure the module is set
    spec.loader.exec_module(module)
    Agent = getattr(module, 'Agent')

    os.environ["SDL_AUDIODRIVER"] = "dummy"

    total_reward = 0
    total_time = 0
    # agent = Agent()

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT) # 
    # env = JoypadSpace(env, [["right"], ["right", "A"]])
    # env = SkipFrame(env, skip=4)
    # env = ResizeObservation(env, shape=(84, 84))
    # env = GrayScaleObservation(env, keep_dim=False)
    # env = FrameStack(env, num_stack=4)

    for episode in tqdm(range(episode_num), desc="Evaluating"):
        obs = env.reset()
        start_time = time.time()
        episode_reward = 0
        agent = Agent()
        idx = 0
        while True:
            
            action = agent.act(obs)

            next_obs, reward, done, _ = env.step(action)
            reward = reward_shaping(obs, reward)
            episode_reward += reward
            idx += 1
            print(reward, idx)

            obs = next_obs

            if time.time() - start_time > time_limit:
                print(f"Time limit reached for episode {episode}")
                break

            if done:
                break

            # Render the environment
            if render:
                env.render()
                time.sleep(0.01)

        end_time = time.time()
        total_reward += episode_reward
        total_time += (end_time - start_time)

        print(total_reward)

    env.close()

    score = total_reward / episode_num
    return score

def eval_score():
    args = parse_arguments()
    
    # retrive submission meta info from the XML file
    xml_file_path = 'meta.xml'

    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Find the 'info' element and extract the 'name' value
    for book in root.findall('info'):
        team_name =  book.find('name').text

    ### Start of evaluation section
    agent_avg_score = run_agent(episode_num=2,
                                time_limit=180,
                                render=True)

    print(f"Final Score: {agent_avg_score}")

    ### End of evaluation section

    # push to leaderboard
    params = {
        'act': 'add',
        'name': team_name,
        'score': str(agent_avg_score),
        'token': args.token
    }
    url = 'http://140.114.89.61/drl_hw3/action.php'

    response = requests.get(url, params=params)
    if response.ok:
        print('Success:', response.text)
    else:
        print('Error:', response.status_code)

if __name__ == '__main__':
    eval_score()