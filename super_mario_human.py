from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import pygame

# Initialize environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Initialize pygame
pygame.init()
pygame.display.set_mode((400, 300))
pygame.display.set_caption('mario controller')

# Set up FPS (Frames per second) control
clock = pygame.time.Clock()

done = False
obs = env.reset()

while True:
    action = 0  # Default action (NOOP)
    
    # Process pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:  # Right arrow
                action = 1  # You can experiment with other values like 2 or 3 for faster movement
                # print("Right key pressed")
                for i in range(10):
                    bs, reward, terminated, truncated, info = env.step(action)
                    env.render()
            elif event.key == pygame.K_LEFT:  # Left arrow
                action = 6
                for i in range(10):
                    bs, reward, terminated, truncated, info = env.step(action)
                    env.render()
            elif event.key == pygame.K_a:  # A (jump)
                action = 5

            elif event.key == pygame.K_b:  # B (run)
                action = 3
                for i in range(10):
                    bs, reward, terminated, truncated, info = env.step(action)
                    env.render()

    # Take action and update environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Render the environment
    env.render()

    # Control the frame rate to avoid over-slowing the game
    clock.tick(60)  # Set FPS to 60 (adjust for smoother or faster gameplay)
    # print(SIMPLE_MOVEMENT)
    if done:
        env.reset()


env.close()