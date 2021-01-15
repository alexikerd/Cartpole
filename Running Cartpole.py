import gym
from matplotlib import pyplot as plt
import math
import keyboard
import numpy as np

env = gym.make("CartPole-v1")
env.reset()




# cartpole tunnel syndrome

done = False
final_reward = 0
action = 0
rewards = []

def calc_exit_time(observation):
    
    return (np.sign(observation[1])*2.4 - observation[0])/observation[1]

def calc_fall_time(observation):
    
    return (np.sign(observation[3])*2.4 - observation[2])/observation[3]


for _ in range(10):

    while not done:
        
        env.render()
        observation, reward, done, info = env.step(action)


        # action = env.action_space.sample()


        # action = 0 if observation[2]<0 else 1

        
#         if keyboard.is_pressed("left arrow"):
#             action = 0
#         elif keyboard.is_pressed("right arrow"):
#             action = 1
#         else:
#             action = env.action_space.sample()

        if calc_exit_time(observation)<calc_fall_time(observation):

            action = 0 if np.sign(observation[1])>0 else 1

        else:

            action = 1 if np.sign(observation[3])>0 else 0



        final_reward += reward

    env.reset()
    done = False
    rewards.append(final_reward)
    final_reward = 0


print(np.max(rewards))