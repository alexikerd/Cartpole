import keyboard
import time
import gym

done = False
final_reward = 0
env = gym.make("CartPole-v1")
observation = env.reset()
env.render()

for i in range(5):
    print(5-i)
    time.sleep(1)

print("Go!")

while not done:

    env.render()

    if keyboard.is_pressed("left arrow"):
        action = 0
    elif keyboard.is_pressed("right arrow"):
        action = 1
    else:
        action = env.action_space.sample()

    observation, reward, done, info = env.step(action)

    final_reward += 1

print(f'Your score was {final_reward}')
