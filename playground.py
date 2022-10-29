import gym
import numpy as np

env = gym.make("point-v0")
observation = env.reset()

def __get_random(n):
	d = np.random.random(n+max(3,int(0.01*n)))
	return d[np.where(d>0)][:n]*2-1

def random_policy(observation):
    action = __get_random(2)
    return action

for _ in range(1000):
    action = random_policy(observation)  # User-defined policy function
    observation, reward, terminated, info = env.step(action)
    env.render()
    if terminated:
        observation = env.reset()
env.close()