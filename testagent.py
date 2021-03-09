import gym
import gym_bilboquet
import pygame
import os

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt


def random_agent(episodes=100):
    env = gym.make("bilboquet-v0")
    env.reset((300, 300))
    env.render()
    for e in range(episodes):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        # print(env.qobserve())
        if done:
            break


def draw_reward():
    dim = 300
    env = gym.make("bilboquet-v0", amplitude=10)
    image = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            image[i, j] = env.reward_helper(
                complex(j, i)-complex(dim/2, dim/2))

    print(image)

    implot = plt.imshow(image, cmap='hot')
    plt.show()


def lr(lr_0=0.00035, decay=1.1):
    def f(remaining):  # remaining is the proportion of epochs to do (it starts at 1 and finishes at 0 when training ends)
        return lr_0 * 1/(1+decay*(1-remaining))
    return f


def trained_agent(episodes=256,  continuous=True, load=None, save_name="test", ent_coef=0.00001, total_timesteps=25000, learning_rate=lr()):
    env = gym.make("bilboquet-v0", continuous=continuous, amplitude=10)
    env.reset((300, 300))

    if load is None:
        model = PPO('MlpPolicy', env, verbose=1, ent_coef=ent_coef, learning_rate=learning_rate,
                    tensorboard_log=f"./ppo_bilboquet_tensorboard/")
        model.learn(total_timesteps=total_timesteps, tb_log_name=save_name)
        model.save(save_name + '.zip')
        print('DONE')
        obs = env.reset()
    else:
        model = PPO.load(load)
        obs = env.reset()

    for i in range(episodes):
        action, _states = model.predict(obs, deterministic=True)
        # print(action)
        obs, reward, done, info = env.step(action)
        # print(reward)
        env.render()
        if done:
            obs = env.reset()


# discrete movement
trained_agent(continuous=False, load='discrete')

# continuous movement
trained_agent(load='continuous')
