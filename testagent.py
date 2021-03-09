import gym
import gym_bilboquet
import pygame
import os

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import PPO, DDPG
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

def lr(lr_0 = 0.001, decay = 3) :
    def f(remaining) : # remaining is the proportion of epochs to do (it starts at 1 and finishes at 0 when training ends) 
        return lr_0 * 1/(1+decay*(1-remaining))
    return f

def trained_agent(episodes=100,  continuous=True, load=None, save_name="test", ent_coef=0.01, total_timesteps=10000, learning_rate=0.0005, vf_coef = 0.5):
    env = gym.make("bilboquet-v0", continuous=continuous, amplitude=10)
    env.reset((300, 300))

    if load is None:
        # discrete
        # model = PPO('MlpPolicy', env, verbose=1,
        #         learning_rate=0.001, n_steps=500)

        # continuous
        model = PPO('MlpPolicy', env, verbose=1, batch_size=512,
                    learning_rate=learning_rate, n_steps=1024, ent_coef=ent_coef,vf_coef=vf_coef, tensorboard_log=f"./ppo_bilboquet_tensorboard/",)
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

# draw_reward()
if __name__ == "__main__":
    # trained_agent(episodes=1000, continuous=True,
    #               load='test_reward_3_speed_both', ent_coef=0.001, total_timesteps = 30000, learning_rate=lr())
    trained_agent(episodes=500, continuous=True, ent_coef=0.001, vf_coef = 0.1,
                  load='test_vf_5', total_timesteps = 20000, learning_rate=lr(lr_0=0.01, decay=19))
