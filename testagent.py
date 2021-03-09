import gym
import gym_bilboquet
import pygame

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


def trained_agent(episodes=100,  continuous=True, load=None, save_name="test", ent_coef=0.01):
    env = gym.make("bilboquet-v0", continuous=continuous, amplitude=10)
    env.reset((300, 300))

    if load is None:
        # discrete
        # model = PPO('MlpPolicy', env, verbose=1,
        #         learning_rate=0.001, n_steps=500)

        # continuous
        model = PPO('MlpPolicy', env, verbose=1, batch_size=50,
                    learning_rate=0.0005, n_steps=500, ent_coef=ent_coef, tensorboard_log="./ppo_bilboquet_tensorboard/")
        model.learn(total_timesteps=10000)
        model.save(save_name)
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


draw_reward()
if __name__ == "__main__":
    trained_agent(episodes=500, continuous=True,
                  load='test0.01.zip', ent_coef=0.01)
