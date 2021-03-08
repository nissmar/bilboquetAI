import gym
import gym_bilboquet
import pygame

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import PPO


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


def trained_agent(episodes=100, load=None):
    env = gym.make("bilboquet-v0", continuous=True, amplitude=10)
    env.reset((300, 300))

    if load is None:
        model = PPO('MlpPolicy', env, verbose=1,
                    learning_rate=0.0005, n_steps=500)
        model.learn(total_timesteps=10000)
        model.save("test")
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


if __name__ == "__main__":
    trained_agent(500, 'continuous')
