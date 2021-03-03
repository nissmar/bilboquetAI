import gym
import gym_bilboquet
import pygame


def random_agent(episodes=100):
    env = gym.make("bilboquet-v0")
    env.reset((300, 300))
    env.render()
    for e in range(episodes):
        action = env.action_space.sample()
        # print(env.observe())
        state, reward, done = env.step(action)
        env.render()
        # print(env.qobserve())
        if done:
            break


if __name__ == "__main__":
    random_agent()
