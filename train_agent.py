import gym
import gym_bilboquet
from math import ceil

from agents.agent_v1 import new_agent, notify_env_answer, save_agent, set_training_params

NUMBER_OF_EPISODES = 100
MAX_STEPS_BY_EPISODE = 10000
RENDER = False
RESET_AGENT = True
INITIAL_POS = (500,300)

SPATIAL_RESOLUTION_DIVIDED_BY_CUP_SIZE = 0.2
SPEED_RESOLUTION_DIVIDED_BY_CUP_SIZE = 0.5 
AVERAGE_SPEED_DIVIDED_BY_CUP_SIZE = 4
accepted_rectangle = [(100,500),(100,500)]
tolerance_threshold = 100
reward_outside_of_accepted_zone = -1

epsilon = 0.1
alpha = 0.2
gamma = 0.9


def get_zone_code(cup_pos, accepted_rectangle):
    if cup_pos[0] < accepted_rectangle[0][0]:
        return ('LEFT',abs(cup_pos[0]-accepted_rectangle[0][0]))
    if cup_pos[0] > accepted_rectangle[0][1]:
        return ('RIGHT',abs(cup_pos[0]-accepted_rectangle[0][1]))
    if cup_pos[1] < accepted_rectangle[1][0]:
        return ('DOWN',abs(cup_pos[1]-accepted_rectangle[1][0]))
    if cup_pos[1] > accepted_rectangle[1][1]:
        return ('UP',abs(cup_pos[1]-accepted_rectangle[1][1]))
    return ('OK',None)

env = gym.make("bilboquet-v0")
timestep = env.timestep
spatial_resolution = SPATIAL_RESOLUTION_DIVIDED_BY_CUP_SIZE * env.cup.r
speed_resolution = SPEED_RESOLUTION_DIVIDED_BY_CUP_SIZE * env.cup.r
avarage_speed = AVERAGE_SPEED_DIVIDED_BY_CUP_SIZE * env.cup.r
max_move = 1 #env.action_space.high[0]
number_of_steps_between_agent_moves = ceil(spatial_resolution/(speed_resolution*timestep))
max_control_speed = max_move/timestep


agent = new_agent(timestep=number_of_steps_between_agent_moves*timestep, spatial_resolution=10, average_speed = avarage_speed, max_control_speed=max_control_speed, reset = RESET_AGENT)
set_training_params(agent, epsilon = epsilon, alpha = alpha, gamma = gamma)
for _ in range(NUMBER_OF_EPISODES):
    env.reset(INITIAL_POS)
    steps_counter = 0
    done = False
    state = env.observe()
    ball_pos = state[:2]
    ball_speed = state[2:4]
    cup_pos = state[4:6]
    next_action = notify_env_answer(agent,cup_pos,ball_pos,ball_speed,None,done,episode_begins=True)
    while not done and steps_counter < MAX_STEPS_BY_EPISODE:
        for _ in range(number_of_steps_between_agent_moves):
            state,reward,done = env.step(next_action)
            if RENDER:
                env.render()
            if done:
                break
        ball_pos = state[:2]
        ball_speed = state[2:4]
        cup_pos = state[4:6]
        zone_code,distance_from_zone = get_zone_code(cup_pos, accepted_rectangle)
        if zone_code != 'OK' and distance_from_zone > tolerance_threshold:
            reward = reward_outside_of_accepted_zone
        if RENDER:
            print(reward)
        next_action = notify_env_answer(agent,cup_pos,ball_pos,ball_speed,reward,done,zone_code=zone_code)
        steps_counter += number_of_steps_between_agent_moves

save_agent(agent)






