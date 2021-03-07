import numpy as np
from numpy.random import rand, poisson
from math import cos, sin, pi
import pickle

"""
This file creates and trains agents that use the cartesian product of a discretization of the vector cup->ball 
                                                                   and of a discretization of ball speed as states space,
and a discretization of the cup control speed as action space.
All these discretizations work with cartesian coordinates.

The agents are trained using a function of intermediar rewards.
"""

RESOLUTION_DIVISION_IF_OUTSIDE_OF_ZONE = 10
POISSON_VARIANCE_AMPLIFICATION = 5

def new_agent(timestep = 0.01, spatial_resolution = 0.1, average_speed = 100, max_control_speed = 500, discretization_mode = 'cartesian', save_file = 'agents/agent_v1.pickle', reset = False):
    if not reset:
        with open(save_file,'rb') as pickle_in:
            agent = pickle.load(pickle_in)
        print(f"Loaded agent with {len(agent['action_values'])} states !")
    else:
        agent = {
            'save_file': save_file,
            'timestep': timestep,
            'spatial_resolution': spatial_resolution,
            'speed_resolution': spatial_resolution/timestep,
            'average_speed': average_speed,
            'max_control_speed': max_control_speed,
            'discretization_mode': discretization_mode,
            'action_values': {},
            'last_state': None,
            'last_action': None,
            'epsilon': 0,
            'alpha': 0,
            'gamma': 0.9
        }
    return agent

def save_agent(agent):
    with open(agent['save_file'], 'wb') as pickle_out:
        pickle.dump(agent, pickle_out)
    print(f"Saved agent with {len(agent['action_values'])} states !")

def set_training_params(agent, epsilon = 0.1, alpha=0.1, gamma = 0.9):
    agent['epsilon'] = epsilon
    agent['alpha'] = alpha
    agent['gamma'] = gamma

def discretize(variable, resolution):
    return int(variable//resolution)

def to_continuous(discretized, resolution):
    return (discretized+1/2)*resolution

def agent_representation_of_state(agent, cup_pos, ball_pos, ball_speed, zone_code):
    spatial_resolution = (agent['spatial_resolution'] if zone_code == 'OK' else agent['spatial_resolution']/RESOLUTION_DIVISION_IF_OUTSIDE_OF_ZONE)
    speed_resolution = (agent['speed_resolution'] if zone_code == 'OK' else agent['speed_resolution']/RESOLUTION_DIVISION_IF_OUTSIDE_OF_ZONE)
    vector_cup_to_ball = np.array(ball_pos) - np.array(cup_pos)
    discretized_x = discretize(vector_cup_to_ball[0], spatial_resolution)
    discretized_y = discretize(vector_cup_to_ball[1], spatial_resolution)
    discretized_vx = discretize(ball_speed[0], speed_resolution)
    discretized_vy = discretize(ball_speed[1], speed_resolution)
    state = "$".join([zone_code,str(discretized_x),str(discretized_y),str(discretized_vx),str(discretized_vy)])
    return state


def random_action(agent):
    max_control_speed = agent['max_control_speed']
    average_speed = agent['average_speed']
    speed_resolution = agent['speed_resolution']
    speed = average_speed + POISSON_VARIANCE_AMPLIFICATION * (poisson(average_speed)-average_speed)
    v = max(0,min(max_control_speed, speed))
    theta = pi * (-1 + 2 * rand())
    vx = v * cos(theta)
    vy = v * sin(theta)
    agent_action = "$".join([str(discretize(vx,speed_resolution)),str(discretize(vy,speed_resolution))])
    return agent_action

def decode_agent_action(agent, agent_action):
    timestep = agent['timestep']
    speed_resolution = agent['speed_resolution']
    vx,vy = map(lambda x: float(x), agent_action.split('$'))
    vx = to_continuous(vx, speed_resolution)
    vy = to_continuous(vy, speed_resolution)
    return (vx, vy)

def choose_action(agent, state):
    action_values = agent['action_values']
    if state not in action_values:
        action_values[state] = {'best_action': None, 'actions_of_state_values':{}}
    if rand() < agent['epsilon'] or action_values[state]['best_action'] is None:
        action = random_action(agent)
        action_values[state]['actions_of_state_values'].setdefault(action,0)
    else:
        action = action_values[state]['best_action']
    return action

def notify_env_answer(agent, cup_pos, ball_pos, ball_speed, reward, done, zone_code = 'OK', episode_begins = False):
    current_state = agent_representation_of_state(agent, cup_pos, ball_pos, ball_speed, zone_code)
    next_action = (choose_action(agent,current_state) if not done else None)
    if not episode_begins:
        action_values = agent['action_values']
        action_value_observation = reward + agent['gamma'] * (action_values[current_state]['actions_of_state_values'][next_action] if not done else 0)
        prev_action_value = action_values[agent['last_state']]['actions_of_state_values'][agent['last_action']]
        new_action_value = prev_action_value + agent['alpha']*(action_value_observation-prev_action_value)
        action_values[agent['last_state']]['actions_of_state_values'][agent['last_action']] = new_action_value
        best_action = action_values[agent['last_state']]['best_action']
        if best_action is None or new_action_value > action_values[agent['last_state']]['actions_of_state_values'][best_action]:
            action_values[agent['last_state']]['best_action'] = agent['last_action']
    agent['last_state'] = current_state
    agent['last_action'] = next_action
    return (decode_agent_action(agent,next_action) if next_action else None)














