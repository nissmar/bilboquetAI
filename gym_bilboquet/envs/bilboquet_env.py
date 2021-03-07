from ball import Ball, Cup
from ballstring import String

import pygame
import gym
from gym import error, spaces, utils
import numpy as np
from math import sqrt, cos

pygame.init()
pygame.display.set_mode((1200, 700))


class GameAI(gym.Env):

    def __init__(self, scale=0.53/700, string_length=200, cup_size=30, amplitude=10):
        self.ball = Ball(0, 0)
        self.ball.fall()
        self.cup = Cup(0, 0, cup_size)
        self.string = String(self.cup, self.ball, string_length)
        self.scale = scale
        self.timestep = 0.01

        pygame.init()
        self.screen = pygame.display.set_mode((1200, 700))

        self.clock = pygame.time.Clock()
        self.last_points = []
        pygame.font.init()
        self.myfont = pygame.font.SysFont('FUTURA', 100)
        self.textwin = self.myfont.render('WIN', False, (255, 255, 255))
        self.textloose = self.myfont.render('LOSE', False, (255, 255, 255))
        self.amplitude = amplitude
        self.action_space = spaces.Discrete(5)

        high = np.array(
            [self.string.length, self.string.length], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, pos=(300, 300)):
        self.cup.set_pos(pos)
        self.ball.set_pos(pos)
        self.ball.pos += 1.0j*self.string.length
        self.ball.immobilize()
        self.ball.fall()
        return self.observe()

    def set_cup(self, tuple):
        """set the postion of the cup to (x,y)"""
        self.cup.set_pos(tuple)

    def move(self, dt):
        """move the element during dt. Returns 'win' 'loose' or None"""
        self.ball.apply_accel(dt, self.scale)
        self.string.apply_tension(dt)
        if self.cup.in_triangle(self.ball, dt):
            if self.cup.is_win(self.ball):
                return 'win'
            return 'lose'
        return

    def reward(self, state):
        bp = self.ball.get_pos()
        cp = self.cup.get_pos()
        if state == "win":
            return 10000
        if state == "lose":
            return -1
        if bp[1] < cp[1]-self.cup.r:
            s = self.string.length*10/(1+abs(bp[0]-cp[0]))
        else:
            s = abs((cp[0]-bp[0]))
        return s

    def observe(self):
        """returns the positions of the elements"""
        z = self.ball.pos-self.cup.pos
        array = np.array([z.real, z.imag])
        assert self.observation_space.contains(array)
        return array

    def step(self, action):
        action = [[self.amplitude, 0], [-self.amplitude, 0],
                  [0, self.amplitude], [0, -self.amplitude], [0, 0]][action]
        x, y = self.cup.get_pos()

        if 0 <= x+action[0] <= 1200 and 0 <= y+action[1] <= 700:
            self.set_cup((x+action[0], y+action[1]))
        state = self.move(self.timestep)
        score = self.reward(state)
        done = not(state is None)

        return self.observe(), score, done,  {}

    def render(self, close=False):
        # Draw elements
        self.screen.fill((0, 0, 0))

        # blue points
        self.last_points.append(self.ball.get_pos())
        if len(self.last_points) > 20:
            self.last_points.pop(0)
        for p in self.last_points:
            pygame.draw.circle(self.screen, (0, 0, 255), p, 2, 2)

        # string, ball and cup
        pygame.draw.line(self.screen, (0, 100, 0), self.cup.get_pos(),
                         self.ball.get_pos(), 1)
        pygame.draw.circle(
            self.screen, (0, 255, 0), self.ball.get_pos(), 5, 1)
        pygame.draw.polygon(self.screen, (0, 255, 0), self.cup.triangle())

        pygame.display.flip()

        self.clock.tick(60)
