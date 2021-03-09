from ball import Ball, Cup
from ballstring import String

import pygame
import gym
from gym import error, spaces, utils
import numpy as np
from math import sqrt, cos


WIDTH = 1200
HEIGHT = 700

pygame.init()
pygame.font.init()
pygame.display.set_mode((WIDTH, HEIGHT))


class GameAI(gym.Env):

    def __init__(self, scale=0.53/HEIGHT, string_length=200, cup_size=30, amplitude=20, continuous=True):
        self.ball = Ball(0, 0)
        self.ball.fall()
        self.cup = Cup(0, 0, cup_size)
        self.string = String(self.cup, self.ball, string_length)
        self.scale = scale
        self.timestep = 0.01

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.last_points = []

        self.myfont = pygame.font.SysFont('FUTURA', 100)
        self.myfont2 = pygame.font.SysFont('FUTURA', 20)
        self.textwin = self.myfont.render('WIN', False, (255, 255, 255))
        self.textlose = self.myfont.render('LOSE', False, (255, 255, 255))
        self.state = None
        self.score = None
        self.amplitude = amplitude
        self.continuous = continuous
        if continuous:
            self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def reset(self, rand=False, pos=(300, 300)):
        self.cup.set_pos(pos)
        self.ball.set_pos(pos)
        bl = complex(np.random.random(), np.random.random()) if rand else 1.0j
        self.ball.pos += bl*self.string.length
        self.ball.immobilize()
        self.ball.fall()
        return self.observe()

    def set_cup(self, tuple):
        """set the postion of the cup to (x,y)"""
        self.cup.set_pos(tuple, self.timestep)

    def move(self, dt):
        """move the element during dt. Returns 'win' 'lose' or None"""
        self.ball.apply_accel(dt, self.scale)
        self.string.apply_tension(dt)
        if self.cup.in_triangle(self.ball, dt):
            if self.cup.is_win(self.ball):
                return 'win'
            return 'lose'
        return

    def reward_helper(self, z):
        s = (abs(z.real)-self.string.length)/self.string.length
        if z.imag < -self.cup.r:
            s = -s-1
        else:
            s = s-2
        return s

    def reward(self, state):
        if state == "win":
            print('WIN')
            return 1
        if state == "lose":
            return -2
        return self.reward_helper(self.ball.pos-self.cup.pos)

    def observe(self):
        """returns the positions of the elements"""
        z = (self.ball.pos-self.cup.pos)/self.string.length
        array = np.array([z.real, z.imag])
        assert self.observation_space.contains(array)
        return array

    def step(self, action):
        if self.continuous:
            action[0] *= self.amplitude
            action[1] *= self.amplitude
        else:
            action = [[self.amplitude, 0],
                      [-self.amplitude, 0], [0, 0]][action]
        x, y = self.cup.get_pos()

        fac = 0.1
        if WIDTH*fac <= x+action[0] <= WIDTH*(1-fac) and fac*HEIGHT <= y+action[1] <= HEIGHT*(1-fac):
            self.set_cup((x+action[0], y+action[1]))
        self.state = self.move(self.timestep)
        self.score = self.reward(self.state)
        done = not(self.state is None)

        return self.observe(), self.score, done,  {}

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

        score = str(self.score)
        score_display = self.myfont2.render(
            score[:min(5, len(score))], False, (255, 255, 255))
        self.screen.blit(score_display, (WIDTH*0.9, 10))
        if not(self.state is None):
            if self.state == 'win':
                self.screen.blit(self.textwin, (10, 10))
            if self.state == 'lose':
                self.screen.blit(self.textlose, (10, 10))
            pygame.display.flip()
            pygame.time.wait(800)
            self.clock.tick()
            reset = True
        else:
            pygame.display.flip()
        self.clock.tick(60)
