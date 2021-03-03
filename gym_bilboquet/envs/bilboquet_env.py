from ball import Ball, Cup
from ballstring import String

import pygame
import gym
from gym import error, spaces, utils
import numpy as np


class GameAI(gym.Env):

    def __init__(self, scale=0.53/700, string_length=200, cup_size=30, amplitude=5):
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
        self.action_space = spaces.Box(
            np.array([-amplitude, -amplitude], dtype=np.float32), np.array([amplitude, amplitude], dtype=np.float32))

    def reset(self, pos):
        self.cup.set_pos(pos)
        self.ball.set_pos(pos)
        self.ball.pos += 1.0j*self.string.length
        self.ball.immobilize()
        self.ball.fall()

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
        r = self.cup.r

        if state == "win":
            return 2

        if state == "lose":
            return -1

        if state == None:
            if bp[1] > cp[1]-r:
                return 0
            else:
                return 1/(1+abs(bp[0]-cp[0]))

    def observe(self):
        """returns the positions of the elements"""
        return self.ball.get_pos() + self.ball.get_speed() + self.cup.get_pos()

    def step(self, action):
        x, y = self.cup.get_pos()
        self.set_cup((x+action[0], y+action[1]))
        score = self.reward(self.move(self.timestep))
        done = False

        if (score == 2 or score == -1):
            done = True

        return self.observe(), score, done

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
