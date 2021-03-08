from ball import Ball, Cup
from ballstring import String
from math import sqrt, cos
import numpy as np


class Game():
    def __init__(self, scale=0.53/700, string_length=200, cup_size=30):
        self.ball = Ball(0, 0)
        self.ball.fall()
        self.cup = Cup(0, 0, cup_size)
        self.string = String(self.cup, self.ball, string_length)
        self.scale = scale

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
        """move the element during dt. Returns 'win' 'lose' or None"""
        self.ball.apply_accel(dt, self.scale)
        self.string.apply_tension(dt)
        if self.cup.in_triangle(self.ball, dt):
            if self.cup.is_win(self.ball):
                return 'win'
            return 'lose'
        return

    def observe(self):
        """returns the positions of the elements"""
        return self.ball.get_pos() + self.ball.get_speed() + self.cup.get_pos() + self.cup.get_speed()

    def reward(self):
        bp = self.ball.get_pos()
        cp = self.cup.get_pos()
        v = (cp[0]-bp[0], cp[1]-bp[1])
        r = sqrt(v[0]**2+v[1]**2)/self.string.length
        theta = np.arctan2(v[0], v[1])
        if cos(theta) > 0:
            if r > 0:
                return cos(theta)/r
            return 0
        return (cp[1]-bp[1])/self.string.length
