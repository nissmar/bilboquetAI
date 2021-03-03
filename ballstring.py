import cmath
import math


class String:
    def __init__(self, cup, ball, length=200):
        self.forceConstant = 10  # Nm^-1
        self.length = length
        self.lastCalcDistance = -1
        self.damping = 0.5
        self.cup = cup
        self.ball = ball

    def apply_tension(self, dt):
        stringVector = self.ball.pos-self.cup.pos
        connectionDist = abs(stringVector)
        if connectionDist > self.length and dt > 0:
            ori = self.ball.pos
            self.ball.pos -= stringVector * \
                (connectionDist-self.length)/self.length
            self.ball.v += (self.ball.pos-ori)/dt
