from ball import Ball, Cup
from string import String


class Game():
    def __init__(self, scale=0.53/700):
        self.ball = Ball(0, 0)
        self.ball.fall()
        self.cup = Cup(0, 0, 30)
        self.string = String(self.cup, self.ball)
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
        """move the element during dt. Returns 'win' 'loose' or None"""
        self.ball.apply_accel(dt, self.scale)
        self.string.apply_tension(dt)
        if self.cup.in_triangle(self.ball, dt):
            if self.cup.is_win(self.ball):
                return 'win'
            return 'loose'
        return

    def observe(self):
        """returns the positions of the elements"""
        return self.ball.get_pos() + self.ball.get_speed() + self.cup.get_pos()
