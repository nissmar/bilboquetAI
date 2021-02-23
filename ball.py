class Ball:
    def __init__(self, x, y):
        self.pos = complex(x, y)
        self.v = complex(0, 0)
        self.a = complex(0, 0)

    def set_pos(self, tuple):
        self.pos = complex(tuple[0], tuple[1])

    def fall(self):
        self.a = complex(0, 9.8)

    def immobilize(self):
        self.v = complex(0, 0)
        self.a = complex(0, 0)

    def get_pos(self):
        return self.pos.real, self.pos.imag

    def get_speed(self):
        return self.v.real, self.v.imag

    def apply_accel(self, dt, scale):
        self.v += self.a * dt/scale
        self.pos += self.v * dt


def ccw(A, B, C):
    return (C.imag-A.imag) * (B.real-A.real) > (B.imag-A.imag) * (C.real-A.real)


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class Cup(Ball):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.r = r

    def triangle(self):
        x, y = self.get_pos()
        return [(x, y), (x+self.r, y-self.r), (x-self.r, y-self.r)]

    def in_triangle(self, ball, dt):
        triangle = [complex(e[0], e[1]) for e in self.triangle()]
        N = 3
        A = ball.pos
        B = ball.pos - ball.v*dt
        for i in range(3):
            if intersect(A, B, triangle[i % 3], triangle[(i+1) % 3]):
                return True
        return False

    def is_win(self, ball):
        if ball.v.imag < 0:
            return False
        proj = ball.pos - ball.v/ball.v.imag * \
            (ball.pos.imag-self.pos.imag+self.r)
        return abs(proj.real-self.pos.real) < self.r
