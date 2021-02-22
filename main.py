# import Modules
import pygame
import os

# import files
from game import Game

# pygame
pygame.init()
screen = pygame.display.set_mode((1200, 700))
scale = 0.53/700  # m/pixel
done = False
clock = pygame.time.Clock()
last_points = []
pygame.font.init()
myfont = pygame.font.SysFont('FUTURA', 100)
textsurface = myfont.render('WIN', False, (255, 255, 255))

# physics
game = Game(scale)

while not done:
    deltaT = clock.get_time()/2000
    for event in pygame.event.get():
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE] or event.type == pygame.QUIT:
            done = True
        if pressed[pygame.K_UP]:
            game.reset()
    game.set_cup(pygame.mouse.get_pos())
    res = game.move(deltaT)
    if res == 'loose':
        print(res)
        game.reset()

    # draw
    screen.fill((0, 0, 0))

    last_points.append(game.ball.get_pos())
    if len(last_points) > 20:
        last_points.pop(0)
    for p in last_points:
        pygame.draw.circle(screen, (0, 0, 255), p, 2, 2)

    pygame.draw.polygon(screen, (0, 255, 0), game.cup.triangle())
    pygame.draw.circle(
        screen, (0, 255, 0), game.ball.get_pos(), 5, 1)
    pygame.draw.line(screen, (0, 255, 0), game.cup.get_pos(),
                     game.ball.get_pos(), 1)

    if res == 'win':
        screen.blit(textsurface, (10, 10))
    pygame.display.flip()
    if res == 'win':
        pygame.time.wait(800)
        game.reset()
    clock.tick(60)
