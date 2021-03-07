# import Modules
import pygame
import os

# import files
from gym_bilboquet.envs.bilboquet_env import GameAI

# pygame
pygame.init()
screen = pygame.display.set_mode((1200, 700))
scale = 0.53/700  # m/pixel
done = False
clock = pygame.time.Clock()
last_points = []
pygame.font.init()
myfont = pygame.font.SysFont('FUTURA', 100)
textwin = myfont.render('WIN', False, (255, 255, 255))
textloose = myfont.render('LOOSE', False, (255, 255, 255))

# physics
game = GameAI(scale)
reset = True

while not done:
    deltaT = clock.get_time()/2000

    for event in pygame.event.get():
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE] or event.type == pygame.QUIT:
            done = True
        if pressed[pygame.K_UP]:
            reset = True

    if reset:
        game.reset(pygame.mouse.get_pos())
        reset = False

    game.set_cup(pygame.mouse.get_pos())
    res = game.move(deltaT)
    print(game.reward(res))
    # print(game.observe())
    # print(game.reward())

    # Draw elements
    screen.fill((0, 0, 0))

    # blue points
    last_points.append(game.ball.get_pos())
    if len(last_points) > 20:
        last_points.pop(0)
    for p in last_points:
        pygame.draw.circle(screen, (0, 0, 255), p, 2, 2)

    # string, ball and cup
    if res != 'win':
        pygame.draw.line(screen, (0, 100, 0), game.cup.get_pos(),
                         game.ball.get_pos(), 1)
        pygame.draw.circle(
            screen, (0, 255, 0), game.ball.get_pos(), 5, 1)
    pygame.draw.polygon(screen, (0, 255, 0), game.cup.triangle())

    if res == 'win':
        screen.blit(textwin, (10, 10))
    if res == 'loose':
        screen.blit(textloose, (10, 10))
    pygame.display.flip()
    if res != None:
        pygame.time.wait(800)
        reset = True
        clock.tick()
    clock.tick(60)
