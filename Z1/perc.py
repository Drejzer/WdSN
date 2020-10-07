import pygame
from pygame.locals import *

pygame.init()
DISPLAYSURF = pygame.display.set_mode((300,300))
pygame.draw.circle(DISPLAYSURF, (0,0,0), (200,50),30)

while True:
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sye.exit()


