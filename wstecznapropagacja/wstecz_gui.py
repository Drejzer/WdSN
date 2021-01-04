import pygame
from pygame.locals import *
import sys

pygame.init()
EKRAN = pygame.display.set_mode((900,400))

mousepos=pygame.mouse.get_pos()

pygame.display.update()

def display_func():
    pygame.display.update()
    
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==MOUSEBUTTONUP:
            mousepos=pygame.mouse.get_pos()
            print(mousepos)



