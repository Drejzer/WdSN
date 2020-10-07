import sys
import pygame
from pygame.locals import *

pygame.init()
EKRAN = pygame.display.set_mode((900,900))
pygame.display.update()

slots = []
for i in range(9):
    row = []
    for j in range(5):
        row.append(False)
    slots.append(row)

while True:
    mpos=pygame.mouse.get_pos()
    for i in range(9):
        for j in range(5):
            if ((100*j) <= mpos[0] < (100*(j+1))) and ((100*i) <= mpos[1] < (100*(i+1))):
                pygame.draw.rect(EKRAN,(0,0,255),(100*j,100*i,100,100))
    for i in range(9):
        for j in range(5):
            if slots[i][j]:
                pygame.draw.rect(EKRAN,(255,255,255),(100*j,100*i,100,100))

    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==MOUSEBUTTONUP:
            slots[min(int(mpos[1]/100),8)][min(int(mpos[0]/100),4)]^=True
    pygame.display.update()
    for i in range(9):
        for j in range(5):
            if slots[i][j]:
                pygame.draw.rect(EKRAN,(255,255,255),(100*j,100*i,100,100))
            else:
                pygame.draw.rect(EKRAN,(0,0,0),(100*j,100*i,100,100))
