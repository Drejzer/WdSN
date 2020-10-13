import sys
import numpy as np
import pygame
from pygame.locals import *

pygame.init()
EKRAN = pygame.display.set_mode((900,900))
#pygame.display.update()

class Perceptron(object):


        def __init__(self,no_of_inputs, learning_rate=0.01,iters=100):
            self.iters = iters
            self.learning_rate = learning_rate
            self.no_of_inputs = no_of_inputs
            self.weights = np.zeros(self.no_of_inputs +1)

        def train(self,training_data,labels):
            for _ in range(self.iters):
                for input, label in zip(training_data, labels):
                    prediction = self.predict(input)
                    self.weights[1:]+=self.learning_rate * (label - prediction)*input
                    self.weights[0]+=self.learning_rate * (label - prediction)

        def predict(self, input):
            summation = np.dot(input,self.weights[1:])+self.weights[0]
            if summation > 0:
                activation = 1
            else:
                activation=0
            return activation

def spit_table(arr):
    j=0
    for i in arr:
        text = np.array2string(i)
        font=pygame.font.Font("freesansbold.ttf",16)
        ArrSurf = font.render(text,True,(255,255,255))
        ArrRect = ArrSurf.get_rect()
        ArrRect.center = (750,25*(j+1))
        EKRAN.blit(ArrSurf,ArrRect)
        j+=1

slots = np.full((9,5),0,int)

while True:
    mpos=pygame.mouse.get_pos()
    mposg = (0,0)
    for i in range(9):
        for j in range(5):
            if ((100*j+j) <= mpos[0] < (100*(j+1)+j+1)) and ((100*i+i) <= mpos[1] < (100*(i+1)+i+1)):
                pygame.draw.rect(EKRAN,(0,0,255),(100*j+j,100*i+i,100,100))
                mposg=(i,j)

    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==MOUSEBUTTONUP:
            slots[mposg[0],mposg[1]]^=True

    spit_table(slots)
    pygame.display.update()
    EKRAN.fill((0,0,0))
    for i in range(9):
        for j in range(5):
            if slots[i][j]:
                pygame.draw.rect(EKRAN,(0,255,0),(100*j+j,100*i+i,100,100))
            else:
                pygame.draw.rect(EKRAN,(255,0,0),(100*j+j,100*i+i,100,100))
