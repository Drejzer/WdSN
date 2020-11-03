import sys
import numpy as np
import random
import pygame
from pygame.locals import *
import data

pygame.init()
EKRAN = pygame.display.set_mode((900,900))
#pygame.display.update()

def noisy(input):
    ninp=input
    random.seed()
    for i in ninp:
        if random.random()>0.95:
            if i:
                i=0
            else:
                i=1
    return ninp

class Perceptron(object):


        def __init__(self,no_of_inputs, learning_rate=0.01,iters=100):
            self.iters = iters
            self.learning_rate = learning_rate
            self.no_of_inputs = no_of_inputs
            self.weights = np.zeros(self.no_of_inputs +1)
            for i in self.weights:
                i = random.uniform(-0.5,0.5)

        def train(self,training_data,labels):
            datset=list(zip(training_data,labels))
            random.shuffle(datset)
            a,b = zip(*datset)
            for _ in range(self.iters):
                for input, label in zip(a,b):
                    input=noisy(input)
                    prediction = self.output(input)
                    self.weights[1:]+=self.learning_rate * (label - prediction)*input
                    self.weights[0]+=self.learning_rate * (label - prediction)

        def output(self, input):
            summation = np.dot(input,self.weights[1:])+self.weights[0]
            if summation > 0:
                activation = 1
            else:
                activation=0
            return activation

def check_number():
    pred=[]
    inp=np.ravel(slots)
    for i in range(10):
        if perceptrons[i].output(inp)==1:
            pred.append(i)
    pred = np.array(pred)    
    text = np.array2string(pred)
    font=pygame.font.Font("freesansbold.ttf",16)
    ArrSurf = font.render(text,True,(255,255,255))
    ArrRect = ArrSurf.get_rect()
    ArrRect.center = (750,800)
    EKRAN.blit(ArrSurf,ArrRect)

def spit_table(arr):
    j=0
    for i in arr:
        j+=1
        text = np.array2string(i)
        font=pygame.font.Font("freesansbold.ttf",16)
        ArrSurf = font.render(text,True,(255,255,255))
        ArrRect = ArrSurf.get_rect()
        ArrRect.center = (750,25*(j+1))
        EKRAN.blit(ArrSurf,ArrRect)
    
    pred=[]
    inp=np.ravel(slots)
    for i in range(10):
        if perceptrons[i].output(inp)==1:
            pred.append(i)
    pred = np.array(pred)    
    text = np.array2string(pred)
    font=pygame.font.Font("freesansbold.ttf",16)
    ArrSurf = font.render(text,True,(255,255,255))
    ArrRect = ArrSurf.get_rect()
    ArrRect.center = (750,800)
    EKRAN.blit(ArrSurf,ArrRect)

slots = np.full((9,5),0,int)

perceptrons = []
train_inputs= [np.ravel(n) for n in data.dataset]
for _ in range(10):
    perceptrons.append(Perceptron(5*9))
def trainer():
    for i in range(10):
        labelz= np.zeros(10)
        labelz[i]=1
        for _ in range(5):
            perceptrons[i].train(train_inputs,labelz)


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
        elif event.type==MOUSEBUTTONUP:
            slots[mposg[0],mposg[1]]^=True
        elif event.type==pygame.KEYDOWN:
            if event.key==pygame.K_0:
                slots = np.full((9,5),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[8][i]=1
                print(slots)
            elif event.key==pygame.K_1:
                slots = np.full((9,5),0,int)
                for i in range(9):
                    slots[i][4]=1
                print(slots)
            elif event.key==pygame.K_2:
                slots = np.full((9,5),0,int)
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                    slots[8-i][0]=1
                    slots[i][4]=1
                print(slots)
            elif event.key==pygame.K_3:
                slots = np.full((9,5),0,int)
                for i in range(9):
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                print(slots)
            elif event.key==pygame.K_4:
                slots = np.full((9,5),0,int)
                for i in range(9):
                    slots[i][4]=1
                for i in range(5):
                    slots[4][i]=1
                    slots[i][0]=1
                print(slots)
            elif event.key==pygame.K_5:
                slots = np.full((9,5),0,int)
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                    slots[i][0]=1
                    slots[8-i][4]=1
                print(slots)
            elif event.key==pygame.K_6:
                slots = np.full((9,5),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[4][i]=1
                    slots[8][i]=1
                    slots[0][i]=1
                for i in [1,2,3]:
                    slots[i][4]=0
                print(slots)
            elif event.key==pygame.K_7:
                slots = np.full((9,5),0,int)
                for i in range(5):
                    slots[0][i]=1
                for i in range(3):
                    slots[i][4]=1
                    slots[i+2][3]=1
                    slots[i+4][2]=1
                    slots[i+6][1]=1
                print(slots)
            elif event.key==pygame.K_8:
                slots = np.full((9,5),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[8][i]=1
                    slots[4][i]=1
                print(slots)
            elif event.key==pygame.K_9:
                slots = np.full((9,5),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                for i in [1,2,3]:
                    slots[8-i][0]=0
                print(slots)
            elif event.key==pygame.K_SPACE:
                check_number()
            elif event.key==pygame.K_t:
                trainer()



    spit_table(slots)
    pygame.display.update()
    EKRAN.fill((0,0,0))
    for i in range(9):
        for j in range(5):
            if slots[i][j]:
                pygame.draw.rect(EKRAN,(0,255,0),(100*j+j,100*i+i,100,100))
            else:
                pygame.draw.rect(EKRAN,(255,0,0),(100*j+j,100*i+i,100,100))
