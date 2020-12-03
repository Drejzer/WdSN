import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import data # plik z danymi uczacymi

pygame.init()
EKRAN = pygame.display.set_mode((900,900))
#pygame.display.update()

def fourier_transform(x):
    a = np.abs(np.fft.fft(x))
    a[0]=0
    return a/np.amax(a)

def noisy(input):
    ninp=input
    random.seed()
    for i in range(ninp.size):
        if random.random()>0.9875:
            ninp[i]^=True
    return ninp

class Adaline(object):


        def __init__(self,no_of_inputs,name="", learning_rate=0.01,iters=100,bias=False):
            self.iters = iters
            self.learning_rate = learning_rate
            self.no_of_inputs = no_of_inputs
            self.errors=[]
            self.weights = np.random.random(2*self.no_of_inputs)
            self.name=name
            if bias:
                self.weights = np.concatenate([0],self.weights)
        
        def _standarise(self,x):
            mx= np.mean(x)
            sx=np.std(x)
            newx = [(i-mx)/sx for i in x]
            return newx

        def _normalise(self,x):
            mn = np.min(x)
            mx = np.max(x)
            newx = [(i-mn)/(mx-mn) for i in x]
            return newx
        
        def train(self,training_data_x,training_data_y):
            training_data_x = self._standarise(training_data_x)
            for _ in range(self.iters):
                e=0
                for x, y in zip(training_data_x,training_data_y):
                    out = self.output(x)
                    self.weights += self.learning_rate*(y-out)*x
                    #self.weights[0]+= self.learning_rate*(y-out)
                    e+=(y-out)**2
                self.errors.append(e)
            plt.plot(range(len(self.errors)),self.errors)
            plt.savefig('learning_curve_'+self.name+'.pdf')
            plt.close
        
        def _activation(self,x):
            return x

        def _activation_derivative(self,x):
            return 1

        def output(self, inp):
            inp=self._normalise(inp)
            #inp = np.concatenate([input,fourier_transform(input)])
            #inp = np.concatenate([1],inp)
            summation = self._activation(np.dot(self.weights,inp))
            return summation

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
    for i in arr:#wypisuje tablicę będącą reprezentacją klikalnych kwadratów
        j+=1
        text = np.array2string(i)
        font=pygame.font.Font("freesansbold.ttf",16)
        ArrSurf = font.render(text,True,(255,255,255))
        ArrRect = ArrSurf.get_rect()
        ArrRect.center = (750,25*(j+1))
        EKRAN.blit(ArrSurf,ArrRect)
    
    pred=[] #lista predykcji perceptronów, który perceptron "wypalił"
    inp=np.ravel(slots)
    tmp=fourier_transform(inp)
    inp=np.concatenate((inp,tmp))
    outs = [p.output(inp) for p in perceptrons]
    res=0
    for i in range(10):
        if outs[i]>outs[res]:
            pred.insert(0,i)
            res=i
    if res==0:
        pred.inset(0,0)

    pred = np.array(pred) 
    text = np.array2string(pred)
    font=pygame.font.Font("freesansbold.ttf",16)
    ArrSurf = font.render(str(res)+'  '+text,True,(255,255,255))
    ArrRect = ArrSurf.get_rect()
    ArrRect.center = (750,800)
    EKRAN.blit(ArrSurf,ArrRect)

slots = np.full((9,9),0,int)

perceptrons = []
train_inputs= [np.ravel(n) for n in data.dataset]
for v in range(10):
    perceptrons.append(Adaline(9*9,name=str(v)))

def trainer(): #Trenuje perceptrony
    for _ in range(4):
        xtab=[]
        for i in train_inputs:
            i = noisy(i)
            xtab.append(np.concatenate((i,fourier_transform(i))))
        for j in range(10):
            labelz=np.zeros(10)-1
            labelz[j]=1
            dataset=list(zip(xtab,labelz))
            random.shuffle(dataset) # losowa kolejnosic
            a,b = zip(*dataset)
            perceptrons[j].train(a,b)

trainer()
# główna pętla pygame
while True:
    mpos=pygame.mouse.get_pos() #pozycja myszy (pixele)
    mposg = (0,0) #pozycja myszy (pole tablicy) 
    for i in range(9):
        for j in range(9):
            if ((75*j+j) <= mpos[0] < (75*(j+1)+j+1)) and ((75*i+i) <= mpos[1] < (75*(i+1)+i+1)):
                pygame.draw.rect(EKRAN,(0,0,255),(75*j+j,75*i+i,75,75))
                mposg=(i,j)

    for event in pygame.event.get():
        if event.type==QUIT: #zamkniecie okna
            pygame.quit()
            sys.exit()
        elif event.type==MOUSEBUTTONUP: #kliknięcie w pole tablicy, przelacza on/of
            slots[mposg[0],mposg[1]]^=True
        elif event.type==pygame.KEYDOWN: # wejście z klawiatury, ustawia tablice na reprezentacje 0-9
            if event.key==pygame.K_0:
                slots = np.full((9,9),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[8][i]=1
                #print(slots)
                slots=np.array(data.dataset[0])
            elif event.key==pygame.K_1:
                slots = np.full((9,9),0,int)
                for i in range(9):
                    slots[i][4]=1
                slots=np.array(data.dataset[1])
                #print(slots)
            elif event.key==pygame.K_2:
                slots = np.full((9,9),0,int)
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                    slots[8-i][0]=1
                    slots[i][4]=1
                slots=np.array(data.dataset[2])
                #print(slots)
            elif event.key==pygame.K_3:
                slots = np.full((9,9),0,int)
                for i in range(9):
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                slots=np.array(data.dataset[3])
                #print(slots)
            elif event.key==pygame.K_4:
                slots = np.full((9,9),0,int)
                for i in range(9):
                    slots[i][4]=1
                for i in range(5):
                    slots[4][i]=1
                    slots[i][0]=1
                slots=np.array(data.dataset[4])
                #print(slots)
            elif event.key==pygame.K_5:
                slots = np.full((9,9),0,int)
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                    slots[i][0]=1
                    slots[8-i][4]=1
                slots=np.array(data.dataset[5])
                #print(slots)
            elif event.key==pygame.K_6:
                slots = np.full((9,9),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[4][i]=1
                    slots[8][i]=1
                    slots[0][i]=1
                for i in [1,2,3]:
                    slots[i][4]=0
                slots=np.array(data.dataset[6])
                #print(slots)
            elif event.key==pygame.K_7:
                slots = np.full((9,9),0,int)
                for i in range(5):
                    slots[0][i]=1
                for i in range(3):
                    slots[i][4]=1
                    slots[i+2][3]=1
                    slots[i+4][2]=1
                    slots[i+6][1]=1
                slots=np.array(data.dataset[7])
                #print(slots)
            elif event.key==pygame.K_8:
                slots = np.full((9,9),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[8][i]=1
                    slots[4][i]=1
                slots=np.array(data.dataset[8])
                #print(slots)
            elif event.key==pygame.K_9:
                slots = np.full((9,9),0,int)
                for i in range(9):
                    slots[i][0]=1
                    slots[i][4]=1
                for i in range(5):
                    slots[0][i]=1
                    slots[4][i]=1
                    slots[8][i]=1
                for i in [1,2,3]:
                    slots[8-i][0]=0
                slots=np.array(data.dataset[9])
                #print(slots)
            elif event.key==pygame.K_t: #trenowanie perceptronów
                trainer()
            elif event.key==pygame.K_n:
                tmp = np.ravel(slots)
                #print(tmp)
                tmp = noisy(tmp)
                #print(tmp)
                slots = tmp.reshape((9,9))
            elif event.key==K_SPACE: # czyszczenie tablicy
                slots = np.full((9,9),0,int)
            elif event.key==K_UP:# translacja sztrzałkami
                nsl = np.full((9,9),0,int)
                for i in range(9):
                    for j in range(9):
                        nsl[i][j]=slots[(10+i)%9][j]
                slots = nsl
            elif event.key==K_DOWN:
                nsl = np.full((9,9),0,int)
                for i in range(9):
                    for j in range(9):
                        nsl[i][j]=slots[(8+i)%9][(9+j)%9]
                slots = nsl
            elif event.key==K_LEFT:
                nsl = np.full((9,9),0,int)
                for i in range(9):
                    for j in range(9):
                        nsl[i][j]=slots[i][(10+j)%9]
                slots = nsl
            elif event.key==K_RIGHT:
                nsl = np.full((9,9),0,int)
                for i in range(9):
                    for j in range(9):
                        nsl[i][j]=slots[i][(8+j)%9]
                slots = nsl
                

    spit_table(slots)#wypisywanie tekstu
    pygame.display.update()
    EKRAN.fill((0,0,0))#czyszczenie ekranu
    for i in range(9):
        for j in range(9): #rysowanie tablicy
            if slots[i][j]:
                pygame.draw.rect(EKRAN,(0,255,0),(75*j+j,75*i+i,75,75))
            else:
                pygame.draw.rect(EKRAN,(255,0,0),(75*j+j,75*i+i,75,75)) 
