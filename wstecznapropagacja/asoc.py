import numpy as np
from PIL import Image
import wstecz_gui as ui

im = np.array(Image.open("./image.png"))
im = im[:,:,:3]

sauce=im

heigth = im.shape[0]
width= im.shape[1]

norm_in= np.array([[1,x/width*0.8+0.1,y/heigth*0.8+0.1,np.sin(x),np.cos(x),np.sin(y),np.cos(y)]for y in range(heigth) for x in range(width)])
#norm_in= np.array([[1,x/width*0.8+0.1,y/heigth*0.8+0.1,np.sin(x),np.cos(x),np.sin(y),np.cos(y)]for y in range(heigth) for x in range(width)])
#norm_in= np.array([[1,x/width*0.8+0.1,y/heigth*0.8+0.1,np.sin(x),np.sin(2*x),np.cos(2*x),np.cos(x),np.sin(y),np.sin(2*y),np.cos(2*y),np.cos(y)]for y in range(heigth) for x in range(width)])
#norm_in= np.array([[1,x/width*0.8+0.1,y/heigth*0.8+0.1,np.sin(x),np.sin(2*x),np.sin(3*x),np.cos(3*x),np.cos(2*x),np.cos(x),np.sin(y),np.sin(2*y),np.sin(3*y),np.cos(3*y),np.cos(2*y),np.cos(y)]for y in range(heigth) for x in range(width)])
norm_corr= np.array([im[x,y]/255.0*0.8+0.1 for y in range(heigth) for x in range(width)])

model_shape = [int(32),int(32),int(32),int(3)]
delta = 0.01
layercount= len(model_shape)

layers=[]
imgs=[]

layers.append(np.random.randn(model_shape[0],norm_in.shape[1]))
for i in range(1,layercount):
    layers.append(np.random.randn(model_shape[i],model_shape[i-1]+1))
error=[]

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def iter(i,x,t):
    y=sigmoid(np.sum(layers[i]*x,axis=1))

    if i+1==layercount:
        d = (y-t)*y*(1-y)
        error.append(np.sum((y-t)**2))
    else:
        d,w=iter(i+1,np.concatenate(([1],y)),t)
        d=np.sum(d*np.transpose(w[:,1:]),axis=1)*y*(1-y)

    w= layers[i].copy()
    layers[i]=w-delta*np.reshape(d,(d.shape[0],1))*np.reshape(x,(1,x.shape[0]))

    return d,w

def calc(i,x):
    y=sigmoid(np.sum(layers[i]*x,axis=1))
    if i+1 ==layercount:
        return y
    else:
        return calc(i+1,np.concatenate(([1],y)))
fin=np.zeros((heigth,width,3))

def learn(length):
    step=length//27
    for i in range(length):
        j= np.random.randint(0,norm_in.shape[0],1)[0]
        iter(0,norm_in[j],norm_corr[j])
        
        if i%step==0:
            out=np.zeros((heigth,width,3))
            for x in range(width):
                for y in range(heigth):
                    out[x,y]=(calc(0,norm_in[y*heigth+x])-0.1)/0.8*255.0
            imgs.append(out)

    for x in range(width):
        for y in range(heigth):
            fin[x,y]=(calc(0,norm_in[y*heigth+x])-0.1)/0.8*255.0


learn(864000)

while True:
    ui.display_func()
    ui.EKRAN.fill((0,0,0))
    ui.EKRAN.blit(ui.pygame.surfarray.make_surface(sauce.swapaxes(0,1)),(300,0))
    ui.EKRAN.blit(ui.pygame.surfarray.make_surface(fin.swapaxes(0,1)),(500,0))
    for i in range(len(imgs)):    
        ui.EKRAN.blit(ui.pygame.surfarray.make_surface(imgs[i].swapaxes(0,1)),((i%9)*100,100+(i//9)*100))

    #print(len(imgs))
