#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import matplotlib.pyplot as plt
def fourier1(omega,t):
    return 3*np.cos(omega*t)+ 2*np.cos(3*omega*t)+np.cos(5*omega*t)
omega=1
T=2*np.pi
t=np.linspace(0,T,50)
t1=np.linspace(0,T,11)
y=fourier1(omega,t)
y1=fourier1(omega,t1)
plt.subplot(2,2,1)
plt.plot(t,y)
plt.scatter(t1,y1,color='g')



x=fourier1(omega,t1)

def transformada(x):
    N = len(x)
    x= np.zeros(N)
   
    for i in range(len(x)):
        for n in range(len(x)):
            X[i] +=  ((np.exp(-2.0 * np.pi * 1.0j*i*n )/ N )*x[i])
        
    return X

X=transformada(x)
N=len(X)
plt.subplot(2,2,2)
plt.scatter(t1, np.abs(X)/N)
plt.stem(t1, np.abs(X)/N)
plt.savefig('figure1.png')


# In[81]:


def fourier3(omega,t):
    return np.sin(omega*t)+ 2*np.sin(3*omega*t)+ 3*np.sin(5*omega*t)
y=fourier3(omega,t)
y1=fourier3(omega,t1)
plt.subplot(4,2,1)
plt.plot(t,y)
plt.scatter(t1,y1,color='g')

x= fourier3(omega,t1)
X=transformada(x)
N=len(X)
plt.subplot(4,2,2)
plt.scatter(t1, np.abs(X)/N)
plt.stem(t1, np.abs(X)/N)
plt.savefig('figure3.png')

def fourier4(omega,t):
    return 5*np.sin(omega*t)+ 2*np.cos(3*omega*t)+ np.sin(5*omega*t)

y=fourier4(omega,t)
y1=fourier4(omega,t1)
plt.subplot(4,2,3)
plt.plot(t,y)
plt.scatter(t1,y1,color='g')

x= fourier4(omega,t1)
X=transformada(x)
N=len(X)
plt.subplot(4,2,4)
plt.scatter(t1, np.abs(X)/N)
plt.stem(t1, np.abs(X)/N)
plt.savefig('figure4.png')


# In[82]:


def fourier5(t):
    a1= 5+10*np.sin(t+2)
    a2=10*np.sin(t+2)
    a3= 5+10*np.sin(t)
    a4=10*np.sin(t)
    return a1,a2,a3,a4
y1,y2,y3,y4=fourier5(t)
    
plt.subplot(4,2,1)
plt.plot(t,y1) 
plt.subplot(4,2,3)
plt.plot(t,y2) 
plt.subplot(4,2,5)
plt.plot(t,y3) 
plt.subplot(4,2,7)
plt.plot(t,y4)

x1,x2,x3,x4= fourier5(t1)
X1=transformada(x1)
X2=transformada(x2)
X3=transformada(x3)
X4=transformada(x4)
N=len(X)
plt.subplot(4,2,2)
plt.scatter(t1, np.abs(X1)/N)
plt.stem(t1, np.abs(X1)/N)
plt.subplot(4,2,4)
plt.scatter(t1, np.abs(X2)/N)
plt.stem(t1, np.abs(X2)/N)
plt.subplot(4,2,6)
plt.scatter(t1, np.abs(X3)/N)
plt.stem(t1, np.abs(X3)/N)
plt.subplot(4,2,8)
plt.scatter(t1, np.abs(X4)/N)
plt.stem(t1, np.abs(X4)/N)
plt.savefig('figure5.png')


# In[ ]:




