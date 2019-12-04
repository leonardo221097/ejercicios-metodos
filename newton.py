#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def funcion(x):
    return np.sqrt(10-x)*np.tan(10-x)

x= np.linspace(1,4,100)
plt.figure(1)
plt.plot(x,funcion(x))
def funcion1(x1):
    return np.sqrt(x1)*1/np.tan(np.sqrt(10-x1))-np.sqrt(10-x1)
x1= np.linspace(1,3.139,100)
plt.figure(2)
plt.plot(x1,funcion1(x1))



    


# In[3]:



def bis(f,xmenos,xmas):
   
    eps = 1e-6
    imax = 100
    error = 0
    x = 0
    for i in range(0,imax+1):
        x = (xmenos+xmas)/2
        if f(xmas)*f(x) > 0:
            xpos = x
        else:
            xneg = x
        error = abs(f(x)-eps)
        fx = f(x)
        
        
    return x
        
    



# In[ ]:





# In[25]:


def nwt(f,x1):
    imax = 100
    dx = 1e-8
    for j in range(0,imax+1):
        fx1 = f(x1)
        df = (f(x1 + dx/2)-f(x1-dx/2))/dx
        xnew = x1- fx1/df
        xn = xnew
    return xn
 
print('El resultado obtenido por bisección es de ' + str(bis(funcion,0,9)))  
print('El resultado obtenido por Newton Raphson es de ' + str(nwt(funcion,7)))


# In[ ]:


def nwt(f,xi):
    eps = 1e-7
    imax = 100
    s = 1
    error = 1
    dx = 1e-8
    while error > eps and s < imax:
        fxi = f(xi)
        der = (f(xi + h)-fxi)/h
        xnew = xi - fxi/der
        fnew = f(xnew)
        
        error = abs(fnew)
        i += 1
        xi = xnew
    return xi

