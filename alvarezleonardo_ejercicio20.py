#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import matplotlib.pyplot as plt

# ejercicio 1
def s1(t):
    return 1/(1-0.9*np.sin(t))


n=100
h=2*np.pi/n
t=np.linspace(0,(n-1)*h,n)
k=np.arange(n)
t1=np.linspace(0,6,7)
y=s1(t)
yy=s1(t1)



def FT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        X[k] = 0.0j
        for n in range(N):
            X[k] += x[n] * np.exp(-2.0 * np.pi * 1.0j / N ) ** (k * n) 
        
    return X
x=s1(t)
y1= FT(x)
plt.figure(figsize=(15,6))


plt.figure (1)
y2=abs(y1)**2
plt.subplot(2,3,1)
plt.plot(t,y2)
plt.scatter(t,y2,s=5)
plt.semilogy()
plt.xlabel('t')
plt.title('power spectrum de s|w|^2')

# para realizar la grafica de autocorrelacion me guié de: https://stackoverrun.com/es/q/100936
y=s1(t)
yunbiased = y-np.mean(y) 
ynorm = np.sum(yunbiased**2) 
y5=  np.correlate(yunbiased, yunbiased, "same")/ynorm 
plt.subplot(2,3,2)
plt.plot(t,y5)
plt.xlabel('t')
plt.title(' autocorrelacion')


A= FT(y5)
plt.subplot(2,3,3)
plt.stem(k,abs(A),use_line_collection='True')
plt.xlabel('k')
plt.title('DFT autocorrelacion')
plt.savefig('potencias.png')


# In[63]:


#ejercicio 2
a=np.zeros(n)
for i in range(n):
    a[i] +=np.random.random()
def s2(alpha,t):
    m=alpha*((2*a)-1)
    s=1/(1-0.9*np.sin(t))
    return s+m
alpha=3
xs2=s2(alpha,t)


plt.figure(2,figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(t,xs2)
plt.xlabel('t')
plt.title('y(ti) = s(ti) + α(2ri − 1)')

x1= FT(xs2)
plt.subplot(1,3,2)
plt.scatter(k,abs(x1))
plt.stem(k,abs(x1),use_line_collection='True')
plt.title(' DFT y(ti)')
plt.xlabel('k')
plt.subplot(1,3,3)
yy=abs(x1)**2

plt.plot(k,yy)
plt.semilogy()
plt.xlabel('k')
plt.title('power spectrum')
plt.savefig('ruido.png')






# In[64]:



plt.figure(3,figsize=(10,10))
plt.subplot(2,2,1)
y=s2(alpha,t)
yunbiased = y-np.mean(y) 
ynorm = np.sum(yunbiased**2) 
y5=  np.correlate(yunbiased, yunbiased, "same")/ynorm 
plt.plot(t,y5)
plt.xlabel('t')
plt.title(' autocorrelacion')

AA=FT(y5)
plt.subplot(2,2,2)
plt.stem(k,abs(AA),use_line_collection='True')
plt.scatter(k,abs(AA))
plt.xlabel('k')
plt.title('DFT autocorrelacion')
plt.savefig('correlacion.png')


# In[65]:


plt.figure(4)
plt.semilogy(k,yy)
plt.scatter(k,abs(AA))
plt.stem(k,abs(AA),use_line_collection='True')
plt.xlabel('k')
plt.savefig('comparacion.png')


# In[66]:



yy=FT(s2(10**4,t))
y=s2(10**4,t)
yunbiased = y-np.mean(y) 
ynorm = np.sum(yunbiased**2) 
y5=  np.correlate(yunbiased, yunbiased, "same")/ynorm 
plt.figure(5)
plt.semilogy(k,yy)
plt.scatter(k,abs(y5))
plt.stem(k,abs(y5))
plt.xlabel('k')
plt.savefig('alpha.png')


# In[ ]:




