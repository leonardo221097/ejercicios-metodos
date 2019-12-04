#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
# funcion coseno
def f(x):
    return np.cos(x)
def cos_analitica(x):
    return -np.sin(x)

def funcion(t,h):
    FD = (f( t +h ) - f( t ) ) / h 
    error1=abs((FD-cos_analitica(t))/cos_analitica(t))
    
    CD = ( f( t +h / 2 ) -f( t-h / 2 ) ) / h 
    error2=abs((CD-cos_analitica(t))/cos_analitica(t))
    
    ED = ( 8*( f ( t +h / 4 )-f( t-h / 4 ) ) -( f( t +h / 2 )-f( t-h / 2 ) ) ) / 3 / h 
    error3=abs((ED-cos_analitica(t))/cos_analitica(t))
    return error1,error2,error3

t=[0.1,1,100]
h=np.logspace(-10,-1,50)
k=1
plt.figure(figsize=(15,10))
for i in t:
    e1, e2,e3 =funcion(i,h)
    plt.subplot(2,2,k)
    plt.plot(h,e1)
    plt.plot(h,e2)
    plt.plot(h,e3)
    c = "t="+str(i)
    plt.title(c)
    plt.xlabel(' logspace(h)')
    plt.ylabel('|error|')
    plt.loglog()
    k+=1
plt.savefig('error_coseno')
    


# In[35]:


#funcion exponencial
def f1(x):
    return np.exp(x)


def funcion1(t,h):
    FD = (f1( t +h ) - f1( t ) ) / h 
    CD = ( f1( t +h / 2 ) -f1( t-h / 2 ) ) / h 
    ED = ( 8*( f1 ( t +h / 4 )-f1( t-h / 4 ) ) -( f1( t +h / 2 )-f1( t-h / 2 ) ) ) / 3 / h 
    error11=abs((FD-f1(t))/f1(t))
    error22=abs((CD-f1(t))/f1(t))
    error33=abs((ED-f1(t))/f1(t))
    
    return error11,error22,error33

t1=[0.1,1,100]
h1=np.logspace(-10,-1,50)
m=1
plt.figure(figsize=(15,10))
for j in t1:
    e11, e22,e33 =funcion1(j,h1)
    plt.subplot(2,2,m)
    plt.plot(h1,e11)
    plt.plot(h1,e22)
    plt.plot(h1,e33)
    b= "t="+str(j)
    plt.title(b)
    plt.xlabel(' logspace(h)')
    plt.ylabel('|error|')
    plt.loglog()
    m+=1
plt.savefig('error_exponencial')
    


# In[52]:


#segunda derivada coseno ec. 7.18-7.19
def y(x):
    return np.cos(x)
def cos_anal(x):
    return -np.cos(x)

def funcion2(t,h):
    sd1=((y(t+h)-y(t))-(y(t)-y(t-h)))/h**2
    err1=abs((sd1-cos_anal(t))/cos_anal(t))
    
    sd2=(y(t+h)+y(t-h)-2*y(t))/h**2
    err2=abs((sd2-cos_anal(t))/cos_anal(t))
    return err1,err2
    
t2=[0.1,1,100]
h2=np.logspace(-10,np.pi/10,50)
n=1
plt.figure(figsize=(15,10))
for k in t2:
    e1, e2 =funcion2(k,h2)
    plt.subplot(2,3,n)
    plt.plot(h2,e1,label='sd1')
    plt.plot(h2,e2,label='sd2')
    
    a = "t="+str(k)
    plt.title(a)
    plt.xlabel('h')
    plt.ylabel('|error|')
    plt.loglog()
    plt.legend()
    
    n+=1
plt.savefig('error_2derivadacoseno')    
    


# In[ ]:


for i in t:
    e1, e2,e3 =funcion(i,h)
    plt.subplot(2,2,k)
    plt.plot(h,e1)
    plt.plot(h,e2)
    plt.plot(h,e3)
    c = "t="+str(i)
    plt.title(c)
    plt.xlabel(' logspace(h)')
    plt.ylabel('|error|')
    plt.loglog()
    k+=1

