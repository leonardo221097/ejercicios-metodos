#!/usr/bin/env python
# coding: utf-8

# In[61]:


#ejercicio 8.4.3
#1
import numpy as np 

A=np.array([[4,-2,1],[3,6,-4],[2,1,8] ])
print ("la matriz A es",A)
A_inv = np.linalg.inv(A)
print("la matriz inversa de A es",A_inv)
print("la matriz identidad de A es",np.dot(A,A_inv))
inv_analitica=np.array([[52,17,2],[-32,30,19],[-9,-8,30]])*(1/263)

decimales=A_inv-inv_analitica
print("el numero de decimales es",decimales)


# In[62]:


#2
A=np.array([[4,-2,1],[3,6,-4],[2,1,8] ])
b1=np.array([12,-25,32] )
x1 = np.linalg.solve(A, b1)
print("x1 es",x1)
b2=np.array([4,-10,22] )
x2 = np.linalg.solve(A, b2)
print("x2 es",x2)
b3=np.array([20,-30,40] )
x3 = np.linalg.solve(A, b3)
print("x3 es",x3)


# In[64]:


#3
from numpy.linalg import eig
alpha=5
beta=3
B=np.array([[alpha,beta],[-beta,alpha]])
Es,Evectors=eig(B)
print("lamda 1,2 es:",Es)


# In[65]:


#4
I = np.array ( [ [-2,2,-3  ] , [2,1,-6 ] , [ -1,-2,0 ] ] ) 
print (" I = " , I ,)

Es, evectors = eig ( I )
print ('autovalores son =', Es )
print("")
print("el auto vector de 5 es ",evectors[1])
print("el auto vector de -3 es ",evectors[0])
print("el auto vector de -3 es ",evectors[2])
print("")
#autovector analitico 
x1=np.array([-1,-2,1])*(1/np.sqrt(6))
x2=np.array([-2,1,0])*(1/np.sqrt(5))
x3=np.array([3,0,1])*(1/np.sqrt(10))

dif1=(evectors[1]-x1)
dif2=evectors[0]-x2
dif3=evectors[2]-x3
print("la diferencia de x1  es", dif1)
print("la diferencia de x2  es", dif2)
print("la diferencia de x3  es", dif3)



# In[70]:


#5
A=[]
for i in range(1,101):
    for j in range(1,101):
        a=1/(i+j-1)
        A.append(a)
A1=np.array(A)
A2=A1.reshape(100,100) #matriz A
B=[]
for k in range (1,101):
    b=1/k
    B.append(b)
B1=np.array(B)
B2=B1.reshape(100,1) #arreglo de b
y= np.linalg.solve(A2,B2) #VALORES DE Y
print("la solucion de y es:")
print(y)

    





# In[ ]:




