#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[48]:


datos = pd.read_csv('USArrests.csv')

lugar=np.array(['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 
       'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois'
       , 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 
       'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 
       'Montana', 'Nebraska', 'Nevada', 'NewHampshire', 'NewJersey', 'NewMexico',
       'NewYork', 'NorthCarolina', 'NorthDakota', 'Ohio', 'Oklahoma', 'Oregon',
       'Pennsylvania', 'RhodeIsland', 'SouthCarolina', 'SouthDakota',
       'Tennessee', 'Texas', 'Utah', 
       'Vermont', 'Virginia', 'Washington', 'WestVirginia', 'Wisconsin', 'Wyoming'])
nvectors=np.array(['Murder','Assault','UrbanPop','Rape'])
lista1=[]
lista2=[]
lista3=[]
lista4=[]

x1=np.array(datos['Murder'])
for i in range(len(x1)):
    x=(x1[i]-x1.mean())/x1.std()
    lista1.append(x)
x2=np.array(datos['Assault'])
for j in range(len(x2)):
    x=(x2[j]-x2.mean())/x2.std()
    lista2.append(x)
x3=np.array(datos['UrbanPop'])
for k in range(len(x3)):
    x=(x3[k]-x3.mean())/x3.std()
    lista3.append(x)
x4=np.array(datos['Rape'])
for m in range(len(x4)):
    x=(x4[m]-x4.mean())/x4.std()
    lista4.append(x)

A=np.array([lista1,lista2,lista3,lista4])
Acov=np.cov(A)
aval, avec = np.linalg.eig(Acov)


vecp1= avec[:,0]
vecp2 = avec[:,1]

ep1=np.dot(vecp1,A)
ep2=np.dot(vecp2,A)
plt.figure(1)
plt.figure(figsize=(18,18))
# se pone sigo negativo en el eje y para que la grafica sea igual a la del libro.
plt.scatter(ep1,-ep2)
# me guié de la documentación de pyplot para ponerle nombre a las variable dentro de la grafica.
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.text.html
for i in range(len(lugar)):
    plt.text(ep1[i],-ep2[i],lugar[i])
#para hacer los vectores me guié de:
# http://benalexkeen.com/principle-component-analysis-in-python/
plt.arrow(0.0, 0.0, 2*vecp1[0], -2*vecp2[0],color='g',width=0.0005,head_width=0.08)
plt.arrow(0.0, 0.0, 2*vecp1[1], -2*vecp2[1],color='g',width=0.0005,head_width=0.08)
plt.arrow(0.0, 0.0, 2*vecp1[2], -2*vecp2[2],color='g',width=0.0005,head_width=0.08)
plt.arrow(0.0, 0.0, 2*vecp1[3], -2*vecp2[3],color='g',width=0.0005,head_width=0.08)

for i in  range (len(nvectors)):
    plt.text(2*vecp1[i], -2*vecp2[i],nvectors[i],color='r')
    
plt.xlabel('componente principal')
plt.ylabel('componente secundaria')
plt.savefig('arrestos.png')


# In[47]:


datos = pd.read_csv('Cars93.csv')
modelo=np.array(datos['Model'])
nvector=np.array(['Horsepower','Length','Width','Fuel.tank.capacity'])

x1=np.array(datos['Horsepower'])
lista1=[]
lista2=[]
lista3=[]
lista4=[]

for i in range(len(x1)):
    x=(x1[i]-x1.mean())/x1.std()
    lista1.append(x)
x2=np.array(datos['Length'])
for j in range(len(x2)):
    x=(x2[j]-x2.mean())/x2.std()
    lista2.append(x)
x3=np.array(datos['Width'])
for k in range(len(x3)):
    x=(x3[k]-x3.mean())/x3.std()
    lista3.append(x)
x4=np.array(datos['Fuel.tank.capacity'])
for m in range(len(x4)):
    x=(x4[m]-x4.mean())/x4.std()
    lista4.append(x)

A=np.array([lista1,lista2,lista3,lista4])
Acov=np.cov(A)
aval, avec = np.linalg.eig(Acov)


vecp1= avec[:,0]
vecp2 = avec[:,1]

ep1=np.dot(vecp1,A)
ep2=np.dot(vecp2,A)
plt.figure(2)
plt.figure(figsize=(18,18))
plt.scatter(ep1,ep2)
# me guié de la documentación de pyplot para ponerle nombre a las variable dentro de la grafica.
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.text.html
for i in range(len(modelo)):
    plt.text(ep1[i],ep2[i],modelo[i])
    
#para hacer los vectores me guié de:
# http://benalexkeen.com/principle-component-analysis-in-python/
plt.arrow(0.0, 0.0, 2*vecp1[0], 2*vecp2[0],color='g',width=0.0005,head_width=0.08)
plt.arrow(0.0, 0.0, 2*vecp1[1], 2*vecp2[1],color='g',width=0.0005,head_width=0.08)
plt.arrow(0.0, 0.0, 2*vecp1[2], 2*vecp2[2],color='g',width=0.0005,head_width=0.08)
plt.arrow(0.0, 0.0, 2*vecp1[3], 2*vecp2[3],color='g',width=0.0005,head_width=0.08)
for i in  range (len(nvector)):
    plt.text(2*vecp1[i], 2*vecp2[i],nvector[i],color='r')
plt.xlabel('componente principal')
plt.ylabel('componente secundaria')
plt.savefig('Cars93.png')


# In[ ]:




