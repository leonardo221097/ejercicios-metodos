
import numpy as np
import matplotlib.pyplot as plt
def maxw(x,c):
    return x**2*(np.exp((-0.5*x**2)/c**2))
c=[1,2,10]
x=np.linspace(0,100,1000)
k=0
plt.subplot(2,3,1)
plt.plot(x,maxw(x,c[0]))
plt.title('c=1')
plt.subplot(2,3,2)
plt.plot(x,maxw(x,c[1]))
plt.title('c=2')
plt.subplot(2,3,3)
plt.plot(x,maxw(x,c[2]))
plt.title('c=10')
plt.savefig('mb.png')


# In[72]:


#integracion 0 a infinito
def gauss(c,f):
    N=20
    a=5
 
    
    x1, w1 = np.polynomial.legendre.leggauss(N)
    #Limites de integracion basandome en el libro de landau pagina 137
    xx = a*((1+x1)/(1-x1))
    w2 = (2*a/(1-x1)**2)*w1
    
    return np.sum(f(xx,c)*w2) 
c=np.linspace(1,20,100)
e=np.ones(100)
k=0
for i in c:
    e1=gauss(i,maxw )
    e[k]=e1
    k+=1
plt.plot(c,e)
plt.loglog()
plt.savefig('mb_int.png')


    



# In[76]:


#eercicio 3
def funcion(f,maxw,c):
    h=0.0001
    
    
    return ( f(c+h/2,maxw) -f( c-h/2,maxw ) ) / h 

c1=np.linspace(1,20,100)
ee=np.ones(100)
k1=0
for j in c1:
    e11=gauss(j,maxw )
    ee[k1]=e11
    k1+=1
plt.plot(c,e)
plt.loglog()
plt.savefig('mb_int_prime.png')

    
