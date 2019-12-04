import numpy as np
import matplotlib.pyplot as plt


def trapezoide( N = 100):
    a=0
    b=1
    sumatrap = 0
    for i in range(a,N):
        m = (b-a)/(N-1)
        x = a + (i-1)*((b-a)/(N-1))
        if  i==N:
            m= ((b-a)/(N-1))/2
        else:
            m=(b-a)/(N-1)
        sumatrap +=m*np.exp(-x)
    return sumatrap




# me guié de: https://stackoverflow.com/questions/33457880/different-intervals-for-gauss-legendre-quadrature-in-numpy
def gauss( N=100):
    a=0
    b=1
    
    x1, w1 = np.polynomial.legendre.leggauss(N)
    x = (a+b)/2 + (b-a)*x1/2
    w2 = (b-a)*w1/2
    sumagauss = 0
    for j in range(0,N):
        sumagauss += np.exp(-x[j-1])*w2[j-1]
    return sumagauss


def error(funcion):
    v =1-np.exp(-1)
    error=abs((v-funcion)/v)
    return error
listanum= [4,10,20,40,80,160,320,640]
errortrap = []
for k in listanum:
    errortrap.append(error(trapezoide(k)))
print(errortrap)
errorgaus = []
for k in listanum:
    errorgaus.append(error(gauss(k)))
print(errorgaus)   


plt.plot((listanum),(errorgaus), label= 'func. Gausiana')
plt.plot((listanum), (errortrap),  label= 'Trapezoid rule')
plt.xlabel('N')
plt.ylabel('error')
plt.savefig('erroresintegrales.png')

plt.loglog()
plt.show()
