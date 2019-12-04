
# (20 puntos) El cálculo de números que siguen una distribución gaussiana 
# usando el Algoritmo 1.18 de Krauth (Graficar el histograma de los valores 
# y comparar con la función que describe una gaussiana).

def gauss(sigma, n_points):
    phi = np.random.random(n_points) * 2.0 * np.pi
    gamma = -np.log(np.random.random(n_points))
    r = sigma * np.sqrt(2.0 * gamma)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def gaussian(x, sigma):
    return np.exp(-x**2/(2.0*sigma**2))/np.sqrt(2.0*np.pi*sigma**2)

plt.figure()
n_points = 10000
sigma = 1.0
x, y = gauss(sigma ,n_points)

x_model = np.linspace(x.min(), x.max(), n_points)
y_model = gaussian(x_model, sigma)

_ = plt.hist(x, bins=30, density=True, label='Box-Mueller')
plt.plot(x_model, y_model, label='Modelo')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("gaussian.png")

# (20 puntos) cálculo de números que siguen una distribución gaussiana usando Metrópolis-Hastings 

def gauss_metropolis(sigma, N=100000, delta=1.0):
    lista = [np.random.random()]

    for i in range(1,N):
        propuesta  = lista[i-1] + (np.random.random()-0.5)*delta
        r = min(1, gaussian(propuesta, sigma)/gaussian(lista[i-1], sigma))
        alpha = np.random.random()
        if(alpha<r):
            lista.append(propuesta)
        else:
            lista.append(lista[i-1])
    return np.array(lista)


plt.figure()
n_points = 10000
sigma = 1.0
x = gauss_metropolis(sigma, N=n_points)

x_model = np.linspace(x.min(), x.max(), n_points)
y_model = gaussian(x_model, sigma)

_ = plt.hist(x, bins=30, density=True, label='Metropolis-Hastings')
plt.plot(x_model, y_model, label='Modelo')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("gaussian_metropolis.png")
