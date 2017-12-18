import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy

mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

data = np.genfromtxt("content/var_temp.txt", unpack=True)

#Apperaturkonstante
K = 1.1738*10**(-8)
Kerr = 0.1338*10**(-8)
#Dichte der Kugel
pk = 2413.16
pkerr = 0.49
#Dichte der Flüssigkeit
pf = 988

n = np.zeros(int(data[0].size)*2)
nerr = np.zeros(int(data[0].size)*2)
T = np.zeros(int(data[0].size)*2)
T = np.append(data[2], data[2])
n = np.append(data[0], data[1])

n *= K*(pk-pf)
nerr = n*np.sqrt((Kerr*(pk-pf))**2 + (pkerr*K)**2)
T += 273.15

def f(T, A, B):
    return A*np.exp(B/T)

params, covar = curve_fit(f, T, n, absolute_sigma=True, sigma=nerr, p0=(0, 1.14))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print("Parameter A und B: ")
print(uparams)


#print(1/T)
#print(n)

plt.plot(1/T, n,"." ,label="Messwerte")
plt.plot(1/T, f(T, *params), label="Regression")
plt.yscale("log")
plt.ylabel(r"$\eta/\si{\pascal\second}$")
plt.xlabel(r"$T^-1/\si{\per\second}$")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("build/temp.pdf")

#print("Viskosoítät bei Raumtemperatur: ")
#print(f(20+273.15, *params))
