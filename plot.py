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

nerr = n*np.sqrt((Kerr*(pk-pf))**2 + (pkerr*K)**2)
#data[1] = n*np.sqrt((Kerr*(pk-pf))**2 + (pkerr*K)**2 + (data[1])**2)
data[2] *= K*(pk-pf)
data[0] += 273.15
data[1] += 273.15

for i in range(data[0].size())
    T1 = data[0][i]
    T2 = data[1][i]
    data[0][i] = (T1+T2)/2
    data[1][i] = (T1-T2)/2

print(n)
print(nerr)
def f(T, A, B):
    return A*np.exp(B/T)

params, covar = curve_fit(f, data[2], data[0], absolute_sigma=True, sigma=data[1], p0=(0, 1.14))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print("Parameter A und B: ")
print(uparams)


#print(1/T)
#print(n)

plt.errorbar(1/T, n, yerr = nerr, elinewidth=0.7, capthick=0.7, capsize=3, fmt=".", color="xkcd:blue", label="Messwerte")
plt.plot(1/T, f(T, *params), color="xkcd:orange", label="Regression")
plt.yscale("log")
plt.ylabel(r"$\eta/\si{\pascal\second}$")
plt.xlabel(r"$T^-1/\si{\per\second}$")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("build/temp.pdf")

#print("Viskosoítät bei Raumtemperatur: ")
#print(f(20+273.15, *params))
