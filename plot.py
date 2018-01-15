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
K = 1.1735*10**(-8)
Kerr = 0.0056*10**(-8)
#Dichte der Kugel
pk = 2413.49
pkerr = 1.46
#Dichte der Flüssigkeit
pf = 988

data[2] += 273.15

T = np.zeros(data[0].size)
t = np.zeros(data[0].size)
terr = np.zeros(data[0].size)
n = np.zeros(data[0].size)
nerr = np.zeros(data[0].size)

T = data[2]

for i in range(data[0].size):
    T1 = data[0][i]
    T2 = data[1][i]
    t[i] = (T1+T2)/2
    terr[i] = np.abs((T1-T2)/2)

n = K*(pk-pf)*t
nerr = np.sqrt((Kerr*(pk-pf)*t)**2 + (K*t*pkerr)**2 + (K*(pk-pf)*terr)**2)
#nerr = np.sqrt((K*(pk-pf)*terr)**2)


def f(T, A, B):
    return A*np.exp(B/T)


params, covar = curve_fit(f, T, n, absolute_sigma=True, sigma = nerr, p0=(0, 1.14))
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print("Parameter A und B: ")
print(uparams)

plt.errorbar(1/T, n, yerr = nerr, elinewidth=0.7, capthick=0.7, capsize=3, fmt=".", color="xkcd:blue", label="Messwerte")
plt.plot(1/T, f(T, *params), color="xkcd:orange", label="Regression")
plt.yscale("log")
plt.ylabel(r"$\eta/\si{\pascal\second}$")
plt.xlabel(r"$T^-1/\si{\per\kelvin}$")

plt.legend()
plt.grid(which="both")

plt.tight_layout()
plt.savefig("build/temp.pdf")

for i in range(t.size):
    print(T[i], " \t& ", t[i], r"\pm", terr[i], " \t& ", n[i], r"\pm", nerr[i], " \t& ",  1/T[i], sep="", end=r"\\")
    print()
