import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def general_lorentz(x, gamma, n):
    return n * np.sin(0.5 * np.pi / n) * gamma ** (2*n - 1) / np.pi / (x ** (2*n) + gamma ** (2*n))

def general_gauss(x, gamma, n):
    return n * gamma / gamma_func(0.5 / n) * np.exp(-(gamma * x) ** (2*n))

def inf_dist(x, gamma):
    return 0.5 / gamma * (np.abs(x) < gamma)

omegas = np.arange(-6, 6, 0.01)

plt.figure(figsize=(16, 6))
plt.rcParams["font.size"] = 20

plt.subplot(1,2,1)
plt.title("(a)", loc="left")
plt.xlim(-5,5)
plt.ylim(-0.01, 0.6)
plt.xlabel("$\omega$")
plt.ylabel("$g_{n}^{(L)}(\omega)$")
for n in [1,2,3]:
    plt.plot(omegas, general_lorentz(omegas, 1.0, n), label=f"$n={n}$", lw=2)
plt.plot(omegas, inf_dist(omegas, 1.0), label=r"$n=\infty$", lw=2, ls="dashed")
plt.legend()

plt.subplot(1,2,2)
plt.title("(b)", loc="left")
plt.xlim(-5,5)
plt.ylim(-0.01, 0.6)
plt.xlabel("$\omega$")
plt.ylabel("$g_{n}^{(G)}(\omega)$")
for n in [1,2,3]:
    plt.plot(omegas, general_gauss(omegas, 1.0, n), label=f"$n={n}$", lw=2)
plt.plot(omegas, inf_dist(omegas, 1.0), label=r"$n=\infty$", lw=2, ls="dashed")
plt.legend()

plt.savefig("../figs/general_dist.pdf", bbox_inches="tight")