import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def cauchy(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0)**2 + gamma**2)

def stable_branch(K, Kc):
    return np.sqrt(1.0 - Kc / K) if K > Kc else 0.0

gamma = 1.0
Kc = 2.0 * gamma

Ks = np.arange(0, 7, 0.01)
xs = np.arange(-10.0, 10.0, 0.01)

plt.figure(figsize=(16, 6))
plt.rcParams["font.size"] = 20

plt.subplot(1,2,1)
plt.xlim(-5,5)
plt.xlabel("$\omega$")
plt.ylabel("$g(\omega)$")
plt.plot(xs, cauchy(xs, 0.0, gamma), lw=2)

plt.subplot(1,2,2)
plt.xlim(0,6)
plt.xlabel("$K$")
plt.ylabel("$r$")
plt.plot(Ks, [stable_branch(K, Kc) for K in Ks], label=r"stable branch", lw=2, color="tab:blue")
plt.plot([2, 7], [0, 0], label=r"unstable branch", ls="dashed", lw=2, color="tab:red")
plt.legend()

plt.savefig("../figs/lorentz-bif.pdf", bbox_inches="tight")