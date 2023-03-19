import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

Nmin, Nmax = 30, 600
Ns, ps, mus = np.loadtxt("connectivity.txt", delimiter=",", unpack=True)
mu_N = dict([(i, 0) for i in range(Nmin, Nmax+1)])
for i in range(len(Ns)):
    N = Ns[i]
    if N >= Nmin and N<= Nmax:
        mu = mus[i]
        if mu_N[N] < mu:
            mu_N[N] = mu

ns = [i for i in range(Nmin, Nmax+1)]
ms = [mu_N[n] for n in ns]

plt.rcParams['font.size']=22
plt.figure(figsize=[8,6])

plt.xlim(Nmin-1,Nmax+1)
plt.scatter(ns, ms, s = 10, zorder=10)
plt.plot([Nmin-1, Nmax+1], [1277./1870., 1277./1870.], linestyle="dashed", color="gray",label=r"$\textrm{Previous lower bound of } \mu_{\mathrm{c}}$")
plt.xlabel(r'$N$')
plt.ylabel(r'$\max_{1\leq p\leq \lfloor N/2\rfloor} \mu^{(N,p)}$')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("../figs/max_connect.pdf",bbox_inches='tight')