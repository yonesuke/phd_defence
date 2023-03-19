import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def f(x):
    return -np.cos(x)*(1-np.cos(x))

plt.rcParams['font.size']=25
plt.figure(figsize=[8,6])

plt.xlim(0,2*np.pi)
_xs = np.arange(0, 2*np.pi, 0.01)
_ys = f(_xs)
#plt.plot(_xs, _ys, label=r"$-\cos\theta+\cos^{2}\theta$", color="gray", linewidth=1, zorder=1)
plt.plot(_xs, _ys, color="gray", linewidth=1, zorder=1)
plt.plot([0, 2*np.pi], [0, 0], linestyle="dashed", color="gray", zorder=0)

N = 60
kc = 20
colors = ["white" if kc<=i<=N-kc else "tab:blue" for i in range(1,N)]
xs = [2*np.pi*l/N for l in range(1,N)]
ys = f(xs)
plt.scatter(xs, ys, zorder=10, color=colors, edgecolors="black", linewidths=1.0, s=50)

plt.xlabel(r"$2\pi pl/N$")
plt.ylabel(r"$b^{(N,p)}_{l}$")
xlocs = [0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi]
xlabs = ["$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
plt.xticks(xlocs, xlabs)
#plt.legend(loc="upper right", frameon=False)
plt.tight_layout()

plt.savefig("../figs/N60p1.pdf",bbox_inches='tight')