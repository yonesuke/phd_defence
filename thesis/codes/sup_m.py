import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def f(x):
    return -np.cos(x)*(1-np.cos(x))

def t(x):
    return 4*np.pi*x-4*np.sin(2*np.pi*x)+np.sin(4*np.pi*x)

def critical_index(N):
    xs = np.array([2*np.pi*l/N for l in range(1,N)])
    ys = f(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    return kc


def sup_m(N):
    xs = np.array([2*np.pi*l/N for l in range(1,N)])
    ys = f(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    return (2*kc-1-2*sk[kc-2]/ys[kc-1])/N


eps = 10**(-8)
xl, xr = 0.335, 0.345
while (xr-xl)>eps:
    xm = (xl + xr) * 0.5
    if t(xm)>0:
        xr = xm
    else:
        xl = xm
Kc = xl

max_N=100
ns = [i for i in range(7,max_N+1)]
ss = [sup_m(n) for n in ns]

plt.figure(figsize=[8,6])
plt.rcParams['font.size']=25
plt.xlim(7,max_N)
plt.xlabel("$\widetilde{N}$")
plt.ylabel(r"$\alpha_{\widetilde{N}}/\widetilde{N}$")
# plt.plot([7,max_N],[ss[12],ss[12]],color="gray",linestyle="dashed",label=r"$\textrm{New lower bound of } \mu_{\mathrm{c}}$",zorder=10)
plt.plot([7,max_N],[1277/1870,1277/1870],color="gray",linestyle="dashed",label=r"$\textrm{Previous lower bound of } \mu_{\mathrm{c}}$",zorder=10)
plt.plot([7,max_N],[2*Kc, 2*Kc],color="red",label=r"$2K_{\mathrm{c}}$",zorder=0)
plt.scatter(ns,ss,s=10,zorder=10)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("../figs/sup_m.pdf",bbox_inches='tight')