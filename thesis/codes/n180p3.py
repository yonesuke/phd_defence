import matplotlib.pyplot as plt
import numpy as np
import math
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def f(x):
    return -np.cos(x)*(1-np.cos(x))

def critical_index(N):
    xs = np.array([2*np.pi*l/N for l in range(1,N)])
    ys = f(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    return kc

def additional_one(N,p):
    m = math.gcd(N, p)
    nt, pt = int(N/m), int(p/m)
    xs = np.array([2*np.pi*l/nt for l in range(1,nt)])
    ys = f(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    additional = 2*math.ceil(-m*sk[kc-2]/ys[kc-1]-1)
    return additional

plt.figure(figsize=[16,6])
plt.rcParams['font.size']=25

m = 3
nt = 60
N, p = nt*m, m

plt.xlim(0,m*2*np.pi)
_xs = np.arange(0, m*2*np.pi, 0.01)
_ys = f(_xs)
#plt.plot(_xs, _ys, label=r"$-\cos\theta+\cos^{2}\theta$", color="gray", linewidth=1, zorder=1)
plt.plot(_xs, _ys, color="gray", linewidth=1, zorder=1)
plt.plot([0, m*2*np.pi], [0, 0], linestyle="dashed", color="gray", zorder=0)

kc = critical_index(nt)
colors = ["white" for _ in range(1,N)]
markers = ["o" for _ in range(1,N)]
edgecolors = ["black" for _ in range(1,N)]
# kcまでを塗る
for i in range(m):
    for j in range(kc-1):
        colors[i*nt+j] = "royalblue"
        colors[i*nt+nt-kc+j] = "royalblue"
# ntの上を塗る
for i in range(1,m):
    colors[i*nt-1] = "red"
    markers[i*nt-1] = "D"
#追加で塗る
#(N,p)=(60*3,3)のとき追加で4つ塗る
step3color = "mediumspringgreen"
colors[kc-1] = step3color
colors[N-kc-1] = step3color
colors[nt-kc-1] = step3color
colors[N-nt+kc-1] = step3color

markers[kc-1] = "s"
markers[N-kc-1] = "s"
markers[nt-kc-1] = "s"
markers[N-nt+kc-1] = "s"
markers[nt+kc-1] = "s"
markers[2*nt-kc-1] = "s"

edgecolors[kc-1] = step3color
edgecolors[N-kc-1] = step3color
edgecolors[nt-kc-1] = step3color
edgecolors[N-nt+kc-1] = step3color
edgecolors[nt+kc-1] = step3color
edgecolors[2*nt-kc-1] = step3color

xs = [2*np.pi*l/nt for l in range(1,N)]
ys = f(xs)
# plt.scatter(xs, ys, zorder=10, markers = markers, color=colors, edgecolors="black", linewidths=1.0, s=50)
for i in range(N-1):
    plt.scatter(xs[i], ys[i], zorder=10, marker = markers[i], color=colors[i], edgecolors=edgecolors[i], linewidths=1.0, s=50)
# plt.scatter(xs, ys, zorder=10, color=colors, linewidths=1.0, s=50)

plt.xlabel(r"$2\pi pl/N$")
plt.ylabel(r"$b^{(N,p)}_{l}$")
xlocs = [0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi,
    2.5*np.pi,3*np.pi,3.5*np.pi,4*np.pi,
    4.5*np.pi,5*np.pi,5.5*np.pi,6*np.pi]
xlabs = [
    "$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", 
    r"$2\pi$", r"$\frac{5\pi}{2}$", r"$3\pi$", r"$\frac{7\pi}{2}$",
    r"$4\pi$", r"$\frac{9\pi}{2}$", r"$5\pi$", r"$\frac{11\pi}{2}$",
    r"$6\pi$"
]
plt.xticks(xlocs, xlabs)
plt.tight_layout()

plt.savefig("../figs/N180p3.pdf", bbox_inches="tight")