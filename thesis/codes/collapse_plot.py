import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

sizes=[6400,12800,25600]
plt.figure(figsize=[8,6])
plt.rcParams["font.size"]=25
a,n='00',1
cmap=plt.get_cmap("tab10")
Kc,beta,nu=2.051,0.4722,2.456

markers=['o','^','D','s']

plt.figure(figsize=[16,6])
plt.subplot(1,2,1)
plt.title("(a)", loc="left")
plt.xlabel("$K$")
plt.ylabel("$r_{N}(K)$")
for i,size in enumerate(sizes):
    Ks,rs,stds=np.loadtxt(f'data/a-{a}_n-{n}_size-{size}.txt',unpack=True)
    # plt.scatter((Ks-Kc)*size**(1/nu),rs*size**(beta/nu),label=f'$N={size}$',facecolors='none',edgecolors=cmap(i),marker=markers[i],s=10)
    plt.scatter(Ks,rs,label=f'$N={size}$',facecolors='none',edgecolors=cmap(i),marker=markers[i],s=10)
plt.legend(markerscale=2)

plt.subplot(1,2,2)
plt.title("(b)", loc="left")
plt.xlabel(r'$(K-K_{\mathrm{c}})N^{1/\bar{\nu}}$')
plt.ylabel(r'$r_{N}(K)N^{\beta/\bar{\nu}}$')
for i,size in enumerate(sizes):
    Ks,rs,stds=np.loadtxt(f'data/a-{a}_n-{n}_size-{size}.txt',unpack=True)
    plt.scatter((Ks-Kc)*size**(1/nu),rs*size**(beta/nu),label=f'$N={size}$',facecolors='none',edgecolors=cmap(i),marker=markers[i],s=10)
plt.legend(markerscale=2)

plt.savefig("../figs/collapse_plot.pdf", bbox_inches="tight")