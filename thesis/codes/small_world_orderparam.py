import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

sizes=[1600,3200,6400,12800,25600]
plt.figure(figsize=[16, 6])
plt.rcParams["font.size"]=25
plt.subplots_adjust(wspace=0.03)
styles=['solid','dotted','dashed','dashdot',(0, (1, 1))]
for i,a in enumerate(['00','-02']):
    n=1
    plt.subplot(1,2,i+1)
    plt.xlabel('$K$')
    if i==0:
        plt.ylabel('$r_{N}(K)$')
    if i==1:
        plt.tick_params(labelleft=False)
    for j,size in enumerate(sizes):
        Ks,rs,stds=np.loadtxt(f'data/a-{a}_n-{n}_size-{size}.txt',unpack=True)
        plt.ylim(0,1)
        plt.errorbar(Ks,rs,yerr=stds,label=f'$N={size}$',linestyle=styles[j])
    title='$\mathrm{(a)}$' if i==0 else '$\mathrm{(b)}$'
    plt.title(title,loc='left')
    plt.legend()
# plt.savefig('arxiv/bif-fss-all.eps',bbox_inches='tight')
plt.savefig("../figs/small_world_orderparam.pdf", bbox_inches="tight")