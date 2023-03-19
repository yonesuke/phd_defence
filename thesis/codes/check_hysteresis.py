import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

plt.figure(figsize=[16,12])
plt.rcParams["font.size"]=20

plt.subplot(2,2,1)
plt.title('$(\mathrm{a})~a=0$',loc='left')
plt.xlim(1.8,2.4)
plt.ylabel('$r_N(K)$')
paths=['data/a-00_n-1_size-25600.txt','data/hysteresis/a-00_n-1_size-25600.txt','data/hysteresis/back_a-00_n-1_size-25600.txt']
labels=['$\mathrm{random}$','$r_{N}^{(\mathrm{forward})}(K)$','$r_{N}^{(\mathrm{backward})}(K)$']
styles=['solid','dashed']
Ks,r1s,std1s=np.loadtxt(paths[1],unpack=True)
plt.errorbar(Ks,r1s,yerr=std1s,label=labels[1],linestyle=styles[0],color='black')
Ks,r2s,std2s=np.loadtxt(paths[2],unpack=True)
Ks,r2s,std2s=Ks[::-1],r2s[::-1],std2s[::-1]
plt.errorbar(Ks,r2s,yerr=std2s,label=labels[2],linestyle=styles[1],color='red')
plt.legend()

plt.subplot(2,2,2)
plt.title('$(\mathrm{b})~a=-0.2$',loc='left')
plt.xlim(2.0,2.6)
plt.ylabel('$r_N(K)$')
paths=['data/a--02_n-1_size-25600.txt','data/hysteresis/a--02_n-1_size-25600.txt','data/hysteresis/back_a--02_n-1_size-25600.txt']
labels=['$\mathrm{random}$','$r_{N}^{(\mathrm{forward})}(K)$','$r_{N}^{(\mathrm{backward})}(K)$']
styles=['solid','dashed']
Ks,r1s,std1s=np.loadtxt(paths[1],unpack=True)
plt.errorbar(Ks,r1s,yerr=std1s,label=labels[1],linestyle=styles[0],color='black')
Ks,r2s,std2s=np.loadtxt(paths[2],unpack=True)
Ks,r2s,std2s=Ks[::-1],r2s[::-1],std2s[::-1]
plt.errorbar(Ks,r2s,yerr=std2s,label=labels[2],linestyle=styles[1],color='red')
plt.legend()

ax1=plt.subplot(2,2,3)
ax1.set_title('$(\mathrm{c})~a=0.5$',loc='left')
left, bottom, width, height = [0.62, 0.26, 0.3,0.3]
#ax2 = fig.add_axes([left, bottom, width, height])
#ax2=inset_axes(ax1,width="40%",height="40%",loc="lower right",)
ax2=ax1.inset_axes([0.62, 0.2, 0.35, 0.35])

#ax1
ax1.set_xlim(1.5,1.9)
#ax1.xlim(1.5,1.9)
#ax1.ylim(0,1)
ax1.set_ylim(0,1)
ax1.set_xlabel('$K$')
ax1.set_ylabel('$r_N(K)$')
paths=['data/a-05_n-1_size-25600.txt','data/hysteresis/a-05_n-1_size-25600.txt','data/hysteresis/back_a-05_n-1_size-25600.txt']
labels=['$\mathrm{random}$','$r_{N}^{(\mathrm{forward})}(K)$','$r_{N}^{(\mathrm{backward})}(K)$']
styles=['solid','dashed']
counter=0
#forward, backwardそれぞれ
Ks,r1s,std1s=np.loadtxt(paths[1],unpack=True)
ax1.errorbar(Ks,r1s,yerr=std1s,label=labels[1],linestyle=styles[0],color='black')
Ks,r2s,std2s=np.loadtxt(paths[2],unpack=True)
Ks,r2s,std2s=Ks[::-1],r2s[::-1],std2s[::-1]
ax1.errorbar(Ks,r2s,yerr=std2s,label=labels[2],linestyle=styles[1],color='red')
ax1.legend()

#ax2
ax2.set_xlim(1.5,1.9)
ax2.set_ylim(0,0.06)
ax2.plot(Ks,np.abs(r2s-r1s),color='blue')
#xticklabels=ax2.get_xticklabels()
#yticklabels=ax2.get_yticklabels()
ax2.set_xticks([1.6,1.8])
ax2.set_xticklabels(['$1.6$','$1.8$'],fontsize=15)
ax2.set_yticklabels(['$0$','$0.05$'],fontsize=15)
ax2.set_xlabel('$K$',fontsize=15)
ax2.xaxis.set_label_coords(0.5,-0.15)

plt.savefig("../figs/check_hysteresis.pdf",bbox_inches="tight")
