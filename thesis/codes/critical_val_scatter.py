import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

df=pd.read_csv('data/critical_exponent_raw_data.csv')

a,n='00',1
sizes=[1600,3200,6400]
markers=['o','^','D']
rescale_min,rescale_max=-20,10
plt.figure(figsize=[16,6])
plt.subplots_adjust(wspace=0.03)
plt.rcParams['font.size']=25

plt.subplot(1,2,1)
plt.title('$\mathrm{(a)}$',loc='left')
plt.ylim(2,2.3)
for i,size in enumerate(sizes):
    c1s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c1']
    c2s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c2']
    Kcss=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['Kc']
    c1s,c2s,Kcss=np.array(c1s),np.array(c2s),np.array(Kcss)
    c1s,c2s,Kcss=c1s[-c2s<0.25],c2s[-c2s<0.25],Kcss[-c2s<0.25]
    betas=-c2s/c1s
    nus=1/c1s
    plt.scatter(betas,Kcss,s=20,alpha=0.5,label='$N_{\min}='+f'{size}$',marker=markers[i])
plt.xlabel('$\\beta$')
plt.ylabel('$K_{\mathrm{c}}$')
plt.legend(markerscale=2)

plt.subplot(1,2,2)
plt.title('$\mathrm{(b)}$',loc='left')
plt.ylim(2,2.3)
plt.tick_params(labelleft=False)
for i,size in enumerate(sizes):
    c1s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c1']
    c2s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c2']
    Kcss=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['Kc']
    c1s,c2s,Kcss=np.array(c1s),np.array(c2s),np.array(Kcss)
    c1s,c2s,Kcss=c1s[-c2s<0.25],c2s[-c2s<0.25],Kcss[-c2s<0.25]
    betas=-c2s/c1s
    nus=1/c1s
    plt.scatter(nus,Kcss,s=20,alpha=0.5,label='$N_{\min}='+f'{size}$',marker=markers[i])
plt.xlabel('$\\bar{\\nu}$')
plt.legend(markerscale=2)
# plt.savefig('arxiv/scatter.eps',bbox_inches='tight')
plt.savefig("../figs/critical_val_scatter.pdf", bbox_inches="tight")