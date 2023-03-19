import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
import pandas as pd

df=pd.read_csv('data/critical_exponent_raw_data.csv')

#xs,ys,ws: np.array or list
def ab_vavb(xs,ys,ws):
    xs,ys,ws=np.array(xs),np.array(ys),np.array(ws)
    delta=ws.sum()*(ws*xs**2).sum()-((ws*xs).sum())**2
    a=((ws*ys).sum()*(ws*xs**2).sum()-(ws*xs).sum()*(ws*xs*ys).sum())/delta
    b=(ws.sum()*(ws*xs*ys).sum()-(ws*xs).sum()*(ws*ys).sum())/delta
    va=(ws*xs**2).sum()/delta
    vb=ws.sum()/delta
    return [a,b,va,vb]

a=0.0
cmap=plt.get_cmap("tab10")
ns=['$n=1$','$n=2$','$n=3$','$n=\infty$']
sizes=[1600,3200,6400]
rescale_min,rescale_max=-20,10
xs=[1/size for size in sizes]
y1ss,w1ss=[],[]
betas,slope1s,v_betas=[],[],[]
for n in [1,2,3,4]:
    y1s,w1s,y2s,w2s,y3s,w3s=[],[],[],[],[],[]
    for size in sizes:
        c1s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c1']
        c2s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c2']
        Kcss=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['Kc']
        c1s,c2s,Kcss=np.array(c1s),np.array(c2s),np.array(Kcss)
        c1s,c2s,Kcss=c1s[-c2s<0.25],c2s[-c2s<0.25],Kcss[-c2s<0.25]
        _betas,nus=-c2s/c1s,1/c1s
        y1s.append(_betas.mean())
        w1s.append(1/(_betas.std()**2))
    y1ss.append(y1s)
    w1ss.append(w1s)
    beta,slope1,v_beta,_=ab_vavb(xs,y1s,w1s)
    betas.append(beta)
    slope1s.append(slope1)
    v_betas.append(v_beta)

#図作るpart

plt.figure(figsize=[16, 6])
plt.rcParams['font.size']=20
plt.subplots_adjust(hspace=0.21)
markers=['o','^','D','s']
linestyles=["solid", "dashed", "dashdot", "dotted"]

plt.subplot(1,2,1)
plt.title('$\mathrm{(a)}\ a=0$',loc='left')
xrange=np.arange(0,1/1380,1/100000)
plt.xticks([0,*xs],['$1/\infty$','$1/1600$','$1/3200$','$1/6400$'])
plt.yticks([0.2,0.3,0.4,0.5,0.6],['$0.2$','$0.3$','$0.4$','$0.5$','$0.6$'])
# plt.tick_params(labelbottom=False,)
plt.xlim(0,1/1400)
plt.ylim(0.2,0.62)
plt.xlabel('$1/N_{\min}$')
plt.ylabel(r'$\beta$')

for i in range(4):
    y1s,w1s,beta,slope1,v_beta=y1ss[i],w1ss[i],betas[i],slope1s[i],v_betas[i]
    s1s=1/np.sqrt(np.array(w1s))
    plt.errorbar(xs,y1s,s1s,fmt=markers[i],capsize=8,ecolor=cmap(i),markeredgecolor=cmap(i),color='w',label=ns[i])
    plt.plot(xrange,slope1*xrange+beta,linestyle=linestyles[i],color=cmap(i))
    e=plt.errorbar([0],[beta],[np.sqrt(v_beta)],fmt=markers[i],capsize=8,ecolor=cmap(i),markeredgecolor=cmap(i),color='w',clip_on=False,zorder=10,lw=3)
    for b in e[1]:
        b.set_clip_on(False)
    for b in e[2]:
        b.set_clip_on(False)
plt.legend(fontsize=20)


a=-0.2
sizes=[1600,3200,6400]
rescale_min,rescale_max=-20,10
xs=[1/size for size in sizes]
y1ss,w1ss=[],[]
betas,slope1s,v_betas=[],[],[]
for n in [1,2,3,4]:
    y1s,w1s,y2s,w2s,y3s,w3s=[],[],[],[],[],[]
    for size in sizes:
        c1s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c1']
        c2s=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['c2']
        Kcss=df.query(f'size=={size} and a=={a} and n=={n} and rescale_min=={rescale_min} and rescale_max=={rescale_max}')['Kc']
        c1s,c2s,Kcss=np.array(c1s),np.array(c2s),np.array(Kcss)
        c1s,c2s,Kcss=c1s[-c2s<0.25],c2s[-c2s<0.25],Kcss[-c2s<0.25]
        _betas,nus=-c2s/c1s,1/c1s
        y1s.append(_betas.mean())
        w1s.append(1/(_betas.std()**2))
    y1ss.append(y1s)
    w1ss.append(w1s)
    beta,slope1,v_beta,_=ab_vavb(xs,y1s,w1s)
    betas.append(beta)
    slope1s.append(slope1)
    v_betas.append(v_beta)






plt.subplot(1,2,2)
plt.title('$\mathrm{(b)}\ a=-0.2$',loc='left')
xrange=np.arange(0,1/1380,1/100000)
plt.xticks([0,*xs],['$1/\infty$','$1/1600$','$1/3200$','$1/6400$'])
plt.yticks([0.2,0.3,0.4,0.5,0.6],['$0.2$','$0.3$','$0.4$','$0.5$','$0.6$'])
plt.xlim(0,1/1400)
plt.ylim(0.2,0.62)
plt.ylabel(r'$\beta$')
plt.xlabel('$1/N_{\min}$')

for i in range(4):
    y1s,w1s,beta,slope1,v_beta=y1ss[i],w1ss[i],betas[i],slope1s[i],v_betas[i]
    s1s=1/np.sqrt(np.array(w1s))
    plt.errorbar(xs,y1s,s1s,fmt=markers[i],capsize=8,ecolor=cmap(i),markeredgecolor=cmap(i),color='w',label=ns[i])
    plt.plot(xrange,slope1*xrange+beta,linestyle=linestyles[i],color=cmap(i))
    e=plt.errorbar([0],[beta],[np.sqrt(v_beta)],fmt=markers[i],capsize=8,ecolor=cmap(i),markeredgecolor=cmap(i),color='w',clip_on=False,zorder=10,lw=3)
    for b in e[1]:
        b.set_clip_on(False)
    for b in e[2]:
        b.set_clip_on(False)
plt.legend(fontsize=20)

plt.savefig("../figs/beta_N_dep.pdf",bbox_inches="tight")