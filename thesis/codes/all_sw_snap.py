from utils import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

K=4.5
t_max,dt=500,0.1
ts=np.arange(0,t_max,dt)
size=6400

#ネットワークを決める
k_nearest,p=6,0.2
ws=nx.watts_strogatz_graph(size,k_nearest,p,seed=0)
ws=nx.to_scipy_sparse_matrix(ws)

omegas=np.random.normal(0,np.sqrt(2),size)

#all_to_all
args=(omegas,K)
thetas_all=2*pi*np.random.rand(size)
_,thetas_all=rungekutta(kuramoto,thetas_all,args,t_max,dt)

phi_all = np.angle(np.mean(np.exp(1j*thetas_all)))
thetas_sw = np.mod(thetas_all, 2*np.pi)
thetas_all -= (np.pi + phi_all)

#small world
args=(omegas,ws,k_nearest,K)
thetas_sw=2*pi*np.random.rand(size)
_,thetas_sw=rungekutta(kuramoto_sparse,thetas_sw,args,t_max,dt)

phi_sw = np.angle(np.mean(np.exp(1j*thetas_sw)))
thetas_sw = np.mod(thetas_sw, 2*np.pi)
thetas_sw -= (np.pi + phi_sw)

plt.figure(figsize=[16,6])
plt.rcParams["font.size"]=21

plt.subplot(1,2,1)
plt.title(r'$\mathrm{(a)}$',loc='left')
# plt.xlim(0,2*pi)
plt.xlim(-pi, pi)
plt.ylim(-6,6)
plt.xlabel(r'$\theta_{i}$')
plt.ylabel(r'$\omega_{i}$')
plt.scatter(np.mod(thetas_all, 2*np.pi)-np.pi,omegas,s=5)

plt.subplot(1,2,2)
plt.title(r'$\mathrm{(b)}$',loc='left')
# plt.xlim(0,2*pi)
plt.xlim(-pi, pi)
plt.ylim(-6,6)
plt.xlabel(r'$\theta_{i}$')
plt.ylabel(r'$\omega_{i}$')
plt.scatter(np.mod(thetas_sw, 2*np.pi)-np.pi,omegas,s=5)

plt.savefig("../figs/all_sw_snap.pdf",bbox_inches='tight')