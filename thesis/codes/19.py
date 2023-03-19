import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def b(x):
    return -np.cos(x)*(1-np.cos(x))

def critical_index(N):
    xs = np.array([2*np.pi*l/N for l in range(1,N)])
    ys = b(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    return kc

def additional_one(N,p):
    m = math.gcd(N, p)
    nt, pt = int(N/m), int(p/m)
    xs = np.array([2*np.pi*l/nt for l in range(1,nt)])
    ys = b(xs)
    sk = np.cumsum(ys)
    kc = np.argmax(sk>=0) + 1
    additional = 2*math.ceil(-m*sk[kc-2]/ys[kc-1]-1)
    return additional

def optimal_Np(N,p):
    m = math.gcd(N, p)
    nt, pt = int(N/m), int(p/m)
    if nt < 5:
        return "there is no feasible solution"
    kc = critical_index(nt)
    xs = np.zeros(N)
    # step 1
    for i in range(m):
        for j in range(1,kc):
            xs[i*nt+j] = 1.0
            xs[i*nt+nt-kc+j] = 1.0
    # step 2
    for i in range(1,m):
        xs[i*nt] = 1.0
    # step 3
    additional = additional_one(N, p)
    # first enumerate all the possible candidates
    candidates = []
    for i in range(m):
        candidates.append(i*nt+kc)
        candidates.append((i+1)*nt-kc)
    for i in range(int(additional/2)):
        # additionally select from left to right
        xs[candidates[i]] = 1.0
        # also, don't forget to satisfy 
        xs[N-candidates[i]] = 1.0
    return xs

num = 19
plt.figure(figsize=[16,16])
for i in range(1, 4+1):
    plt.subplot(2,2,i)
    plt.title(r"$N={}$, $p={}$".format(num*i, i), loc="left", fontsize=25)
    xs =optimal_Np(num*i, i)
    G = nx.generators.classic.circulant_graph(num*i, [i for i in range(len(xs)) if xs[i] > 0])
    nx.draw_circular(G, with_labels=True, node_color="lightgreen")
plt.savefig("../figs/19.pdf", bbox_inches="tight", transparent=True)
