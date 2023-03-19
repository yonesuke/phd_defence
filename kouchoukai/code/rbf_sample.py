import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def _rbf_fn(x):
    return np.exp(-x**2)

rbf_fn = lambda x, y: _rbf_fn(x-y)

def kern_mat(xs, kern_fn):
    mat = []
    for x in xs:
        arr = kern_fn(xs,x)
        mat.append(arr)
    return np.array(mat)

x_min, x_max = -5.0, 5.0
xs = np.arange(x_min, x_max, 0.01)
mean = 0*xs
sigma = kern_mat(xs, rbf_fn)

n_path = 30
yss = np.random.multivariate_normal(mean, sigma, n_path)

plt.figure(figsize=[12,6])
plt.rcParams["font.size"] = 20

plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.xlim(x_min, x_max)
plt.plot([x_min, x_max], [0, 0], color="gray", ls="dashed", lw=2, zorder=0)
for ys in yss:
    plt.plot(xs, ys, lw=1)
    
plt.savefig("../figs/rbf_sample.pdf", bbox_inches="tight")