import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

def cov_mat(xs, sigma):
    return np.array([[kernel(x, y, sigma) for x in xs] for y in xs])

def sample(xs, sigma, n):
    cov = cov_mat(xs, sigma) + 1e-6*np.eye(len(xs))
    L = np.linalg.cholesky(cov)
    z = np.random.randn(len(xs), n)
    return L @ z

xs = np.arange(-4.5, 4.5, 0.01)
ys = sample(xs, 1.0, 10).T

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 20

plt.xlim(-4., 4.)
plt.xlabel("$x$")
plt.ylabel("$y$")
for y in ys:
    plt.plot(xs, y, lw=2)

plt.savefig("../figs/rbf_samples.pdf", bbox_inches="tight")