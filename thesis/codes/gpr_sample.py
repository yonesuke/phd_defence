import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

def cov_mat(xs, sigma):
    return np.array([[kernel(x, y, sigma) for x in xs] for y in xs])

n_data = 30

test_fn = lambda x: np.sin(2*np.pi*x) + x ** 2 / 5
xs_data = np.random.uniform(-3, 3, n_data)
ys_data = test_fn(xs_data)
data = (xs_data, ys_data)

xs_train = np.arange(-4., 4., 0.01)

def regression(data, xs, sigma):
    xs_data, ys_data = data
    k_xX = np.array([[kernel(x, y, sigma) for y in xs_data] for x in xs])
    k_XX = cov_mat(xs_data, sigma) + 1e-6*np.eye(len(xs_data))
    k_XX_inv = np.linalg.inv(k_XX)
    mu = k_xX @ (k_XX_inv @ ys_data)
    var1 = np.array([kernel(x, x, sigma) for x in xs])
    tmp = k_XX_inv @ k_xX.T
    var2 = np.array([np.dot(k_xX[i, :],  tmp[:, i]) for i in range(len(xs))])
    var = var1 - var2
    return mu, var

mu, var = regression(data, xs_train, 0.5)

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 20
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.xlim(-4, 4)

plt.scatter(xs_data, ys_data, marker="o", color="k", zorder=10)
plt.plot(xs_train, test_fn(xs_train), lw=2, label="true function")
plt.plot(xs_train, mu, lw=2, label="posterior mean function")


plt.fill_between(xs_train, mu - 1.96*np.sqrt(var), mu + 1.96*np.sqrt(var), alpha=0.2, label=r"95\% confidence interval", color="tab:orange")
plt.legend()

plt.savefig("../figs/gpr_sample.pdf", bbox_inches="tight")