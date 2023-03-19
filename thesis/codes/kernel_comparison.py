import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def rbf_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

def matern_kernel(x, y, sigma, nu):
    d = np.linalg.norm(x-y)
    if nu == 1/2:
        return sigma**2 * np.exp(-d)
    elif nu == 3/2:
        return sigma**2 * (1 + np.sqrt(3)*d) * np.exp(-np.sqrt(3)*d)
    elif nu == 5/2:
        return sigma**2 * (1 + np.sqrt(5)*d + 5/3*d**2) * np.exp(-np.sqrt(5)*d)
    else:
        raise ValueError("nu must be 1/2, 3/2, or 5/2")

def periodic_kernel(x, y, sigma):
    return np.exp(-2*np.sin(np.pi*np.linalg.norm(x-y))**2/sigma**2)

def cov_mat_fn(xs, kernel_fn):
    return np.array([[kernel_fn(x, y) for x in xs] for y in xs])

def sample(xs, kernel_fn, n):
    cov = cov_mat_fn(xs, kernel_fn) + 1e-6*np.eye(len(xs))
    L = np.linalg.cholesky(cov)
    z = np.random.randn(len(xs), n)
    return L @ z

n_path = 20

plt.figure(figsize=(16, 18))
plt.rcParams["font.size"] = 25

# RBF kernel
rbf_sigma = 1.0
xs = np.arange(-1.0, 1.0, 0.01)
cov_mat = cov_mat_fn(xs, lambda x, y: rbf_kernel(x, y, rbf_sigma))
ys = sample(xs, lambda x, y: rbf_kernel(x, y, rbf_sigma), n_path).T

plt.subplot(3,2,1)
plt.title("RBF kernel")
plt.xlabel("$x$")
plt.ylabel(r"$\tilde{x}$")
plt.matshow(cov_mat, fignum=0, extent=(-1, 1, -1, 1))
plt.gca().xaxis.tick_bottom()
plt.colorbar()
# plt.plot(xs, ys.T, lw=2)

plt.subplot(3,2,2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.xlim(-1, 1)
for y in ys:
    plt.plot(xs, y, lw=2)

# Matern kernel
matern_sigma = 1.0
nu = 1/2
xs = np.arange(-1.0, 1.0, 0.01)
cov_mat = cov_mat_fn(xs, lambda x, y: matern_kernel(x, y, matern_sigma, nu))
ys = sample(xs, lambda x, y: matern_kernel(x, y, matern_sigma, nu), n_path).T

plt.subplot(3,2,3)
plt.title(r"Matérn kernel ($\nu=1/2$)")
plt.xlabel("$x$")
plt.ylabel(r"$\tilde{x}$")
plt.matshow(cov_mat, fignum=0, extent=(-1, 1, -1, 1))
plt.gca().xaxis.tick_bottom()
plt.colorbar()

plt.subplot(3,2,4)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.xlim(-1, 1)
for y in ys:
    plt.plot(xs, y, lw=2)

# Periodic kernel
periodic_sigma = 1.0
xs = np.arange(-1.0, 1.0, 0.01)
cov_mat = cov_mat_fn(xs, lambda x, y: periodic_kernel(x, y, periodic_sigma))
ys = sample(xs, lambda x, y: periodic_kernel(x, y, periodic_sigma), n_path).T

plt.subplot(3,2,5)
plt.title("Periodic kernel")
plt.xlabel("$x$")
plt.ylabel(r"$\tilde{x}$")
plt.matshow(cov_mat, fignum=0, extent=(-1, 1, -1, 1))
plt.gca().xaxis.tick_bottom()
plt.colorbar()

plt.subplot(3,2,6)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.xlim(-1, 1)
for y in ys:
    plt.plot(xs, y, lw=2)

plt.tight_layout()

plt.savefig("../figs/kernel_comparison.pdf", bbox_inches="tight")