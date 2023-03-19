import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def kernel(x, y, sigma):
    return np.exp(-(x-y)**2/(2*sigma**2))

xs = np.arange(-1, 1, 0.01)

xx1, xx2 = np.meshgrid(xs, xs)

ret = kernel(xx1, 0, 0.2) + kernel(0, xx2, 0.2)

fig = plt.figure(figsize=(16, 6))
plt.rcParams["font.size"] = 20
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.set_xlabel(r"$x_{1}-\tilde{x}_{1}$")
ax1.set_ylabel(r"$x_{2}-\tilde{x}_{2}$")
ax1.plot_surface(xx1, xx2, ret, cmap="jet")

ax2.plot_surface(xx1, xx2, ret, cmap="viridis")

plt.savefig("../figs/additive_kernel.pdf", bbox_inches="tight")