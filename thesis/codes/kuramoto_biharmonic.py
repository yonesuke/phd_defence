import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

Ks = np.arange(0.0, 2.0, 0.1)
Ks = np.concatenate([Ks, np.arange(2.0, 2.5, 0.01)])
Ks = np.concatenate([Ks, np.arange(2.5, 4.0, 0.01)])
fname = "kuramoto_100000_-02.npy"
rss = np.loadtxt(fname)[:,8000:].mean(axis=1)

plt.figure(figsize=(16, 6))
plt.rcParams["font.size"] = 20

plt.subplot(1, 2, 1)
plt.xlabel("$K$")
plt.ylabel("$r$")
xs = np.arange(1.9, 2.5, 0.01)
ys = 1.5 * (xs - 2.0)
plt.xlim(0.0, 4.0)
plt.ylim(-0.01, 1.0)
plt.plot(Ks, rss, label="$N=10^5$", lw=2)
plt.plot(xs, ys, ls="dashed", lw=2, label=r"$r\sim\frac{2(1-a)}{K_{\mathrm{c}}^{3}Ca}(K-K_{\mathrm{c}})$", color="tab:green")
plt.legend(loc="upper left")

plt.subplot(1, 2, 2)

Ks = np.arange(0.0, 4.0, 0.01)
fname_forward = "kuramoto_1000000_05_forward.npy"
fname_backward = "kuramoto_1000000_05_backward.npy"
rs_forward = np.load(fname_forward)[:,40000:].mean(axis=1)
rs_backward = np.load(fname_backward)[:,40000:].mean(axis=1)

plt.xlabel("$K$")
plt.ylabel("$r$")
plt.xlim(0.0, 4.0)
plt.ylim(-0.01, 1.0)
xs = np.arange(1.5, 2.1, 0.01)
ys = -0.25 * (xs - 2.0)
plt.plot(Ks, rs_forward, label="$N=10^6$, forward", lw=2)
plt.plot(Ks, rs_backward, label="$N=10^6$, backward", lw=2)
# plt.plot(xs, ys, ls="dashed", lw=2, label=r"$r\sim\frac{2(1-a)}{K_{\mathrm{c}}^{3}Ca}(K-K_{\mathrm{c}})$", c="tab:green")
plt.legend(loc="upper left")

plt.savefig("../figs/kuramoto_biharmonic.pdf", bbox_inches="tight")