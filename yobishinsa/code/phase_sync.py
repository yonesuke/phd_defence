import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

_thetas = np.linspace(0, 2 * np.pi, 1000)

phi = 0.0

plt.figure(figsize=(6, 6))
plt.axis("off")
plt.plot(np.cos(_thetas), np.sin(_thetas), color="gray", lw=2, ls="dashed")
plt.scatter(np.cos(phi), np.sin(phi), color="tab:blue", s=200, edgecolors="black", zorder=10)
plt.savefig("../figs/phase_sync1.pdf", bbox_inches="tight", pad_inches=0.0)

phis = np.random.normal(loc=phi, scale=0.1, size=10)

plt.figure(figsize=(6, 6))
plt.axis("off")
plt.plot(np.cos(_thetas), np.sin(_thetas), color="gray", lw=2, ls="dashed")
plt.scatter(np.cos(phis), np.sin(phis), color="tab:blue", s=200, edgecolors="black", zorder=10)
plt.savefig("../figs/phase_sync2.pdf", bbox_inches="tight", pad_inches=0.0)

phis = np.random.uniform(low=0, high=2 * np.pi, size=10)

plt.figure(figsize=(6, 6))
plt.axis("off")
plt.plot(np.cos(_thetas), np.sin(_thetas), color="gray", lw=2, ls="dashed")
plt.scatter(np.cos(phis), np.sin(phis), color="tab:blue", s=200, edgecolors="black", zorder=10)
plt.savefig("../figs/phase_sync3.pdf", bbox_inches="tight", pad_inches=0.0)