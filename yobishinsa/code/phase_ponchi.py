import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

_thetas = np.linspace(0, 2 * np.pi, 300)
circle_x, circle_y = np.cos(_thetas), np.sin(_thetas)
phi_i, phi_j = np.pi / 3.0, np.pi * 1.25
r1, r2 = 0.2, 0.22

plt.figure(figsize=(6, 6))
plt.text(0.95, 0.95, r"$\mathbb{S}^{1}$", fontsize=40, va="center", ha="right")
plt.axis("off")

plt.plot([0, 1], [0, 0], c="gray", lw=1.5, ls="dashed")
plt.plot([0, np.cos(phi_i)], [0, np.sin(phi_i)], c="gray", lw=1.5, ls="dashed")
_phis = np.linspace(0, phi_i, 300)
plt.plot(r1 * np.cos(_phis), r1 * np.sin(_phis), c="gray", lw=1.5)
plt.text(r2 * np.cos(0.5 * phi_i), r2 * np.sin(0.5 * phi_i), r"$\theta_{i}$", fontsize=40, va="center", ha="left")

plt.plot(circle_x, circle_y, c="gray", lw=5, alpha=0.8)
plt.scatter(np.cos(phi_i), np.sin(phi_i), c="tab:blue", s=200, edgecolors="black", lw=1.5, zorder=10)

plt.savefig("phase1.pdf", bbox_inches="tight", pad_inches=0.0, transparent=True)

plt.figure(figsize=(6, 6))
plt.text(0.95, 0.95, r"$\mathbb{S}^{1}$", fontsize=40, va="center", ha="right")
plt.axis("off")

plt.plot([0, 1], [0, 0], c="gray", lw=1.5, ls="dashed")
plt.plot([0, np.cos(phi_j)], [0, np.sin(phi_j)], c="gray", lw=1.5, ls="dashed")
_phis = np.linspace(0, phi_j, 300)
plt.plot(r1 * np.cos(_phis), r1 * np.sin(_phis), c="gray", lw=1.5)
plt.text(r2 * np.cos(0.5 * phi_j), r2 * np.sin(0.5 * phi_j), r"$\theta_{j}$", fontsize=40, va="bottom", ha="center")

plt.plot(circle_x, circle_y, c="gray", lw=5, alpha=0.8)
plt.scatter(np.cos(phi_j), np.sin(phi_j), c="tab:green", s=200, edgecolors="black", lw=1.5, zorder=10)

plt.savefig("phase2.pdf", bbox_inches="tight", pad_inches=0.0, transparent=True)