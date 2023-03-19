import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

params = [(3, 1), (5, 1), (7, 2)]
_thetas = np.linspace(0, 2 * np.pi, 1000)
plt.figure(figsize=(24, 8))
plt.subplots_adjust(wspace=0.00)
plt.rcParams["font.size"] = 30
for i, (N, p) in enumerate(params):
    plt.subplot(1, 3, i + 1)
    plt.axis("off")
    plt.title(f"$N = {N}, p = {p}$")
    plt.plot(np.cos(_thetas), np.sin(_thetas), color="gray", lw=2, ls="dashed")
    thetas = np.arange(N) * 2 * np.pi * p / N
    plt.scatter(np.cos(thetas), np.sin(thetas), marker="o", color="red", zorder=10, s=200, edgecolors="k")
plt.savefig("../figs/twisted_state.pdf", bbox_inches="tight")
