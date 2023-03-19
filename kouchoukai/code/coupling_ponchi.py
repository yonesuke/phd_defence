import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

n_oscillator = 3
_thetas = np.arange(0, 2*np.pi, 0.01)

ylabels = [r"$\Gamma_{11}(\theta)$", r"$\Gamma_{12}(\theta)$", r"$\Gamma_{21}(\theta)$", r"$\Gamma_{23}(\theta)$", r"$\Gamma_{31}(\theta)$", r"$\Gamma_{32}(\theta)$"]

counter = -1
plt.figure(figsize=(15, 10))
plt.rcParams["font.size"] = 25
for i in range(n_oscillator):
    for j in range(n_oscillator):
        if i == j:
            continue
        counter += 1
        idx = i * n_oscillator + j + 1
        plt.subplot(n_oscillator, n_oscillator, idx)
        plt.ylabel(ylabels[counter], fontsize=30)
        plt.xticks([0, np.pi, 2*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
        plt.yticks([])
        plt.xlim(0, 2*np.pi)
        plt.plot(_thetas, np.sin(_thetas), color="red", lw=2)

plt.savefig("../figs/gpr_coupling_result_ponchi.pdf", bbox_inches="tight")