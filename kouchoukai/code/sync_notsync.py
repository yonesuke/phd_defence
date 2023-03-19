import numpy as np
import matplotlib.pyplot as plt

n_oscillator = 50

_thetas = np.arange(0, 2*np.pi, 0.01)
phi_notsync = np.random.uniform(0, 2*np.pi, n_oscillator)
phi_sync = np.random.normal(np.pi / 3.0, 0.2, n_oscillator)

plt.figure(figsize=(16, 8))
plt.subplots_adjust(wspace=0.00)

plt.subplot(1, 2, 1)
plt.axis("off")
plt.plot(np.cos(_thetas), np.sin(_thetas), color="gray", lw=2, ls="--")
plt.scatter(np.cos(phi_notsync), np.sin(phi_notsync), color="red", s=200, edgecolors="black", zorder=10)
plt.plot([0, np.cos(phi_notsync).mean()], [0, np.sin(phi_notsync).mean()], color="black", lw=2)
plt.scatter([np.cos(phi_notsync).mean()], [np.sin(phi_notsync).mean()], color="black", s=200, edgecolors="black", zorder=10)

plt.subplot(1, 2, 2)
plt.axis("off")
plt.plot(np.cos(_thetas), np.sin(_thetas), color="gray", lw=2, ls="--")
plt.scatter(np.cos(phi_sync), np.sin(phi_sync), color="red", s=200, edgecolors="black", zorder=10)
plt.plot([0, np.cos(phi_sync).mean()], [0, np.sin(phi_sync).mean()], color="black", lw=2)
plt.scatter([np.cos(phi_sync).mean()], [np.sin(phi_sync).mean()], color="black", s=200, edgecolors="black", zorder=10)

plt.savefig("../figs/sync_notsync.pdf", bbox_inches="tight")