import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
from tqdm.auto import tqdm

n_oscillators = 50000
np.random.seed(seed=39)

def forward(thetas, omegas, K):
    coss = np.cos(thetas)
    sins = np.sin(thetas)
    rx = np.mean(coss)
    ry = np.mean(sins)
    return omegas + K * (ry * coss - rx * sins)

def runge_kutta(func, x, dt):
    k1 = func(x)
    k2 = func(x + 0.5 * dt * k1)
    k3 = func(x + 0.5 * dt * k2)
    k4 = func(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

omegas = np.random.standard_cauchy(n_oscillators)
thetas = np.random.uniform(0, 2 * np.pi, n_oscillators)

dt, t_max = 0.01, 50
K = 5.0

for t in tqdm(np.arange(0, t_max, dt)):
    thetas = runge_kutta(lambda x: forward(x, omegas, K), thetas, dt)

r = np.sqrt(np.mean(np.cos(thetas))**2 + np.mean(np.sin(thetas))**2)
phi = np.angle(np.mean(np.exp(1j*thetas)))
thetas = np.mod(thetas, 2*np.pi)
thetas -= (np.pi + phi)

plt.figure(figsize=(8, 6))
plt.rc("font", size=20)

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\omega$")
plt.xlim(-np.pi, np.pi)
plt.ylim(-10, 10)
plt.scatter(np.mod(thetas, 2*np.pi)-np.pi, omegas, s=0.1, alpha=0.5)
plt.plot([-np.pi, np.pi], [K*r, K*r], lw=1, c="tab:orange")
plt.plot([-np.pi, np.pi], [-K*r, -K*r], lw=1, c="tab:orange", label=r"$|\omega|=Kr$")
phis = np.arange(-0.5*np.pi, 0.5*np.pi, 0.01)
plt.plot(phis, K*r*np.sin(phis), lw=3, c="tab:orange", zorder=0, ls="dashed", label=r"$\omega=Kr\sin\theta,\ |\theta|<\pi/2$")

plt.legend()

plt.savefig("../figs/kuramoto_scatter.pdf", bbox_inches="tight")