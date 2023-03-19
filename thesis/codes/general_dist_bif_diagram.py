import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

from tqdm.auto import tqdm

def general_lorentz(x, gamma, n):
    return n * np.sin(0.5 * np.pi / n) * gamma ** (2*n - 1) / np.pi / (x ** (2*n) + gamma ** (2*n))

def inf_dist(x, gamma):
    return 0.5 / gamma * (np.abs(x) < gamma)

gamma = 1.0
dt = 0.0001
thetas = np.arange(-0.5*np.pi, 0.5*np.pi, dt)

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 20
plt.xlabel("$K$")
plt.ylabel("$r$")
plt.xlim(0, 5)
plt.ylim(-0.01, 1.0)
for n in tqdm([1,2,3]):
    Kc = 2.0 / np.pi / general_lorentz(0.0, gamma, n)
    # Ks = [0, Kc]
    # rs = [0, 0]
    Ks = np.arange(Kc+0.001, Kc+0.5, 0.001)
    Ks = np.concatenate([Ks, np.arange(Kc+0.5, 5.001, 0.1)])
    rs = []
    for K in tqdm(Ks):
        r = 1.0
        for _ in range(100):
            integral_fn = lambda theta: np.cos(theta)**2*general_lorentz(K*r*np.sin(theta), gamma, n)
            integral_val = integral_fn(thetas).sum() * dt
            r = K * r * integral_val
        rs.append(r)
    Ks = np.concatenate([[0, Kc], Ks])
    rs = np.concatenate([[0, 0], rs])
    if n==1:
        Ks = np.arange(0, 6, 0.01)
        rs = np.array([np.sqrt(1.0 - Kc / K) if K>Kc else 0.0 for K in Ks])
        plt.plot(Ks, rs, label=f"$n={n}$", lw=2)
    else:
        plt.plot(Ks, rs, label=f"$n={n}$", lw=2)

# infinite n
Kc = 2.0 / np.pi / inf_dist(0.0, gamma)
Ks = np.arange(Kc+0.001, Kc+0.5, 0.001)
Ks = np.concatenate([Ks, np.arange(Kc+0.5, 5.001, 0.1)])
rs = []
for K in tqdm(Ks):
    r = 1.0
    for _ in range(100):
        integral_fn = lambda theta: np.cos(theta)**2*inf_dist(K*r*np.sin(theta), gamma)
        integral_val = integral_fn(thetas).sum() * dt
        r = K * r * integral_val
    rs.append(r)
plt.plot(Ks, rs, label=r"$n=\infty$", lw=2, c="tab:red")
plt.plot([Kc, Kc], [0, rs[0]], lw=2, c="tab:red", ls="dashed")
plt.plot([0, Kc], [0, 0], lw=2, c="tab:red")

plt.legend()
plt.savefig("../figs/general_dist_bif_diagram.pdf", bbox_inches="tight")