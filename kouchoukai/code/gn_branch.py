import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxkuramoto import theory
from jaxkuramoto.distribution import GeneralNormal, Uniform

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

plt.figure(figsize=(16, 6))
plt.rcParams["font.size"] = 25

plt.subplot(1, 2, 1)
plt.title(r"$g_{n}(\omega)$")
plt.xlabel(r"$\omega$")
xs = jnp.arange(-3, 3, 0.01)
for n in [1, 2, 3, 10]:
    dist = GeneralNormal(0.0, 1.0, n)
    plt.plot(xs, dist.pdf(xs), label=f"$n={n}$", lw=2)
dist = Uniform(-1.0, 1.0)
plt.plot(xs, dist.pdf(xs), label=f"$n=\\infty$", ls="dashed", lw=2)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("bifurcation diagram")
plt.xlabel(r"$K$")
plt.ylabel(r"$r$")
Ks = jnp.arange(0.5, 2.0, 0.01)
for n in [1, 2, 3, 10]:
    dist = GeneralNormal(0.0, 1.0, n)
    rs = theory.orderparam(Ks, dist)
    plt.plot(Ks, rs, label=f"$n={n}$", lw=2)
plt.legend()

plt.savefig("../figs/gn_branch.pdf", bbox_inches="tight")

plt.figure(figsize=(16, 6))
plt.rcParams["font.size"] = 25

plt.subplot(1, 2, 1)
plt.title("uniform dist.")
plt.xlabel(r"$\omega$")
xs = jnp.arange(-3, 3, 0.01)
dist = Uniform(-1.0, 1.0)
plt.plot(xs, dist.pdf(xs), lw=2)

plt.subplot(1, 2, 2)
plt.title("bifurcation diagram")
plt.xlabel(r"$K$")
plt.ylabel(r"$r$")
Ks = jnp.arange(0.5, 2.0, 0.01)
dist = Uniform(-1.0, 1.0)
rs = theory.orderparam(Ks, dist)
Kc = theory.critical_point(dist.pdf)
Ks = np.array(Ks)
rs = np.array(rs)

idx = Ks < Kc
plt.plot(Ks[idx], rs[idx], lw=2, color="tab:blue")

idx = 77
plt.plot([Ks[idx], Ks[idx+1]], [rs[idx], rs[idx+1]], ls="dashed", lw=2, color="tab:blue")

idx = Ks > Kc
plt.plot(Ks[idx], rs[idx], lw=2, color="tab:blue")

plt.savefig("../figs/uniform_branch.pdf", bbox_inches="tight")