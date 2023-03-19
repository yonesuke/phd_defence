import numpy as np
import jax.numpy as jnp
from prax import Oscillator
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

class VanderPol(Oscillator):
    def __init__(self, mu, dt=0.01, eps=10**-5):
        super().__init__(n_dim=2, dt=dt, eps=eps)
        self.mu = mu

    def forward(self, state):
        x, y = state
        vx = y
        vy = self.mu * (1.0 - x*x) * y - x
        return jnp.array([vx, vy])

class StuartLandau(Oscillator):
    def __init__(self, a, b, dt=0.01, eps=10**-5):
        super().__init__(n_dim=2, dt=dt, eps=eps)
        self.a = a
        self.b = b

    def forward(self, state):
        x, y = state
        vx = x - self.a * y - (x - self.b * y) * (x * x + y * y)
        vy = self.a * x + y - (self.b * x + y) * (x * x + y * y)
        return jnp.array([vx, vy])

class FitzHughNagumo(Oscillator):
    def __init__(self, params, dt=0.01, eps=10**-5):
        super().__init__(n_dim=2, dt=dt, eps=eps)
        self.a, self.b, self.c = params

    def forward(self, state):
        x, y = state
        vx = self.c * (x - x ** 3 - y)
        vy = x - self.b * y + self.a
        return jnp.array([vx, vy])

class Brusselator(Oscillator):
    def __init__(self, params, dt=0.01, eps=10**-5):
        super().__init__(n_dim=2, dt=dt, eps=eps)
        self.a, self.b = params

    def forward(self, state):
        x, y = state
        vx = self.a - (self.b + 1.0) * x + x * x * y
        vy = self.b * x - x * x * y
        return jnp.array([vx, vy])

vdp = VanderPol(mu=0.2)
_init_val = jnp.array([0.1, 0.2])
vdp.find_periodic_orbit(_init_val)
vdp_orbit = vdp.periodic_orbit

sl = StuartLandau(1.0, 0.5)
_init_val = jnp.array([0.1, 0.2])
sl.find_periodic_orbit(_init_val)
sl_orbit = sl.periodic_orbit

fhn = FitzHughNagumo(params=(0.2, 0.5, 10.0))
_init_val = jnp.array([0.1, 0.2])
fhn.find_periodic_orbit(_init_val)
fhn_orbit = fhn.periodic_orbit

br = Brusselator(params=(1.0, 3.0))
_init_val = jnp.array([0.1, 0.2])
br.find_periodic_orbit(_init_val, section=2.0)
br_orbit = br.periodic_orbit

plt.figure(figsize=(14, 14))
plt.rcParams["font.size"] = 20
plt.subplots_adjust(wspace=0.1, hspace=0.05)

# vdp
vdp_idxs = np.load("isochron_vdp.npy")
x_min, x_max, y_min, y_max = -2.5, 2.5, -2.5, 2.5
plt.subplot(2, 2, 1)
plt.title("Van der Pol")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.matshow(jnp.flipud(vdp_idxs), cmap="hsv", extent=[x_min, x_max, y_min, y_max], fignum=0, aspect="equal")
cbar = plt.colorbar(ticks=[0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi, 2.0 * jnp.pi])
cbar.ax.set_yticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
plt.plot(vdp_orbit[:, 0], vdp_orbit[:, 1], "k", lw=2)
plt.gca().xaxis.tick_bottom()

# sl
sl_idxs = np.load("isochron_sl.npy")
x_min, x_max, y_min, y_max = -1.2, 1.2, -1.2, 1.2
plt.subplot(2, 2, 2)
plt.title("Stuart Landau")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.matshow(jnp.flipud(sl_idxs), cmap="hsv", extent=[x_min, x_max, y_min, y_max], fignum=0, aspect="equal")
cbar = plt.colorbar(ticks=[0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi, 2.0 * jnp.pi])
cbar.ax.set_yticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
plt.plot(sl_orbit[:, 0], sl_orbit[:, 1], "k", lw=2)
plt.gca().xaxis.tick_bottom()

# fhn
fhn_idxs = np.load("isochron_fhn.npy")
x_min, x_max, y_min, y_max = -1.5, 1.5, -0.6, 0.7
plt.subplot(2, 2, 3)
plt.title("FitzHugh Nagumo")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.matshow(jnp.flipud(fhn_idxs), cmap="hsv", extent=[x_min, x_max, y_min, y_max], fignum=0, aspect=(x_max-x_min)/(y_max-y_min))
cbar = plt.colorbar(ticks=[0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi, 2.0 * jnp.pi])
cbar.ax.set_yticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
plt.plot(fhn_orbit[:, 0], fhn_orbit[:, 1], "k", lw=2)
plt.gca().xaxis.tick_bottom()

# br
br_idxs = np.load("isochron_br.npy")
x_min, x_max, y_min, y_max = 0.0, 4.0, 0.5, 5.0
plt.subplot(2, 2, 4)
plt.title("Brusselator")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.matshow(jnp.flipud(br_idxs), cmap="hsv", extent=[x_min, x_max, y_min, y_max], fignum=0, aspect=(x_max-x_min)/(y_max-y_min))
cbar = plt.colorbar(ticks=[0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi, 2.0 * jnp.pi])
cbar.ax.set_yticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
plt.plot(br_orbit[:, 0], br_orbit[:, 1], "k", lw=2)
plt.gca().xaxis.tick_bottom()

plt.savefig("../figs/isochronous_map.pdf", bbox_inches="tight", pad_inches=0.0)

# plt.savefig("isochron.png", dpi=300, bbox_inches="tight", pad_inches=0.0)