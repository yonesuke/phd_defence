import numpy as np
import jax.numpy as jnp
from prax import Oscillator
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

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

sl = StuartLandau(1.0, 0.5)
_init_val = jnp.array([0.1, 0.2])
sl.find_periodic_orbit(_init_val)
sl_orbit = sl.periodic_orbit


plt.figure(figsize=(8,8))
plt.rcParams["font.size"] = 20
sl_idxs = np.load("isochron_sl.npy")
x_min, x_max, y_min, y_max = -1.2, 1.2, -1.2, 1.2
plt.title("Stuart Landau, $b=0.5$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.matshow(jnp.flipud(sl_idxs), cmap="hsv", extent=[x_min, x_max, y_min, y_max], fignum=0, aspect="equal")
cbar = plt.colorbar(ticks=[0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi, 2.0 * jnp.pi])
cbar.ax.set_yticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
plt.plot(sl_orbit[:, 0], sl_orbit[:, 1], "k")
plt.gca().xaxis.tick_bottom()

plt.savefig("sl.png", dpi=300, bbox_inches="tight")