import numpy as np
import jax.numpy as jnp
from jax import vmap
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
vdp.calc_phase_response()

sl = StuartLandau(1.0, 0.5)
_init_val = jnp.array([0.1, 0.2])
sl.find_periodic_orbit(_init_val)
sl.calc_phase_response()

fhn = FitzHughNagumo(params=(0.2, 0.5, 10.0))
_init_val = jnp.array([0.1, 0.2])
fhn.find_periodic_orbit(_init_val)
fhn.calc_phase_response()

br = Brusselator(params=(1.0, 3.0))
_init_val = jnp.array([0.1, 0.2])
br.find_periodic_orbit(_init_val, section=2.0)
br.calc_phase_response()

plt.figure(figsize=(16, 12))
plt.rcParams["font.size"] = 16

plt.subplot(2, 2, 1)
plt.title("Van der Pol")
plt.xlim(vdp.ts[0], vdp.ts[-1])
plt.xlabel("$t$")
plt.plot(vdp.ts, vdp.phase_response_curve, lw=2)
plt.legend(labels=["$Z_x$", "$Z_y$"])

plt.subplot(2, 2, 2)
plt.title("Stuart-Landau")
plt.xlim(sl.ts[0], sl.ts[-1])
plt.xlabel("$t$")
plt.plot(sl.ts, sl.phase_response_curve, lw=2)
plt.legend(labels=["$Z_x$", "$Z_y$"])

plt.subplot(2, 2, 3)
plt.title("FitzHugh-Nagumo")
plt.xlim(fhn.ts[0], fhn.ts[-1])
plt.xlabel("$t$")
plt.plot(fhn.ts, fhn.phase_response_curve, lw=2)
plt.legend(labels=["$Z_x$", "$Z_y$"])

plt.subplot(2, 2, 4)
plt.title("Brusselator")
plt.xlim(br.ts[0], br.ts[-1])
plt.xlabel("$t$")
plt.plot(br.ts, br.phase_response_curve, lw=2)
plt.legend(labels=["$Z_x$", "$Z_y$"])

plt.savefig("../figs/prc.pdf", bbox_inches="tight")