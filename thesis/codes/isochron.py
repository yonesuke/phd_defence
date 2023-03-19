import numpy as np
import jax.numpy as jnp
from jax import vmap
from prax import Oscillator
from jax.config import config; config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

def get_min_idx(init_val, model, t_max, periodic_orbit):
    final_val = model.run(init_val, t_max)[1][-1]
    idx = jnp.sum((periodic_orbit - final_val)**2, axis=1).argmin()
    return idx

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

n_step = 300

vdp = VanderPol(mu=0.2)
_init_val = jnp.array([0.1, 0.2])
vdp.find_periodic_orbit(_init_val)
vdp_orbit = vdp.periodic_orbit

xs = jnp.linspace(-2.5, 2.5, n_step)
ys = jnp.linspace(-2.5, 2.5, n_step)
init_vals = jnp.vstack([jnp.tile(xs, n_step), jnp.repeat(ys, n_step)]).T
t_max = (100 // vdp.period) * vdp.period
vdp_idxs = vmap(lambda init_val: get_min_idx(init_val, vdp, t_max, vdp_orbit))(init_vals)
vdp_idxs = vdp_idxs.reshape(n_step, n_step) * 2.0 * jnp.pi / 4096

np.save("isochron_vdp.npy", vdp_idxs)

sl = StuartLandau(1.0, 0.5)
_init_val = jnp.array([0.1, 0.2])
sl.find_periodic_orbit(_init_val)
sl_orbit = sl.periodic_orbit

xs = jnp.linspace(-1.2, 1.2, n_step)
ys = jnp.linspace(-1.2, 1.2, n_step)
init_vals = jnp.vstack([jnp.tile(xs, n_step), jnp.repeat(ys, n_step)]).T
t_max = (100 // sl.period) * sl.period
sl_idxs = vmap(lambda init_val: get_min_idx(init_val, sl, t_max, sl_orbit))(init_vals)
sl_idxs = sl_idxs.reshape(n_step, n_step) * 2.0 * jnp.pi / 4096

np.save("isochron_sl.npy", sl_idxs)

fhn = FitzHughNagumo(params=(0.2, 0.5, 10.0))
_init_val = jnp.array([0.1, 0.2])
fhn.find_periodic_orbit(_init_val)
fhn_orbit = fhn.periodic_orbit

xs = jnp.linspace(-1.5, 1.5, n_step)
ys = jnp.linspace(-0.6, 0.7, n_step)
init_vals = jnp.vstack([jnp.tile(xs, n_step), jnp.repeat(ys, n_step)]).T
t_max = (100 // fhn.period) * fhn.period
fhn_idxs = vmap(lambda init_val: get_min_idx(init_val, fhn, t_max, fhn_orbit))(init_vals)
fhn_idxs = fhn_idxs.reshape(n_step, n_step) * 2.0 * jnp.pi / 4096

np.save("isochron_fhn.npy", fhn_idxs)

br = Brusselator(params=(1.0, 3.0))
_init_val = jnp.array([0.1, 0.2])
br.find_periodic_orbit(_init_val, section=2.0)
br_orbit = br.periodic_orbit

xs = jnp.linspace(0.0, 4.0, n_step)
ys = jnp.linspace(0.5, 5.0, n_step)
init_vals = jnp.vstack([jnp.tile(xs, n_step), jnp.repeat(ys, n_step)]).T
t_max = (100 // br.period) * br.period
br_idxs = vmap(lambda init_val: get_min_idx(init_val, br, t_max, br_orbit))(init_vals)
br_idxs = br_idxs.reshape(n_step, n_step) * 2.0 * jnp.pi / 4096

np.save("isochron_br.npy", br_idxs)