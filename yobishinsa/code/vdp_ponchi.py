import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
from mpl_toolkits.mplot3d import Axes3D

def runge_kutta(func, state, dt):
    k1 = func(state)
    k2 = func(state + 0.5 * dt * k1)
    k3 = func(state + 0.5 * dt * k2)
    k4 = func(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def vdp_3d(state, mu):
    x, y, p = state
    return jnp.array([y + p, mu * (1 - x ** 2) * y - x, -p])

def odeint(func, solver, t0, t1, dt, init_state):
    update_fn = jit(lambda state: solver(func, state, dt))
    n_step = int((t1 - t0) / dt)
    orbits = jnp.zeros((n_step, *init_state.shape), dtype=init_state.dtype)
    orbits = orbits.at[0].set(init_state)
    def body_fn(i, val):
        state = val[i-1]
        state = update_fn(state)
        val = val.at[i].set(state)
        return val
    orbits = fori_loop(1, n_step, body_fn, orbits)
    return orbits


mu = 5.0
vector_fn = lambda state: vdp_3d(state, mu)
solver = runge_kutta
t0, t1 = 0.0, 1000.0
dt = 0.01
init_state = jnp.array([-0.5, 3.0, 1.5])
orbits = odeint(vector_fn, solver, t0, t1, dt, init_state)
periodic_orbits = orbits[-1300:]

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title(r"$\mathbb{R}^{d}$", fontsize=40, loc="right", pad=-50)
ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
idx = 200
ax.scatter(*periodic_orbits[idx], c="tab:blue", s=200, edgecolors="black", lw=1.5)
ax.text(*periodic_orbits[idx], "  $x_{i}$", fontsize=40, va="bottom", ha="center")
ax.plot(periodic_orbits[:, 0], periodic_orbits[:, 1], periodic_orbits[:, 2], c="gray", lw=3, alpha=0.8)

plt.savefig("vdp1.pdf", bbox_inches="tight", pad_inches=0.0, transparent=True)

mu = 0.5
vector_fn = lambda state: vdp_3d(state, mu)
solver = runge_kutta
t0, t1 = 0.0, 1000.0
dt = 0.01
init_state = jnp.array([-0.5, 3.0, 1.5])
orbits = odeint(vector_fn, solver, t0, t1, dt, init_state)
periodic_orbits = orbits[-1000:]

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title(r"$\mathbb{R}^{d}$", fontsize=40, loc="right", pad=-50)
ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
idx = 600
ax.scatter(*periodic_orbits[idx], c="tab:green", s=200, edgecolors="black", lw=1.5)
ax.text(*periodic_orbits[idx], "  $x_{j}$", fontsize=40, va="bottom", ha="center")
ax.plot(periodic_orbits[:, 0], periodic_orbits[:, 1], periodic_orbits[:, 2], c="gray", lw=3, alpha=0.8)

plt.savefig("vdp2.pdf", bbox_inches="tight", pad_inches=0.0, transparent=True)