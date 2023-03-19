import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, vmap
import jaxkern as jk
import gpjax as gpx

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

n_val = 100
_xs = jnp.linspace(-1, 1, n_val)
xs = jnp.vstack([jnp.tile(_xs, n_val), jnp.repeat(_xs, n_val)]).T
XX, YY = jnp.meshgrid(_xs, _xs)

seed = 1
key = random.PRNGKey(seed)
kernel = jk.RBF(active_dims=[0], ) + jk.RBF(active_dims=[1])
prior = gpx.Prior(kernel=kernel)
params, *_ = gpx.initialise(prior, key).unpack()
params["kernel"][0]["lengthscale"] = jnp.array([0.3])
params["kernel"][1]["lengthscale"] = jnp.array([0.3])
params["kernel"][0]["variance"] = jnp.array([0.5])
params["kernel"][1]["variance"] = jnp.array([0.5])
rv = prior(params)(xs)
y = rv.sample(seed=key, sample_shape=(1,)).reshape(n_val, n_val)
decompose1, decompose2 = jnp.tile(y[0].reshape(-1, 1), n_val).T, jnp.tile(y[:, 0].reshape(-1, 1), n_val) - y[0,0]

sigma = 0.5
lengthscale = 0.3
kernel_1d_fn = lambda x: sigma **2 * jnp.exp(-0.5 * (x / lengthscale) ** 2)
@jit
def kernel_2d_fn(state):
    x1, x2 = state
    return kernel_1d_fn(x1) * kernel_1d_fn(x2)
kern_vals = vmap(kernel_2d_fn)(xs).reshape(n_val, n_val)
kern1_vals, kern2_vals = jnp.tile(kern_vals[0].reshape(-1, 1), n_val).T, jnp.tile(kern_vals[:, 0].reshape(-1, 1), n_val) - kern_vals[0,0] * 0.5

fnames = ["additive_gp_sample", "additive_gp_decompose1", "additive_gp_decompose2", "additive_gp_kernel", "additive_gp_kernel1", "additive_gp_kernel2"]
arrays = [y, decompose1, decompose2, kern_vals, kern1_vals, kern2_vals]
for i in range(6):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d", facecolor="w")
    ax.set_xlabel("$x_1$", fontdict={"fontsize": 30})
    ax.set_ylabel("$x_2$", fontdict={"fontsize": 30})
    ax.plot_surface(XX, YY, arrays[i], cmap="hsv")
    plt.savefig("../figs/" + fnames[i] + ".pdf", bbox_inches="tight")