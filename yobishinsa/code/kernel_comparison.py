import jax.numpy as jnp
import jax.random as jr
from jax.config import config
import gpjax as gpx
import jaxkern as jk
from jaxtyping import Array, Float
from typing import Dict
import matplotlib.pyplot as plt
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm,mathpazo}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)

class Periodic(jk.base.AbstractKernel):
    """Periodic kernel.

    The periodic kernel is a stationary kernel defined by:

    .. math::
        k(x, x') = \exp(c  \cos(x-x'))

    where :math:`p` is the period and :math:`l` is the lengthscale.

    """
    def __init__(self, period) -> None:
        super().__init__()
        self.period = period
        self.omega: float = 2 * jnp.pi / self.period

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        c = params["c"]
        K = jnp.exp(c * jnp.cos(self.omega * (x - y)))
        return K.squeeze()

    def init_params(self, key: jr.KeyArray) -> dict:
        return {"c": jnp.array([1.0])}

    # This is depreciated. Can be removed once JaxKern is updated.
    def _initialise_params(self, key: jr.KeyArray) -> Dict:
        return self.init_params(key)

kernels = [
    gpx.RBF(),
    gpx.Matern12(),
    Periodic(period=2.0)
]

names = ["RBF", r"Mat\'{e}rn ($\nu=1/2$)", "Periodic"]

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(24, 5))

x = jnp.linspace(-3.0, 3.0, num=200).reshape(-1, 1)

counter = -1
for k, ax in zip(kernels, axes.ravel()):
    counter += 1
    ax.set_xlim(-3.0, 3.0)
    prior = gpx.Prior(kernel=k)
    params, *_ = gpx.initialise(prior, key).unpack()
    rv = prior(params)(x)
    y = rv.sample(seed=key, sample_shape=(10,))

    ax.plot(x, y.T)
    ax.set_title(names[counter], fontdict={"fontsize": 40})

plt.savefig("../figs/kernel_comparison.pdf", bbox_inches="tight")