from typing import List, Union, Optional

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow import Parameter
from gpflow.utilities import positive
from gpflow.utilities.ops import difference_matrix
from gpflow.kernels.base import Kernel
# from gpflow.kernels.stationaries import IsotropicStationary, Stationary

from tqdm import tqdm

def run_adam(model, iterations, data, minibatch_size):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    n_data = data[0].size

    # train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(n_data)
    train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat()
    
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in tqdm(range(iterations)):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf

def run_scipy_minimize(model, iterations, data, minibatch_size, method="l-bfgs-b"):
    n_data = data[0].size

    train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(n_data)
    
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)

    res = gpflow.optimizers.Scipy().minimize(
        training_loss,
        variables = model.trainable_variables,
        method = method,
        options = {"disp": True}
    )
    return res

class NormalizedPeriodic(Kernel):
    def __init__(self):
        super().__init__()
        self.variance = Parameter(1.0, transform=positive())
        self.lengthscale = Parameter(1.0, transform=positive())

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        l = self.lengthscale
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance)) / tf.math.bessel_i0e(0.5 / l**2) / 2.0 / np.pi

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        r = difference_matrix(X, X2)
        v = self.variance
        l = self.lengthscale
        scaled_cosine = tf.cos(r) * 0.5 / l**2
        cos_r = tf.reduce_sum(tf.abs(scaled_cosine), -1)
        K = v * tf.exp(cos_r) / 2.0 / np.pi / tf.math.bessel_i0(0.5 / l**2)
        return K

