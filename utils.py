''' 
Utils

Author: Yigit Demirag, Forschungszentrum Jülich, 2022
'''

from re import L
from typing import Any
from jax import jit
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.lax import cond
from einops import repeat
from torch import true_divide

def sample_sinusoid_task(key, batch_size, num_samples_per_task):
    key_A, key_phi, key_s, key_q = random.split(key, 4)
    A_list = random.uniform(key_A, [batch_size], minval=0.1, maxval=5)
    phi_list = random.uniform(key_phi, [batch_size], minval=0, maxval=jnp.pi)
    xS = random.uniform(key_s, [batch_size, num_samples_per_task, 1], minval=-5, maxval=5)
    xQ = random.uniform(key_q, [batch_size, num_samples_per_task, 1], minval=-5, maxval=5)
    
    yS = jnp.zeros((batch_size, num_samples_per_task, 1))
    yQ = jnp.zeros((batch_size, num_samples_per_task, 1))
    for i, (A, phi) in enumerate(zip(A_list, phi_list)):
        yS = yS.at[i].set(A * jnp.sin(xS[i] + phi))
        yQ = yQ.at[i].set(A * jnp.sin(xQ[i] + phi)) 

    xS, yS, xQ, yQ = tree_map(lambda x: repeat(x, 'b n d -> b n t d', t=100), 
                                                  [xS, yS, xQ, yQ])
    return xS, yS, xQ, yQ

j_sample_sinusoid_task=jit(sample_sinusoid_task, static_argnums=(1,2))

def param_initializer(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out):
    ''' Initialize parameters
    '''
    key_h0, key_h1, key_h2 = random.split(key, 3)
    alpha = jnp.exp(-1e-3/tau_mem) 
    kappa = jnp.exp(-1e-3/tau_out)

    # Weights
    w0 = random.uniform(key_h0, [n_inp, n_h0], minval=-jnp.sqrt(6/(n_inp+n_h0)), 
                                               maxval= jnp.sqrt(6/(n_inp+n_h0))) * 0.1
    w1 = random.uniform(key_h1, [n_h0, n_h1],  minval=-jnp.sqrt(6/(n_h0+n_h1)), 
                                               maxval= jnp.sqrt(6/(n_h0+n_h1))) * 0.1
    w2 = random.uniform(key_h2, [n_h1, n_out], minval=-jnp.sqrt(6/(n_h1+n_out)),
                                               maxval= jnp.sqrt(6/(n_h1+n_out))) * 0.1
    # Biases
    b0 = jnp.zeros(n_h0)
    b1 = jnp.zeros(n_h1)
    b2 = jnp.zeros(n_out)

    neuron_dyn = [jnp.zeros(n_h0), jnp.zeros(n_h0), jnp.zeros(n_h1), 
                  jnp.zeros(n_h1), jnp.zeros(n_out)]
    net_params = [[w0, b0, w1, b1, w2, b2], [alpha, kappa], neuron_dyn]
    return net_params