''' 
Utils

Author: Yigit Demirag, Forschungszentrum Jülich, 2022
'''

import jax 
from jax import jit
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import tree_map
from einops import repeat
from functools import partial

GMAX = 20.0
GMIN = 0.1

BETA = 1

@jax.custom_jvp
def gr_than(x, thr=1.0):
    ''' Heaviside spiking implementation
    '''
    return (x > thr).astype(jnp.float32)


@gr_than.defjvp
def gr_jvp(primals, tangents):
    ''' Twice-differentiable fast sigmoid function to implement
        surrogate gradient learning.

        Gradient scale factor is a critical  hyperparameter that
        needs to be optimized for the task.
    '''
    x, thr = primals
    x_dot, y_dot = tangents
    primal_out = gr_than(x, thr)
    tangent_out = x_dot / (BETA * jnp.absolute(x - thr) + 1) ** 2
    return primal_out, tangent_out

@jax.custom_jvp
def ls_than(x, thr):
    ''' Less than implementation
    '''
    return (x < thr).astype(jnp.float32)

@ls_than.defjvp
def lt_jvp(primals, tangents):
    ''' Straight-through estimator for lt() operation
    '''
    x, thr = primals
    x_dot, y_dot = tangents
    primal_out = ls_than(x, thr)
    tangent_out = x_dot / -(BETA * jnp.absolute(x - thr) + 1) ** 2
    return primal_out, tangent_out

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

@partial(jit, static_argnums=(1,2,3,4,5,6))
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
                                               maxval= jnp.sqrt(6/(n_h0+n_h1))) * 1
    w2 = random.uniform(key_h2, [n_h1, n_out], minval=-jnp.sqrt(6/(n_h1+n_out)),
                                               maxval= jnp.sqrt(6/(n_h1+n_out))) * 0.01
    b0 = jnp.zeros(n_h0)
    b1 = jnp.zeros(n_h1)
    b2 = jnp.zeros(n_out)

    neuron_dyn = [jnp.zeros(n_h0), jnp.zeros(n_h0), jnp.zeros(n_h1), 
                  jnp.zeros(n_h1), jnp.zeros(n_out)]
    net_params = [[w0, b0, w1, b1, w2, b2], [alpha, kappa], neuron_dyn]
    return net_params

@partial(jit, static_argnums=(1,2,3,4,5,6))
def analog_init(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out):
    """
    Maps ideal FP32 weights to G+ and G- differential memristor
    conductances such that W = (G+ - G-) / (GMAX - GMIN) by 
    initializing G+ and calculating G-. 
    This initialization matters only for offline SW training.
    """
    net_params = param_initializer(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out)

    G_pos = [random.uniform(l_key, shape=l_W.shape, minval=7, maxval=12) 
             for l_key, l_W 
             in zip(random.split(key, len(net_params[0])), net_params[0])]

    G_neg = [Gp - W * (GMAX-GMIN) for Gp, W in zip(G_pos, net_params[0])]

    devices = [dict({'G' : jnp.stack([Gp, Gn]),
                     'tp': 0.0 * jnp.stack([Gp, Gp])}) for Gp, Gn in zip(G_pos, G_neg)]

    return [devices, net_params[1], net_params[2]]