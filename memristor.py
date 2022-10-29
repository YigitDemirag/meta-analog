''' 
Simplified differentiable PCM model (material: GST)

This model is a simplified version of the empirical PCM model introduced in
Nandakumar et. al., 2018. The original model estimates the next conductance
based on number of programming pulses applied. Here, we model current
conductance to next conductance (G -> ΔG) transition after the application 
of a single WRITE pulse.

Author: Yigit Demirag, Forschungszentrum Jülich, 2022
'''

from jax import jit 
import jax.numpy as jnp
import jax.random as random
from jax.lax import cond
from jax.tree_util import tree_map

G0 = 0.1   # (µS) Initial device conductance mean
Gmax = 20  # (µS) Maximum device conductance
n_bits = 3 # Ideal device bit-resolution, to compare with SRAM 

@jit
def write(key, device, tp, perf=True):
    ''' Simulates a single SET pulse.
    ''' 

    # Programming noise
    mu_dgn  = - 0.084 * device['G'] + 0.892
    std_dgn =   0.09  * device['G'] + 0.575
    dgn = mu_dgn + std_dgn * random.normal(key, shape=device['G'].shape)

    device['G'] = cond(perf, 
                       lambda G : jnp.clip(G + Gmax/(2**n_bits), 0.1, Gmax),
                       lambda G : jnp.clip(G + dgn, 0.1, Gmax),
                       device['G'])
    device['tp'] = tp * jnp.ones_like(device['tp']) 
    return device

@jit
def read(key, device, t, perf=True):
    ''' Reads the conductance of device at given time.
    '''
    t0 = 38.6 
    v  = 0.04
    m3 = 0.03
    c3 = 0.13

    # Temporal conductance drift
    Gd = device['G'] * jnp.power((t - device['tp'])/t0 , -v)

    # 1/f read noise
    std_nG = m3 * Gd + c3
    nG = random.normal(key, shape=device['G'].shape) * std_nG

    Gn = cond(perf, 
              lambda G : jnp.clip(G, 0.1, Gmax),
              lambda G : jnp.clip(G + nG, 0.1, Gmax),
              device['G'])
    return Gn

@jit
def reset(key, device, tp):
    ''' Simulates a single RESET pulse.
    '''
    device['G']  = G0 + random.normal(key, shape=device['G'].shape) * G0 * 0.1
    device['G']  = device['G'].clip(1e-2, Gmax)
    device['tp'] = tp * jnp.ones_like(device['G'])
    return device

def create_devs(key, shape):
    ''' Creates device array with given shape.
    '''
    devices = reset(key, dict({'G' : jnp.zeros(shape), 
                               'tp': jnp.zeros(shape)}), tp=0.0)
    return devices

def zero_time_dim(dstate):
    ''' Reset the time dimension of device state.
    '''
    dstate = tree_map(lambda dstate: {'G' : dstate['G'], 
                                      'tp': jnp.zeros_like(dstate['tp'])}, 
                      dstate, 
                      is_leaf=lambda x: isinstance(x, dict))
    return dstate