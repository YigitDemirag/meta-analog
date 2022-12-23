''' 
Leaky Integrate-and-Fire (LIF) neuron model with 
surrogate gradient learning

Author: Yigit Demirag, Forschungszentrum Jülich, 2022
'''

import jax.numpy as jnp
from utils import gr_than

def lif_forward(state, x):
    ''' Simplified (no alpha kernel on synapses) 2-layer FF LIF network
    '''
    w0, b0, w1, b1, w2     = state[0]   # Static weights and biases
    alpha, kappa           = state[1]   # Static neuron states
    v0, z0, v1, z1, v2     = state[2]   # Dynamic neuron states

    v0  = alpha * v0 + jnp.dot(x, w0) + b0 - z0 # Membrane volt
    z0  = gr_than(v0) # Spiking state of first layer
    v1  = alpha * v1 + jnp.dot(z0, w1) + b1 - z1 # Membrane volt
    z1  = gr_than(v1) # Spiking state of second layer
    v2  = kappa * v2 + jnp.dot(z1, w2) # Leaky integrator output unit

    return [[w0, b0, w1, b1, w2], [alpha, kappa], [v0, z0, v1, z1, v2]], [z0, z1, v2]
