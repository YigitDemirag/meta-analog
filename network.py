''' 
Leaky Integrate-and-Fire (LIF) neuron model with 
surrogate gradient learning

Author: Yigit Demirag, Forschungszentrum Jülich, 2022
'''

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

BETA = 8

@jax.custom_jvp
def gr_than(x, thr=1):
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

def lif_forward(state, x):
    ''' Simplified (no alpha kernel on synapses) 2-layer FF LIF network
    '''
    w0, b0, w1, b1, w2, b2 = state[0]   # Static weights and biases
    alpha, kappa           = state[1]   # Static neuron states
    v0, z0, v1, z1, v2     = state[2]   # Dynamic neuron states

    v0  = alpha * v0 + jnp.dot(x, w0) + b0 - z0 # Membrane volt
    z0  = gr_than(v0) # Spiking state of first layer
    v1  = alpha * v1 + jnp.dot(z0, w1) + b1 - z1 # Membrane volt
    z1  = gr_than(v1) # Spiking state of second layer
    v2  = kappa * v2 + jnp.dot(z1, w2) # Leaky integrator output unit

    return [[w0, b0, w1, b1, w2, b2], [alpha, kappa], [v0, z0, v1, z1, v2]], [z0, z1, v2]
