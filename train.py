''' 
Model Agnostic Meta-Learning (MAML) training for 
spiking neural networks for adaptation on the edge computing.

Author: Yigit Demirag, Forschungszentrum Jülich, 2022
'''

from jax import vmap, jit, value_and_grad
import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import tree_map
from jax.example_libraries import optimizers
import time 
import matplotlib.pyplot as plt
from network import lif_forward
from utils import sample_sinusoid_task, param_initializer

def train_meta_fp32(key, batch_train, batch_test, n_iter, n_inp,
                    n_out, n_h0, n_h1, task_size, tau_mem, tau_out,
                    lr_in, lr_out, target_fr, lr_drop, lambda_fr):

    def net_step(h, x_t):
        h, out_t = lif_forward(h, x_t)
        return h, out_t

    def task_predict(weights, X):
        _, net_const, net_dyn = param_initializer(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out)
        _, out = scan(net_step, [weights, net_const, net_dyn], X, length=100)
        return out

    batch_task_predict = vmap(task_predict, in_axes=(None, 0))

    def inner_loss(weights, X, Y):
        z0, z1, Yhat = batch_task_predict(weights, X)
        L_mse = jnp.mean((Y - Yhat)**2)
        fr0 = 10*jnp.mean(z0)
        fr1 = 10*jnp.mean(z1)  
        L_fr = jnp.mean(target_fr-fr0) ** 2 + jnp.mean(target_fr-fr1) ** 2
        return L_mse, [fr0, fr1, L_fr, L_mse]

    def outer_loss(weights, X, Y):
        z0, z1, Yhat = batch_task_predict(weights, X)
        L_mse = jnp.mean((Y - Yhat)**2)
        fr0 = 10*jnp.mean(z0)
        fr1 = 10*jnp.mean(z1)  
        L_fr = jnp.mean(target_fr-fr0) ** 2 + jnp.mean(target_fr-fr1) ** 2
        loss = L_mse + lambda_fr * L_fr
        return loss
 
    def update_in(theta, sX, sY): 
        # Calculate gradients 
        value, grads_in = value_and_grad(inner_loss, has_aux=True)(theta, sX, sY)

        # Inner update
        def inner_sgd_fn(g, p):
            p = p - lr_in * g
            p = jnp.clip(p, -1, 1)
            return p

        updated_weights = tree_map(inner_sgd_fn, grads_in, theta)

        metrics ={'Inner L_fr': value[1][2],
                  'Inner L_mse': value[1][3],
                  'fr0': value[1][0],
                  'fr1': value[1][1],
                  'theta-0': theta[0],
                  'theta-1': theta[2],
                  'theta-2': theta[4],
                  'grad-in-0': grads_in[0],
                  'grad-in-1': grads_in[2],
                  'grad-in-2': grads_in[4],
                  'dW-0': updated_weights[0] - theta[0],
                  'dW-1': updated_weights[2] - theta[2],
                  'dW-2': updated_weights[4] - theta[4]}
            
        return updated_weights, metrics

    def maml_loss(theta, sX, sY, qX, qY):
        updated_weights, metrics = update_in(theta, sX, sY)
        return outer_loss(updated_weights, qX, qY), metrics

    def batched_maml_loss(theta, sX, sY, qX, qY):
        task_losses, metrics = vmap(maml_loss, in_axes=(None, 0, 0, 0, 0))(theta, sX, sY, qX, qY)
        return jnp.mean(task_losses), metrics

    @jit
    def update_out(i, opt_state, sX, sY, qX, qY):
        theta = get_params(opt_state)
        (L, metrics), grads_out = value_and_grad(batched_maml_loss, has_aux=True)(theta, sX, sY, qX, qY)
        metrics = {k:jnp.mean(v, axis=0) for (k,v) in metrics.items()}
        metrics['Outer loss'] = L
        opt_state = opt_update(i, grads_out, opt_state)
        return opt_state, metrics

    piecewise_lr = optimizers.piecewise_constant([lr_drop], [lr_out, lr_out/10])
    opt_init, opt_update, get_params = optimizers.adam(step_size=piecewise_lr)
    weights, _, _ = param_initializer(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out)
    opt_state = opt_init(weights)
    
    # Start meta-training
    for epoch in range(n_iter):
        t0 = time.time()
        key, key_eval = random.split(key, 2)
        sX, sY, qX, qY = sample_sinusoid_task(key, batch_size=batch_train, 
                                              num_samples_per_task=task_size)
        opt_state, metrics = update_out(epoch, opt_state, sX, sY, qX, qY)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch} - Loss: {metrics["Outer loss"]:.3f} - Time : {(time.time()-t0):.3f} s')
            wandb.log(metrics)
    print('Meta-training completed.')

    # Evaluate fine tuning
    sX, sY, qX, qY = sample_sinusoid_task(key_eval, batch_size=batch_test, 
                                          num_samples_per_task=100)
    weights = get_params(opt_state)

    sX_t = sX[:,:task_size,:,:] # batch, number, time, dim
    sY_t = sY[:,:task_size,:,:]
    plt.figure(figsize=(10,4));
    c = ['slateblue', 'darkblue']
    for i in range(2): # Initial prediction followed by one-shot prediction
        z0, z1, sYhat = vmap(batch_task_predict, in_axes=(None, 0))(weights, sX_t)
        plt.plot(sX_t[0,:,-1,0], sYhat[0,:,-1,0], '.', label='Prediction '+str(i), color=c[i])
        weights, metrics = vmap(update_in, in_axes=(None, 0, 0))(weights, sX_t, sY_t)
        weights = tree_map(lambda W: jnp.mean(W, axis=0), weights)

    # Save figure
    plt.plot(sX[0,:,-1,0], sY[0,:,-1,0], '.', label='Ground truth', color='red')
    plt.grid(True); plt.legend()
    wandb.log({"Meta-testing": plt})
    plt.savefig('meta_testing.png')
    print('Meta-testing completed.')

if __name__ == '__main__':
    from jax import random
    import argparse
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=23, help='Random seed')
    parser.add_argument('--batch_train', type=int, default=256, help='Batch size for meta-training')
    parser.add_argument('--batch_test', type=int, default=1, help='Batch size for meta-testing')
    parser.add_argument('--n_iter', type=int, default=20000, help='Number of iterations')
    parser.add_argument('--n_inp', type=int, default=1, help='Number of input neurons')
    parser.add_argument('--n_out', type=int, default=1, help='Number of output neurons')
    parser.add_argument('--n_h0', type=int, default=40, help='Number of neurons in the first hidden layer')
    parser.add_argument('--n_h1', type=int, default=40, help='Number of neurons in the second hidden layer')
    parser.add_argument('--task_size', type=int, default=20, help='Number of samples per task')
    parser.add_argument('--tau_mem', type=float, default=10e-3, help='Membrane time constant')
    parser.add_argument('--tau_out', type=float, default=1e-3, help='Output time constant')
    parser.add_argument('--lr_in', type=float, default=1, help='Learning rate for the inner layer')
    parser.add_argument('--lr_out', type=float, default=1e-2, help='Learning rate for the output layer')
    parser.add_argument('--target_fr', type=float, default=2, help='Target firing rate')
    parser.add_argument('--lr_drop', type=int, default=12000, help='The step number for dropping the learning rate')
    parser.add_argument('--lambda_fr', type=float, default=0, help='Regularization parameter for the firing rate')
    args = parser.parse_args()

    wandb.init(project='meta-analog', config=args)
    wandb.config.update(args)

    train_meta_fp32(key=random.PRNGKey(args.seed),
                      batch_train=args.batch_train,
                      batch_test=args.batch_test,
                      n_iter=args.n_iter,
                      n_inp=args.n_inp,
                      n_out=args.n_out,
                      n_h0=args.n_h0,
                      n_h1=args.n_h1,
                      task_size=args.task_size,
                      tau_mem=args.tau_mem,
                      tau_out=args.tau_out,
                      lr_in=args.lr_in,
                      lr_out=args.lr_out,
                      target_fr=args.target_fr,
                      lr_drop=args.lr_drop,
                      lambda_fr=args.lambda_fr)