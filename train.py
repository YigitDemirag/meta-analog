''' 
Model Agnostic Meta-Learning (MAML) training for 
spiking neural networks with memristive synapses for 
adaptation on the edge computing.

Author: Yigit Demirag, Forschungszentrum Jülich, 2022
'''

from jax import vmap, pmap, jit, grad, value_and_grad
import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import tree_map
from jax.example_libraries import optimizers
import time 
import matplotlib.pyplot as plt
from network import lif_forward
from utils import param_initializer, j_sample_sinusoid_task, ls_than, gr_than, analog_init
from memristor import read

def train_meta_analog(key, batch_train, batch_test, n_iter, n_inp,
                      n_out, n_h0, n_h1, task_size, tau_mem, tau_out,
                      lr_out, alpha, target_fr, lambda_fr):

    def net_step(h, x_t):
        h, out_t = lif_forward(h, x_t)
        return h, out_t

    def task_predict(weights, X):
        _, net_const, net_dyn = analog_init(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out)
        _, out = scan(net_step, [weights, net_const, net_dyn], X, length=100)
        return out

    batch_task_predict = vmap(task_predict, in_axes=(None, 0))

    def loss(weights, X, Y): # 20, 100, 1
        z0, z1, Yhat = batch_task_predict(weights, X)
        L_mse = jnp.mean((Y - Yhat)**2)
        fr0 = 10*jnp.mean(z0)
        fr1 = 10*jnp.mean(z1)  
        out_m = jnp.mean(Yhat)
        L_fr = jnp.mean(target_fr-fr0) ** 2 + jnp.mean(target_fr-fr1) ** 2
        loss = L_mse + lambda_fr * L_fr
        return loss, [fr0, fr1, out_m, L_fr, L_mse]
 
    def update_in(devices, key, sX, sY, alpha):
        key_rp, key_rn, key_wp, key_wn = random.split(key, 4)

        # Get G+ and G- devices
        pos_devs = tree_map(lambda dev: tree_map(lambda G: G[0], dev), devices)
        neg_devs = tree_map(lambda dev: tree_map(lambda G: G[1], dev), devices)

        # Effective synaptic weights at t=t_read
        theta = [(read(key_rp, dp, t=1, perf=True)-read(key_rn, dn, t=1, perf=True)) 
                  / (20-0.1) for dp, dn in zip(pos_devs, neg_devs)]
        
        value, grads_in = value_and_grad(loss, has_aux=True)(theta, sX, sY) # 20, 100, 1
        loss_in, [fr0, fr1, out_m, L_fr, L_mse] = value

        def inner_sgd_fn(g, p):
            # Option 1)
            P = p - alpha * g

            # Option 2) 
            #P = jnp.where(np.abs(g) > 0, p - alpha * g, p)

            # Option 3) 
            #softsign = lambda x : x / (1 + jnp.abs(x))
            #P = gr_than(thr, jnp.abs(g)) *  p + (1 - gr_than(thr, jnp.abs(g))) * (p - alpha * softsign(g))

            P = jnp.clip(P, -1, 1)
            return P

        updated_weights = tree_map(inner_sgd_fn, grads_in, theta)

        metrics ={'Inner L_fr': L_fr,
                  'Inner L_mse': L_mse,
                  'fr0': fr0,
                  'fr1': fr1,
                  'theta-0': theta[0],
                  'theta-1': theta[2],
                  'theta-2': theta[4],
                  'grad-0': grads_in[0],
                  'grad-1': grads_in[2],
                  'grad-2': grads_in[4],
                  'dW-0': updated_weights[0] - theta[0],
                  'dW-1': updated_weights[2] - theta[2],
                  'dW-2': updated_weights[4] - theta[4],
                  'out_m': out_m}
            
        return updated_weights, metrics


    def maml_loss(devices, key, sX, sY, qX, qY):
        updated_weights, metrics = update_in(devices, key, sX, sY, alpha)
        return loss(updated_weights, qX, qY), metrics

    def batched_maml_loss(devices, key, sX, sY, qX, qY):
        task_losses, metrics = vmap(maml_loss, in_axes=(None, None, 0, 0, 0, 0))(devices, key, sX, sY, qX, qY)
        return jnp.mean(task_losses[0]), metrics

    @jit
    def update_out(key, i, opt_state, sX, sY, qX, qY):
        devices = get_params(opt_state)
        devices = tree_map(lambda W: jnp.clip(W, 0.1, 20), devices)
        (L, metrics), grads_out = value_and_grad(batched_maml_loss, has_aux=True)(devices, key, sX, sY, qX, qY)
        opt_state = opt_update(i, grads_out, opt_state)
        return opt_state, L, metrics

    opt_init, opt_update, get_params = optimizers.adam(step_size=lr_out)
    devices, _, _ = analog_init(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out)
    opt_state = opt_init(devices)
    
    # Start meta-training
    loss_arr = []
    for epoch in range(n_iter):
        t0 = time.time()
        key, key_device, key_eval = random.split(key, 3)
        sX, sY, qX, qY = j_sample_sinusoid_task(key, batch_size=batch_train, 
                                                num_samples_per_task=task_size)
        opt_state, L, metrics = update_out(key_device, epoch, opt_state, sX, sY, qX, qY)
        if epoch % 100 == 0:
            loss_arr.append(L)
            print(f'Epoch: {epoch} - Loss: {L:.3f} - Time : {(time.time()-t0):.3f} s')
            wandb.log({'Outer loop loss':L})
            wandb.log({k:v[0] for (k,v) in metrics.items()})
    print('Meta-training completed.')


    # Evaluate fine tuning
    sX, sY, qX, qY = j_sample_sinusoid_task(key_eval, batch_size=batch_test, 
                                            num_samples_per_task=100)
    weights = get_params(opt_state)

    sX_t = sX[:,:task_size,:,:] # batch, number, time, dim
    sY_t = sY[:,:task_size,:,:]
    plt.figure(figsize=(10,4));
    c = ['slateblue', 'darkblue']
    for i in range(2): # Initial prediction followed by one-shot prediction
        z0, z1, sYhat = vmap(batch_task_predict, in_axes=(None, 0))(weights, sX_t)
        plt.plot(sX_t[0,:,-1,0], sYhat[0,:,-1,0], '.', label='Prediction '+str(i), color=c[i])
        weights, metrics = vmap(update_in, in_axes=(None, 0, 0, None))(weights, sX_t, sY_t, alpha)
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
    parser.add_argument('--batch_test', type=int, default=10, help='Batch size for meta-testing')
    parser.add_argument('--n_iter', type=int, default=20000, help='Number of iterations')
    parser.add_argument('--n_inp', type=int, default=1, help='Number of input neurons')
    parser.add_argument('--n_out', type=int, default=1, help='Number of output neurons')
    parser.add_argument('--n_h0', type=int, default=40, help='Number of neurons in the first hidden layer')
    parser.add_argument('--n_h1', type=int, default=40, help='Number of neurons in the second hidden layer')
    parser.add_argument('--task_size', type=int, default=20, help='Number of samples per task')
    parser.add_argument('--tau_mem', type=float, default=10e-3, help='Membrane time constant')
    parser.add_argument('--tau_out', type=float, default=1e-3, help='Output time constant')
    parser.add_argument('--lr_out', type=float, default=5e-4, help='Learning rate for the output layer')
    parser.add_argument('--alpha', type=float, default=1, help='Learning rate for the inner updates')
    parser.add_argument('--target_fr', type=float, default=2, help='Target firing rate')
    parser.add_argument('--lambda_fr', type=float, default=0, help='Weight of the firing rate loss')

    args = parser.parse_args()

    wandb.init(project='meta-analog', config=args)
    wandb.config.update(args)

    train_meta_analog(key=random.PRNGKey(args.seed),
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
                      lr_out=args.lr_out,
                      alpha=args.alpha,
                      target_fr=args.target_fr,
                      lambda_fr=args.lambda_fr)
