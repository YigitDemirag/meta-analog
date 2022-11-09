''' 
Model Agnostic Meta-Learning (MAML) training for 
spiking neural networks with memristive synapses for 
adaptation on the edge computing.

Offline training     New task arrives        New task performance
t=0.1s               t=100s      t=101s      t=151s
[deployement] ------ [t_read] -- [t_prog] -- [t_test]

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
from utils import sample_sinusoid_task, ls_than, gr_than, analog_init
from analog import read, write, zero_time_dim, GMIN, GMAX

def train_meta_analog(key, batch_train, batch_test, n_iter, n_inp,
                      n_out, n_h0, n_h1, task_size, tau_mem, tau_out,
                      lr_out, t_read, t_prog, t_wait, t_test, target_fr, 
                      lr_drop, lambda_fr, grad_thr, perf):

    def net_step(h, x_t):
        h, out_t = lif_forward(h, x_t)
        return h, out_t

    def task_predict(weights, X):
        _, net_const, net_dyn = analog_init(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out)
        _, out = scan(net_step, [weights, net_const, net_dyn], X, length=100)
        return out

    batch_task_predict = vmap(task_predict, in_axes=(None, 0))

    # TODO: Learn threshold
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
 
    def update_in(devices, key, sX, sY): 
        # TODO: Only output loop update
        key_rp, key_rn, key_wp, key_wn = random.split(key, 4)

        # Get G+ and G- devices
        pos_devs = tree_map(lambda dev: tree_map(lambda G: G[0], dev), devices)
        neg_devs = tree_map(lambda dev: tree_map(lambda G: G[1], dev), devices)

        # Effective synaptic weights at t=t_read
        theta = [(read(key_rp, dp, t=t_read, perf=perf)-read(key_rn, dn, t=t_read, perf=perf)) 
                  / (GMAX-GMIN) for dp, dn in zip(pos_devs, neg_devs)]
        
        # Calculate gradients 
        value, grads_in = value_and_grad(inner_loss, has_aux=True)(theta, sX, sY)

        # Calculate grad masks
        pos_grad_mask  = tree_map(lambda grads: ls_than(grads, -grad_thr), grads_in)
        neg_grad_mask  = tree_map(lambda grads: gr_than(grads, grad_thr), grads_in)
 
        # Extend grad mask pytree
        pos_grad_mask = [{'G':grad_mask, 'tp':grad_mask} for grad_mask in pos_grad_mask]
        neg_grad_mask = [{'G':grad_mask, 'tp':grad_mask} for grad_mask in neg_grad_mask]

        # Device program with inner loop update
        updated_pos_devs = tree_map(lambda dev, up_mask: 
                                    write(key_wp, dev, up_mask, tp=t_prog, perf=perf),
                                    pos_devs, pos_grad_mask, is_leaf=lambda x: isinstance(x, dict))

        updated_neg_devs = tree_map(lambda dev, up_mask: 
                            write(key_wn, dev, up_mask, tp=t_prog, perf=perf),
                            neg_devs, neg_grad_mask, is_leaf=lambda x: isinstance(x, dict))

        # Read device states at t=t_prog + t_wait
        updated_weights = [(read(key_rp, dp, t=t_prog + t_wait, perf=perf) - read(key_rn, dn, t=t_prog + t_wait, perf=perf)) 
                           / (GMAX-GMIN) for dp, dn in zip(updated_pos_devs, updated_neg_devs)]

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


    def maml_loss(devices, key, sX, sY, qX, qY):
        updated_weights, metrics = update_in(devices, key, sX, sY)
        return outer_loss(updated_weights, qX, qY), metrics

    def batched_maml_loss(devices, key, sX, sY, qX, qY):
        task_losses, metrics = vmap(maml_loss, in_axes=(None, None, 0, 0, 0, 0))(devices, key, sX, sY, qX, qY)
        return jnp.mean(task_losses), metrics

    @jit
    def update_out(key, i, opt_state, sX, sY, qX, qY):
        devices = get_params(opt_state)
        devices = zero_time_dim(devices)
        (L, metrics), grads_out = value_and_grad(batched_maml_loss, has_aux=True)(devices, key, sX, sY, qX, qY)

        metrics = {k:jnp.mean(v, axis=0) for (k,v) in metrics.items()}
        metrics['Outer loss'] = L
        metrics['grad-out-0'] = grads_out[0]['G']
        metrics['grad-out-1'] = grads_out[2]['G']
        metrics['grad-out-2'] = grads_out[4]['G']
        opt_state = opt_update(i, grads_out, opt_state)
        return opt_state, metrics

    piecewise_lr = optimizers.piecewise_constant([lr_drop], [lr_out, lr_out/10])
    opt_init, opt_update, get_params = optimizers.adam(step_size=piecewise_lr)
    devices, _, _ = analog_init(key, n_inp, n_h0, n_h1, n_out, tau_mem, tau_out)
    opt_state = opt_init(devices)
    
    # Start meta-training
    for epoch in range(n_iter):
        t0 = time.time()
        key, key_device, key_eval = random.split(key, 3)
        sX, sY, qX, qY = sample_sinusoid_task(key, batch_size=batch_train, 
                                              num_samples_per_task=task_size)
        opt_state, metrics = update_out(key_device, epoch, opt_state, sX, sY, qX, qY)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch} - Loss: {metrics["Outer loss"]:.3f} - Time : {(time.time()-t0):.3f} s')
            wandb.log(metrics)
    print('Meta-training completed.')


    # Evaluate fine tuning
    key_eval, key_data, key_dev = random.split(key_eval, 3)
    sX, sY, qX, qY = sample_sinusoid_task(key_data, batch_size=batch_test, 
                                          num_samples_per_task=100)
    devices = get_params(opt_state)
    key_rp, key_rn, key_w = random.split(key_dev, 3)

    # Get G+ and G- devices
    pos_devs = tree_map(lambda dev: tree_map(lambda G: G[0], dev), devices)
    neg_devs = tree_map(lambda dev: tree_map(lambda G: G[1], dev), devices)

    # Effective synaptic weights at t=t_read
    theta = [(read(key_rp, dp, t=t_test, perf=perf)-read(key_rn, dn, t=t_test, perf=perf)) 
                / (GMAX-GMIN) for dp, dn in zip(pos_devs, neg_devs)]

    sX_t = sX[:,:task_size,:,:] # batch, number, time, dim
    sY_t = sY[:,:task_size,:,:]
    plt.figure(figsize=(10,4));
    c = ['slateblue', 'darkblue']
    for i in range(2): # Initial prediction followed by one-shot prediction
        _, _, sYhat = vmap(batch_task_predict, in_axes=(None, 0))(theta, sX_t)
        plt.plot(sX_t[0,:,-1,0], sYhat[0,:,-1,0], '.', label='Prediction '+str(i), color=c[i])
        theta, metrics = vmap(update_in, in_axes=(None, None, 0, 0))(devices, key_w, sX_t, sY_t)
        theta = tree_map(lambda W: jnp.mean(W, axis=0), theta)

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
    parser.add_argument('--lr_out', type=float, default=1e-2, help='Learning rate for the output layer')
    parser.add_argument('--tread', type=float, default=100, help='New task read time')
    parser.add_argument('--tprog', type=float, default=101, help='New task programming time')
    parser.add_argument('--twait', type=float, default=50, help='New task optimized target time')
    parser.add_argument('--ttest', type=float, default=251, help='New task test time') 
    parser.add_argument('--target_fr', type=int, default=2, help='Target firing rate')
    parser.add_argument('--lr_drop', type=int, default=20000, help='The step number for dropping the learning rate')
    parser.add_argument('--lambda_fr', type=float, default=0, help='Regularization parameter for the firing rate')
    parser.add_argument('--grad_thr', type=float, default=1, help='Threshold for the gradient value for init update')
    parser.add_argument('--perf', action='store_true', help='Enable performance mode')
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
                      t_read=args.tread, 
                      t_prog=args.tprog,
                      t_wait=args.twait,
                      t_test=args.ttest,
                      target_fr=args.target_fr,
                      lr_drop=args.lr_drop,
                      lambda_fr=args.lambda_fr,
                      grad_thr=args.grad_thr,
                      perf=args.perf)
