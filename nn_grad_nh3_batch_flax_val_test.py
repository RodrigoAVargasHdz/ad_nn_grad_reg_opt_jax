import os
import argparse
import time
import datetime
import itertools
import numpy.random as onpr

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from jax import value_and_grad, grad, jacfwd, jacrev
# from jax.experimental import optimizers

from jax.config import config
config.update("jax_debug_nans", True)
jax.config.update('jax_enable_x64', True)

from flax.core.frozen_dict import freeze, unfreeze,FrozenDict
from flax import serialization, jax_utils
from flax import linen as nn
from flax import optim

#ravh 
from data import *
from flax_mlp import *

# mase to jax
# from monomials import f_monomials as f_mono
# from polynomials import f_polynomials as f_poly

Ha2cm = 220000

r_dir = 'Results_nn_adiab'

# --------------------------------
def load_data(file_results,N,l):
    if os.path.isfile(file_results):
        D = jnp.load(file_results,allow_pickle=True)
        Dtr = D.item()['Dtr']
        Dval = D.item()['Dval']
        Dt = Data_nh3()
    else:
        Dtr,Dval,Dt = split_trainig_val_test_coup(N,l)
        Xtr,gXtr,gXctr,ytr = Dtr
        Dtr = (Xtr,gXtr,gXctr,ytr)
        
    return Dtr,Dval,Dt    
# ---------------------------------------------
def main_opt(N,l,i0,nn_arq,act_fun,n_epochs,lr,w_decay,rho_g):

    start_time = time.time()

    str_nn_arq = ''
    for item in nn_arq:
        str_nn_arq = str_nn_arq + '_{}'.format(item)
        
    f_job = 'nn_arq{}_N_{}_i0_{}_l_{}_batch'.format(str_nn_arq,N,i0,l)
    f_out = '{}/out_opt_{}.txt'.format(r_dir,f_job)
    f_w_nn = '{}/W_{}.npy'.format(r_dir,f_job)
    file_results = '{}/data_nh3_{}.npy'.format(r_dir,f_job)

#     --------------------------------------    
#     Data
    n_atoms = 4
    batch_size = 768 #1024#768#512#256#128#64#32
    Dtr,Dval,Dt = load_data(file_results,N,l)
    Xtr,gXtr,gXctr,ytr = Dtr
    Xval,gXval,gXcval, yval = Dval
    Xt,gXt,gXct,yt = Dt
    print(gXtr.shape,gXtr.shape,gXctr.shape,ytr.shape)
# --------------------------------
#     BATCHES

    n_complete_batches, leftover = divmod(N, batch_size)
    n_batches = n_complete_batches + bool(leftover)
    
    def data_stream():
        rng = onpr.RandomState(0)
        while True:
            perm = rng.permutation(N)
            for i in range(n_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield Xtr[batch_idx],gXtr[batch_idx], gXctr[batch_idx], ytr[batch_idx]
    batches = data_stream()
# --------------------------------

    f = open(f_out,'a+')
    print('-----------------------------------',file=f)
    print('Starting time', file=f)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
    print('-----------------------------------',file=f)
    print(f_out,file=f)
    print('N = {}, n_atoms = {}, data_random = {}, NN_random = {}'.format(N,n_atoms,l,i0),file=f)
    print(nn_arq,file=f)
    print('lr = {}, w decay = {}'.format(lr,w_decay),file=f)
    print('Activation function = {}'.format(act_fun),file=f)
    print('N Epoch = {}'.format(n_epochs),file=f)
    print('rho G = {}'.format(rho_g),file=f)
    print('-----------------------------------',file=f)
    f.close()

#     --------------------------------------    
#     initialize NN

    nn_arq.append(3) 
    tuple_nn_arq = tuple(nn_arq)
    nn_model = NN_adiab(n_atoms,tuple_nn_arq)
    
    def get_init_NN_params(key):
        x = Xtr[0,:]
        x = x[None,:]#         x = jnp.ones((1,Xtr.shape[1]))
        variables = nn_model.init(key, x)
        return variables

#     Initilialize parameters   
    rng = random.PRNGKey(i0)
    rng, subkey = jax.random.split(rng)
    params = get_init_NN_params(subkey)

    f = open(f_out,'a+')                
    if os.path.isfile(f_w_nn):
        print('Reading NN parameters from prev calculation!',file=f)
        print('-----------------------',file=f)   
        
        nn_dic = jnp.load(f_w_nn,allow_pickle=True)
        params = unfreeze(params) 
        params['params'] = nn_dic.item()['params']
        params = freeze(params)
#         print(params)

    f.close()
    init_params = params
    
#     --------------------------------------    
#     Phys functions
        
    @jit
    def nn_adiab(params,x):
        y_ad_pred = nn_model.apply(params,x)     
        return y_ad_pred 

    @jit
    def jac_nn_adiab(params,x):
      g_y_pred = jacrev(nn_adiab,argnums=1)(params,x[None,:])
      return jnp.reshape(g_y_pred,(2,g_y_pred.shape[-1]))

#     --------------------------------------    
#    training loss functions  
    @jit
    def f_loss_ad_energy(params,batch):
        X_inputs,_,_,y_true = batch
        y_pred = nn_adiab(params,X_inputs)
        diff_y = y_pred - y_true #Ha2cm*
        return jnp.linalg.norm(diff_y,axis=0)

    @jit
    def f_loss_jac(params,batch):
        X_inputs, gX_inputs,_,y_true = batch
        gX_pred = vmap(jac_nn_adiab,(None,0))(params,X_inputs)
        diff_g_X = gX_pred - gX_inputs
        # jnp.linalg.norm(diff_g_X,axis=0)
        
        diff_g_X0 = diff_g_X[:,0,:]
        diff_g_X1 = diff_g_X[:,1,:]
        l0 = jnp.linalg.norm(diff_g_X0)
        l1 = jnp.linalg.norm(diff_g_X1)
        return jnp.stack([l0,l1]) 

   #     ------
    @jit
    def f_loss(params,rho_g,batch):
        rho_g = jnp.abs(rho_g)
        loss_ad_energy = f_loss_ad_energy(params,batch)
        loss_jac_energy = f_loss_jac(params,batch)
        loss = jnp.vdot(jnp.ones_like(loss_ad_energy),loss_ad_energy) + jnp.vdot(rho_g,loss_jac_energy)
        return loss
#     --------------------------------------
#     Optimization  and Training   

#     Perform a single training step.

    @jit
    def train_step(optimizer,rho_g,batch):#, learning_rate_fn, model
        grad_fn = jax.value_and_grad(f_loss)
        loss, grad = grad_fn(optimizer.target,rho_g,batch)
        optimizer = optimizer.apply_gradient(grad) #, {"learning_rate": lr}
        return optimizer, (loss, grad)

#     @jit
    def train(rho_g,nn_params):
        optimizer = optim.Adam(learning_rate=lr,weight_decay=w_decay).create(nn_params)
        optimizer = jax.device_put(optimizer)    
           
        train_loss = []  
        loss0 = 1E16
        loss0_tot = 1E16
        itercount = itertools.count()
        f_params = init_params
        for epoch in range(n_epochs):
            for _ in range(n_batches):
                optimizer, loss_and_grad = train_step(optimizer,rho_g,next(batches))
                loss, grad = loss_and_grad
                
#             f = open(f_out,'a+')
#             print(i,loss,file=f)
#             f.close()
            
            train_loss.append(loss)
#             params = optimizer.target
#             loss_tot = f_validation(params)
        
        nn_params = optimizer.target

        return nn_params,loss_and_grad, train_loss

    @jit
    def val_step(optimizer,nn_params):#, learning_rate_fn, model
    
        rho_g_prev = optimizer.target
        nn_params, loss_and_grad_train, train_loss_iter = train(rho_g_prev,nn_params)
        loss_train, grad_loss_train = loss_and_grad_train
        
        grad_fn_val = jax.value_and_grad(f_loss, argnums=1)
        loss_val, grad_val = grad_fn_val(nn_params,optimizer.target,Dval)
        optimizer = optimizer.apply_gradient(grad_val) #, {"learning_rate": lr}
        return optimizer, nn_params, (loss_val,loss_train,train_loss_iter), (grad_loss_train,grad_val)

 #     Initilialize rho_G   
    rng = random.PRNGKey(0)
    rng, subkey = jax.random.split(rng)
        
    rho_G0 = random.uniform(subkey,shape=(2,),minval=-1, maxval=0.01)

    init_G = rho_G0#
           
    optimizer_out = optim.Adam(learning_rate=2E-3,weight_decay=0.).create(init_G)
    optimizer_out = jax.device_put(optimizer_out)    
    
    f_params = init_params
   
    for i in range(15000):
        start_va_time = time.time()
        optimizer_out, f_params, loss_all, grad_all = val_step(optimizer_out, f_params)
        rho_g = optimizer_out.target
        loss_val,loss_train,train_loss_iter = loss_all
        grad_loss_train,grad_val = grad_all
        
        loss0_tot = f_loss(f_params,rho_g,Dt)
        
        dict_output = serialization.to_state_dict(f_params)
        jnp.save(f_w_nn,dict_output)#unfreeze()
        
        f = open(f_out,'a+')
#         print(i,rho_g, loss0, loss0_tot, (time.time() - start_va_time),file=f)
        print(i,loss_val,loss_train, (time.time() - start_va_time),file=f)
        print(jnp.exp(rho_g), file=f)
        print(grad_val, file=f)
#         print(train_loss_iter ,file=f) 
#         print(grad_val,file=f)
#         print(grad_loss_train,file=f)
        f.close()

#     --------------------------------------
#     Prediction
    f = open(f_out,'a+')
    print('Prediction of the entire data set', file=f)
    print('N = {}, n_atoms = {}, random = {}'.format(N,n_atoms,i0),file=f)
    print('NN : {}'.format(nn_arq),file=f)
    print('lr = {}, w decay = {}, rho G = {}'.format(lr,w_decay,rho_g),file=f)
    print('Activation function = {}'.format(act_fun),file=f)
    print('Total points  = {}'.format(yt.shape[0]),file=f)
    
    y_pred = nn_adiab(f_params, Xt)
    gX_pred = vmap(jac_nn_adiab,(None,0))(f_params,Xt)
    
    diff_y = y_pred - yt
    rmse_Ha = jnp.linalg.norm(diff_y)
    rmse_cm = jnp.linalg.norm(Ha2cm*diff_y)
    mae_Ha = jnp.linalg.norm(diff_y,ord=1)
    mae_cm = jnp.linalg.norm(Ha2cm*diff_y,ord=1)

    print('RMSE = {} [Ha]'.format(rmse_Ha),file=f)
    print('RMSE(tr) = {} [cm-1]'.format(loss0),file=f)
    print('RMSE = {} [cm-1]'.format(rmse_cm),file=f)
    print('MAE = {} [Ha]'.format(mae_Ha),file=f)
    print('MAE = {} [cm-1]'.format(mae_cm),file=f)

    Dpred = jnp.column_stack((Xt,y_pred))    
    data_dic = {'Dtr':Dtr,
            'Dpred': Dpred,
            'gXpred': gX_pred,
            'loss_tr':loss0,
            'error_full':rmse_cm,
            'N':N,
            'l':l,
            'i0':i0,
            'rho_g':rho_g}
          
    jnp.save(file_results,data_dic)
      
    print('---------------------------------', file=f)
    print('Total time =  %.6f seconds ---'% ((time.time() - start_time)), file=f)
    print('---------------------------------', file=f)
    f.close()


def main():
	parser = argparse.ArgumentParser(description='opt PIP-NN')
	parser.add_argument('--N', type=int, default=3500, help='initeger data')
	parser.add_argument('--l', type=int, default=0, help='training data label')
	parser.add_argument('--i', type=int, default=0, help='random integer NN')
	parser.add_argument('--f', type=str, default='tanh',help='activation function' )
	parser.add_argument('--lr', type=float, default=5E-4, help='learning rate')
	parser.add_argument('--wdecay', type=float, default=1E-3, help='weight decay')
	parser.add_argument('--lG', type=float, default=1E-2, help='lambda gradient')
	parser.add_argument('--n_epoch', type=int, default=5000, help='number of epochs')
	parser.add_argument('-nn', '--list', help='NN arq', type=str)
    
	args = parser.parse_args()
	i0 = args.i
	l = args.l
	N = args.N
	nn_arq = [int(item) for item in args.list.split(',')]
	act_fun = args.f
	lr = args.lr
	rho_g = args.lG
	w_decay = args.wdecay
	n_epochs = args.n_epoch
	
	main_opt(N,l,i0,nn_arq,act_fun,n_epochs,lr,w_decay,rho_g)
	
	'''
	lr_ = jnp.array([5E-5])#2E-3,
	for lr in lr_:
	    nn_arq = [int(item) for item in args.list.split(',')]
	    main_opt(N,l,i0,nn_arq,act_fun,n_epochs,lr,w_decay)    

    '''

if __name__ == "__main__":
    main()

'''
    start_time = time.time()
    loss_val = f_validation(params)
    loss_jac_val = f_jac_validation(params)
    loss_nac_val = f_nac_validation(params)
    print('Tot adiab energies loss = ', loss_val)
    print('Tot jac adiab energies loss = ', loss_jac_val)
    print('Tot nac loss = ', loss_nac_val)
    print('Total time =  %.6f seconds ---'% ((time.time() - start_time)))
    assert 0

#     norm_gXt = jnp.linalg.norm(gXt,axis=2)
#     j0 = jnp.argmax(norm_gXt,axis=0)
#     rho_g = 10.0*(jnp.amax(ytr))**2/jnp.vdot(norm_gXt[j0],norm_gXt[j0])

#     --------------------------------------    
#     Validation loss functions  
      
    @jit
    def f_validation(params):
        y_pred = nn_adiab(params, Xt)
        diff_y = y_pred - yt
        z = jnp.linalg.norm(diff_y)
        return z

    @jit
    def f_jac_validation(params):
        gX_pred = vmap(jac_nn_adiab,(None,0))(params,Xt)
        diff_y = gX_pred - gXt
        z = jnp.linalg.norm(diff_y)
        return z


'''
  