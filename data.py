import numpy as onp

import jax
import jax.numpy as jnp

from jax.config import config
config.update("jax_debug_nans", True)
jax.config.update('jax_enable_x64', True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

Ha2cm = 220000
# --------------------------------
def Data_nh3(bool_norm = False):
    X = jnp.load('Data_nh3/nh3_cgeom.npy')
    y = jnp.load('Data_nh3/nh3_energy.npy')
    gX0 = jnp.load('Data_nh3/nh3_grad_energy_state_0.npy')
    gX1 = jnp.load('Data_nh3/nh3_grad_energy_state_1.npy') 
    
    gXc = jnp.load('Data_nh3/nh3_grad_energy_state_1.npy')  
    
#     DELETE BAD POINT
    j0 = 1824 #BAD POINT
    X = onp.delete(X,j0, 0)
    y = onp.delete(y,j0, 0)
    gX0 = onp.delete(gX0,j0, 0)
    gX1 = onp.delete(gX1,j0, 0) 
    gXc = onp.delete(gXc,j0, 0)
  
    gX = jnp.concatenate((gX0[None],gX1[None]),axis=0)
    gX = jnp.reshape(gX,(gX.shape[1],gX.shape[0],gX.shape[2])) 
    
    if bool_norm:
        y = normalize(y,axis=0)
        print(y)
    
    return X, gX, gXc, y

# --------------------------------
def split_trainig_test(N=2000,ir=0, bool_save = False):
    X, gX, _, y = Data_nh3()
    gX0 = gX[:,0]
    gd = gX0.shape[1]
    gX1 = gX[:,1]
    d = X.shape[1]
    Xt = onp.column_stack((X,gX0))
    Xt = onp.column_stack((Xt,gX1))
    
    
#      represents the absolute number of test samples
    N_tst = y.shape[0] - N
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=N_tst, random_state=ir)
    
    X_train, gX_train = Xt_train[:,:d],Xt_train[:,d:]
    gX0_train,gX1_train = gX_train[:,:gd],gX_train[:,gd:]
    gX_train = onp.concatenate((gX0_train[None],gX0_train[None]),axis=0)
    
    gX_train = jnp.reshape(gX_train,(gX_train.shape[1],gX_train.shape[0],gX_train.shape[2]))
    
    X_test, gX_test = Xt_test[:,:d], Xt_test[:,d:]
    gX0_test,gX1_test = gX_test[:,:gd],gX_test[:,gd:]
    gX_test = onp.concatenate((gX0_test[None],gX0_test[None]),axis=0)

    print(X_train.shape,y_train.shape,gX_train.shape)
    
    
    if bool_save:
        jnp.save('Data_nh3/X_geom_train_N_{}_i0_{}_nh3.npy'.format(N,ir),X_train)
        jnp.save('Data_nh3/gX_geom_train_N_{}_i0_{}_nh3.npy'.format(N,ir),gX_train)
        jnp.save('Data_nh3/y_energy_train_N_{}_i0_{}_nh3.npy'.format(N,ir),y_train)

        jnp.save('Data_nh3/X_geom_test_N_{}_i0_{}_nh3.npy'.format(N,ir),X_test)
        jnp.save('Data_nh3/gX_geom_test_N_{}_i0_{}_nh3.npy'.format(N,ir),gX_test)
        jnp.save('Data_nh3/y_energy_test_N_{}_i0_{}_nh3.npy'.format(N,ir),y_test)    
    
    return (X_train,gX_train,y_train),(X,gX,y)#(X_test,y_test,gX_test)

def split_trainig_test_coup(N=2000,ir=0, bool_norm = False,bool_save = False):
    X, gX, gXc, y = Data_nh3(bool_norm)
    gX0 = gX[:,0]
    gd = gX0.shape[1]
    gX1 = gX[:,1]
    d = X.shape[1]
    Xt = onp.column_stack((X,gX0))
    Xt = onp.column_stack((Xt,gX1))
    Xt = onp.column_stack((Xt,gXc))

#      represents the absolute number of test samples
    N_tst = y.shape[0] - N
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=N_tst, random_state=ir)
    
    X_train, gX_train = Xt_train[:,:d],Xt_train[:,d:]
    gX0_train,gX1_train,gXc_train = gX_train[:,:gd],gX_train[:,gd:2*gd],gX_train[:,2*gd:]
    gX_train = onp.concatenate((gX0_train[None],gX0_train[None]),axis=0)
    
    gX_train = jnp.reshape(gX_train,(gX_train.shape[1],gX_train.shape[0],gX_train.shape[2]))
    
    X_test, gX_test = Xt_test[:,:d], Xt_test[:,d:]
    gX0_test,gX1_test,gXc_test = gX_test[:,:gd],gX_test[:,gd:2*gd],gX_test[:,2*gd:]
    gX_test = onp.concatenate((gX0_test[None],gX0_test[None]),axis=0)
    gX_test = jnp.reshape(gX_test,(gX_test.shape[1],gX_test.shape[0],gX_test.shape[2]))
    
    
    if bool_save:
        jnp.save('Data_nh3/X_geom_train_N_{}_i0_{}_nh3.npy'.format(N,ir),X_train)
        jnp.save('Data_nh3/gX_geom_train_N_{}_i0_{}_nh3.npy'.format(N,ir),gX_train)
        jnp.save('Data_nh3/y_energy_train_N_{}_i0_{}_nh3.npy'.format(N,ir),y_train)

        jnp.save('Data_nh3/X_geom_test_N_{}_i0_{}_nh3.npy'.format(N,ir),X_test)
        jnp.save('Data_nh3/gX_geom_test_N_{}_i0_{}_nh3.npy'.format(N,ir),gX_test)
        jnp.save('Data_nh3/y_energy_test_N_{}_i0_{}_nh3.npy'.format(N,ir),y_test)    
    
    return (X_train,gX_train,gXc_train,y_train),(X,gX,gXc,y)#(X_test,y_test,gX_test)

def split_trainig_val_test_coup(N=2000,ir=0, bool_norm = False):
    X, gX, gXc, y = Data_nh3(bool_norm)
    gX0 = gX[:,0]
    gd = gX0.shape[1]
    gX1 = gX[:,1]
    d = X.shape[1]
    Xt = onp.column_stack((X,gX0))
    Xt = onp.column_stack((Xt,gX1))
    Xt = onp.column_stack((Xt,gXc))

#      represents the absolute number of test samples
    N_tst = y.shape[0] - N
    Xt_train, Xt_test, y_train, y_test = train_test_split(Xt, y, test_size=N_tst, random_state=ir)
    
    X_train, gX_train = Xt_train[:,:d],Xt_train[:,d:]
    gX0_train,gX1_train,gXc_train = gX_train[:,:gd],gX_train[:,gd:2*gd],gX_train[:,2*gd:]
    gX_train = onp.concatenate((gX0_train[None],gX0_train[None]),axis=0)
    
    gX_train = jnp.reshape(gX_train,(gX_train.shape[1],gX_train.shape[0],gX_train.shape[2]))
    
    X_test, gX_test = Xt_test[:,:d], Xt_test[:,d:]
    gX0_test,gX1_test,gXc_test = gX_test[:,:gd],gX_test[:,gd:2*gd],gX_test[:,2*gd:]
    gX_test = onp.concatenate((gX0_test[None],gX0_test[None]),axis=0)
    gX_test = jnp.reshape(gX_test,(gX_test.shape[1],gX_test.shape[0],gX_test.shape[2]))
       
    
    return (X_train,gX_train,gXc_train,y_train),(X_test,gX_test,gXc_test,y_test),(X,gX,gXc,y)

def load_nh3_data(N=500,ir=0):
    
    X_train = jnp.load('Data_nh3/X_geom_train_N_{}_i0_{}_nh3.npy'.format(N,ir))
    gX_train = jnp.load('Data_nh3/gX_geom_train_N_{}_i0_{}_nh3.npy'.format(N,ir))
    y_train = jnp.load('Data_nh3/y_energy_train_N_{}_i0_{}_nh3.npy'.format(N,ir))

    X_test = jnp.load('Data_nh3/X_geom_test_N_{}_i0_{}_nh3.npy'.format(N,ir))
    gX_test = jnp.load('Data_nh3/gX_geom_test_N_{}_i0_{}_nh3.npy'.format(N,ir))
    y_test = jnp.load('Data_nh3/y_energy_test_N_{}_i0_{}_nh3.npy'.format(N,ir))    
    
    return (X_train,y_train,gX_train),(X_test,y_test,gX_test)
    
