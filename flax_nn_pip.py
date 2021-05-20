import jax
from jax import numpy as jnp, random, lax, vmap
import flax
from flax import linen
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, compact, Dense, LayerNorm, initializers 
from flax.linen import swish as silu
from flax.linen import sigmoid

from monomials import f_monomials as f_mono
from polynomials import f_polynomials as f_poly



from jax.config import config
jax.config.update('jax_enable_x64', True)

# Require JAX omnistaging mode.
jax.config.enable_omnistaging()

class NN_pip_adiab(Module):
  n_atoms: Iterable[int] 
  sizes: Iterable[int]
  
  lambd_init: Callable = initializers.ones
  dtype: Any = jnp.float32

  @compact
  def __call__(self, x):
  
    l = self.param('lambd',
                        self.lambd_init, # Initialization function
                        (1, ))
    l = jnp.asarray(3.*l, self.dtype)
#     l = 3.
                        
    def f_bond_length(x):
#     reshape (n_atoms,3)
        x = jnp.reshape(x,(self.n_atoms,3))
#     compute all difference 
        z = x[:,None] - x[None,:]
#     select upper diagonal (LEXIC ORDER)
        i0 = jnp.triu_indices(self.n_atoms,1)
        diff = z[i0]
#     compute the bond length
        r = jnp.linalg.norm(diff,axis=1)
        return r

    def dot_cross_product(x):
#     (R_N - R_H1)dot-prod [(R_N - R_H2)cross-prod(R_N - R_H3)]/ (r_NH1 * r_NH2 * r_NH3)

#         r = f_bond_length(x)
#         d = jnp.prod(r[:3]) # r_NH1 * r_NH2 * r_NH3
        
        x = jnp.reshape(x,(self.n_atoms,3))       
        R_nh1 = x[0,:] - x[1,:]
        R_nh1 = R_nh1/jnp.linalg.norm(R_nh1)
        R_nh2 = x[0,:] - x[2,:]
        R_nh2 = R_nh2/jnp.linalg.norm(R_nh2)
        R_nh3 = x[0,:] - x[3,:]
        R_nh3 = R_nh3/jnp.linalg.norm(R_nh3)
        
        b = jnp.cross(R_nh2,R_nh3)
        c = jnp.dot(R_nh1,b)
        
        return c#/d
        
    def f_morse(x):
        x = f_bond_length(x) # internuclear-distances
        x = jnp.exp(-x/l) # morse variables 
#         x = 1./x # inv. distance
        x = f_poly(x) # PIP
        return x

    q_NHHH = vmap(dot_cross_product)(x)
    x = vmap(f_morse)(x)
        
#     NN
    for size in self.sizes[:-1]:
        x = Dense(size,dtype=jnp.float64)(x)
        x = linen.relu(x)
#         x = silu(x)
#         x = jnp.tanh(x)
#         x = linen.relu(x)
    x = Dense(self.sizes[-1],dtype=jnp.float64)(x) #new

#     Adiabatic energies
    def f_adiab(x,q_NHHH):#
        w00 = x[0]
        w11 = x[1]
        w01 = x[2]*q_NHHH
        W = jnp.diag(jnp.array([w00,w11]))
        W = W.at[0,1].set(w01)
        W = W.at[1,0].set(w01)
        w,_ = jnp.linalg.eigh(W)
        return w
    
    x = vmap(f_adiab,(0,0))(x,q_NHHH)#,q_NHHH
    return x

