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

class NN_Coulomb_adiab(Module):
  n_atoms: Iterable[int] 
  sizes: Iterable[int]
  
  lambd_init: Callable = initializers.ones
  dtype: Any = jnp.float32

  @compact
  def __call__(self, x):
                        
    def f_Coulomb_Matrix(x):
        z_atoms = jnp.array([7.,1.,1.,1.])
        z_diag = 0.5*z_atoms**2.4
        M = jnp.multiply(z_atoms[:,None], z_atoms[None,:])
        M = M.at[jnp.diag_indices(self.n_atoms)].set(z_diag)  

        x = jnp.reshape(x,(self.n_atoms,3))
        r = x[:,None] - x[None,:]
        r = jnp.asarray(r)
        i0 = jnp.diag_indices(self.n_atoms,2)
        r = r.at[i0].set(1.) 
        r = jnp.linalg.norm(r,axis=2)
        r = 1./r 
        
        Z = jnp.multiply(M,r)
#         i0 = jnp.triu_indices(self.n_atoms,0)
#         Z[i0]
        return Z.ravel()

#     Adiabatic energies
    def f_adiab(x):
        w00 = x[0]
        w11 = x[1]
        w01 = x[2]
        W = jnp.diag(jnp.array([w00,w11]))
        W = W.at[0,1].set(w01)
        W = W.at[1,0].set(w01)
        w,_ = jnp.linalg.eigh(W)
        return w
#    ------------     
    x = vmap(f_Coulomb_Matrix)(x)        
#     NN
    for size in self.sizes[:-1]:
        x = Dense(size,dtype=jnp.float64)(x)
        x = linen.relu(x)
#         x = silu(x)
#         x = jnp.tanh(x)
#         x = linen.relu(x)
    x = Dense(self.sizes[-1],dtype=jnp.float64)(x) #new
    
    x = vmap(f_adiab)(x)
    return x

