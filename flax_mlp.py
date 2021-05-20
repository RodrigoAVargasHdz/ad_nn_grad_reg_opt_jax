# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax import numpy as jnp, random, lax, vmap
import flax
from flax import linen
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, compact, Dense, LayerNorm, initializers
from flax.linen import swish as silu
from flax.linen import sigmoid

#import numpy as np
#from pprint import pprint
#from dense import Dense

from jax.config import config
jax.config.update('jax_enable_x64', True)

# Require JAX omnistaging mode.
jax.config.enable_omnistaging()

class NN_adiab(Module):
  n_atoms: Iterable[int]
  sizes: Iterable[int]
  
  dtype: Any = jnp.float32

  @compact
  def __call__(self, x):

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
    
    x = vmap(f_bond_length)(x)
                             
#     NN
    for size in self.sizes[:-1]:
        x = Dense(size)(x)#,dtype=jnp.float64
        x = linen.relu(x)
#         x = silu(x)
#         x = jnp.tanh(x)
#         x = linen.relu(x)
    x = Dense(self.sizes[-1])(x) #,dtype=jnp.float64
    
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
    
    x = vmap(f_adiab,(0))(x)
    return x

class NN_diab(Module):
  n_atoms: Iterable[int]
  sizes: Iterable[int]
  
  dtype: Any = jnp.float32

  @compact
  def __call__(self, x):

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
    
    x = vmap(f_bond_length)(x)
                             
#     NN
    for size in self.sizes[:]:
        x = Dense(size)(x)
        x = jnp.tanh(x)
#         x = LayerNorm()(x)
    
    return x

class NN_ad_coup(Module):
  n_atoms: Iterable[int]
  sizes: Iterable[int]
  
  dtype: Any = jnp.float32

  @compact
  def __call__(self, x):

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
    
    x = vmap(f_bond_length)(x)
                             
#     NN
    for size in self.sizes[:]:
        x = Dense(size)(x)
        x = jnp.tanh(x)
#         x = LayerNorm()(x)
    
#     Adiabatic energies
    def f_adiab(x):
        w00 = x[0]
        w11 = x[1]
        w01 = x[2]
        W = jnp.diag(jnp.array([w00,w11]))
        W = W.at[0,1].set(w01)
        W = W.at[1,0].set(w01)
        w,_ = jnp.linalg.eigh(W)
        return jnp.stack((w,x[-1]))
    
    x = vmap(f_adiab,(0))(x)
    return x

#
class MLP(Module):
  sizes: Iterable[int]

  @compact
  def __call__(self, x):
    
    for size in self.sizes[:-1]:
        x = Dense(size)(x)
        x = jnp.tanh(x)
#         x = flax.linen.sigmoid(x)
    return Dense(self.sizes[-1])(x)
    
'''
class NN_cross_prod(Module):
  n_atoms: Iterable[int] 
  sizes: Iterable[int]
  
  dtype: Any = jnp.float32

  @compact
  def __call__(self, x):
                        
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
        x = jnp.reshape(x,(self.n_atoms,3))
        a = x[0,:] - x[1,:]
        b = jnp.cross(x[0,:] - x[2,:],x[0,:] - x[3,:])
        c = jnp.dot(a,b)
        r = f_bond_length(x)
        return c/jnp.prod(r)

#     dot-cross product
    q_NHHH = vmap(dot_cross_product)(x)
        
#     NN
    for size in self.sizes[:]:
        x = Dense(size)(x)
        x = jnp.tanh(x)
#         x = LayerNorm()(x)
    
#     Adiabatic energies
    def f_adiab(x,q_NHHH):
        w00 = x[0]
        w11 = x[1]
        w01 = q_NHHH * x[2] + (q_NHHH**3)*x[3]
        W = jnp.diag(jnp.array([w00,w11]))
        W = W.at[0,1].set(w01)
        W = W.at[1,0].set(w01)
        w,_ = jnp.linalg.eigh(W)
        return w
    
    x = vmap(f_adiab,(0,0))(x,q_NHHH)
    return x
'''
