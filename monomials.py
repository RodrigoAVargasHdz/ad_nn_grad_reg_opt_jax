import jax 
import jax.numpy as jnp 

# File created from MOL_1_3_4.MONO 

# Total number of monomials = 33 

def f_monomials(r): 

    mono = jnp.zeros(33) 

    mono = mono.at[0].set(1.) 
    mono = mono.at[1].set(r[5]) 
    mono = mono.at[2].set(r[4]) 
    mono = mono.at[3].set(r[3]) 
    mono = mono.at[4].set(r[2]) 
    mono = mono.at[5].set(r[1]) 
    mono = mono.at[6].set(r[0]) 
    mono = mono.at[7].set(mono[1] * mono[2]) 
    mono = mono.at[8].set(mono[1] * mono[3]) 
    mono = mono.at[9].set(mono[2] * mono[3]) 
    mono = mono.at[10].set(mono[3] * mono[4]) 
    mono = mono.at[11].set(mono[2] * mono[5]) 
    mono = mono.at[12].set(mono[1] * mono[6]) 
    mono = mono.at[13].set(mono[4] * mono[5]) 
    mono = mono.at[14].set(mono[4] * mono[6]) 
    mono = mono.at[15].set(mono[5] * mono[6]) 
    mono = mono.at[16].set(mono[1] * mono[9]) 
    mono = mono.at[17].set(mono[1] * mono[10]) 
    mono = mono.at[18].set(mono[2] * mono[10]) 
    mono = mono.at[19].set(mono[1] * mono[11]) 
    mono = mono.at[20].set(mono[3] * mono[11]) 
    mono = mono.at[21].set(mono[2] * mono[12]) 
    mono = mono.at[22].set(mono[3] * mono[12]) 
    mono = mono.at[23].set(mono[2] * mono[13]) 
    mono = mono.at[24].set(mono[3] * mono[13]) 
    mono = mono.at[25].set(mono[1] * mono[14]) 
    mono = mono.at[26].set(mono[3] * mono[14]) 
    mono = mono.at[27].set(mono[1] * mono[15]) 
    mono = mono.at[28].set(mono[2] * mono[15]) 
    mono = mono.at[29].set(mono[4] * mono[15]) 
    mono = mono.at[30].set(mono[2] * mono[24]) 
    mono = mono.at[31].set(mono[1] * mono[26]) 
    mono = mono.at[32].set(mono[1] * mono[28]) 

    return mono 



