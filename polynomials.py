import jax 
import jax.numpy as jnp 

from monomials import f_monomials as f_monos 


# File created from MOL_1_3_4.POLY 


# Total number of monomials = 51 

def f_polynomials(r): 

    mono = f_monos(r) 

    poly = jnp.zeros(51) 

    poly = poly.at[0].set(mono[0]) 
    poly = poly.at[1].set(mono[1] + mono[2] + mono[3]) 
    poly = poly.at[2].set(mono[4] + mono[5] + mono[6]) 
    poly = poly.at[3].set(mono[7] + mono[8] + mono[9]) 
    poly = poly.at[4].set(mono[10] + mono[11] + mono[12]) 
    poly = poly.at[5].set(poly[1] * poly[2] - poly[4]) 
    poly = poly.at[6].set(mono[13] + mono[14] + mono[15]) 
    poly = poly.at[7].set(poly[1] * poly[1] - poly[3] - poly[3]) 
    poly = poly.at[8].set(poly[2] * poly[2] - poly[6] - poly[6]) 
    poly = poly.at[9].set(mono[16]) 
    poly = poly.at[10].set(mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22]) 
    poly = poly.at[11].set(poly[2] * poly[3] - poly[10]) 
    poly = poly.at[12].set(mono[23] + mono[24] + mono[25] + mono[26] + mono[27] + mono[28]) 
    poly = poly.at[13].set(poly[1] * poly[6] - poly[12]) 
    poly = poly.at[14].set(mono[29]) 
    poly = poly.at[15].set(poly[1] * poly[3] - poly[9] - poly[9] - poly[9]) 
    poly = poly.at[16].set(poly[1] * poly[4] - poly[10]) 
    poly = poly.at[17].set(poly[2] * poly[7] - poly[16]) 
    poly = poly.at[18].set(poly[2] * poly[4] - poly[12]) 
    poly = poly.at[19].set(poly[1] * poly[8] - poly[18]) 
    poly = poly.at[20].set(poly[2] * poly[6] - poly[14] - poly[14] - poly[14]) 
    poly = poly.at[21].set(poly[1] * poly[7] - poly[15]) 
    poly = poly.at[22].set(poly[2] * poly[8] - poly[20]) 
    poly = poly.at[23].set(poly[9] * poly[2]) 
    poly = poly.at[24].set(mono[30] + mono[31] + mono[32]) 
    poly = poly.at[25].set(poly[3] * poly[6] - poly[24]) 
    poly = poly.at[26].set(poly[14] * poly[1]) 
    poly = poly.at[27].set(poly[9] * poly[1]) 
    poly = poly.at[28].set(poly[3] * poly[4] - poly[23]) 
    poly = poly.at[29].set(poly[1] * poly[10] - poly[23] - poly[28] - poly[23]) 
    poly = poly.at[30].set(poly[1] * poly[11] - poly[23]) 
    poly = poly.at[31].set(poly[1] * poly[12] - poly[25] - poly[24] - poly[24]) 
    poly = poly.at[32].set(poly[1] * poly[13] - poly[25]) 
    poly = poly.at[33].set(poly[4] * poly[5] - poly[25] - poly[31]) 
    poly = poly.at[34].set(poly[2] * poly[11] - poly[25]) 
    poly = poly.at[35].set(poly[4] * poly[6] - poly[26]) 
    poly = poly.at[36].set(poly[2] * poly[12] - poly[26] - poly[35] - poly[26]) 
    poly = poly.at[37].set(poly[2] * poly[13] - poly[26]) 
    poly = poly.at[38].set(poly[14] * poly[2]) 
    poly = poly.at[39].set(poly[3] * poly[3] - poly[27] - poly[27]) 
    poly = poly.at[40].set(poly[3] * poly[7] - poly[27]) 
    poly = poly.at[41].set(poly[1] * poly[16] - poly[28]) 
    poly = poly.at[42].set(poly[2] * poly[21] - poly[41]) 
    poly = poly.at[43].set(poly[1] * poly[18] - poly[33]) 
    poly = poly.at[44].set(poly[7] * poly[8] - poly[43]) 
    poly = poly.at[45].set(poly[6] * poly[6] - poly[38] - poly[38]) 
    poly = poly.at[46].set(poly[2] * poly[18] - poly[35]) 
    poly = poly.at[47].set(poly[1] * poly[22] - poly[46]) 
    poly = poly.at[48].set(poly[6] * poly[8] - poly[38]) 
    poly = poly.at[49].set(poly[1] * poly[21] - poly[40]) 
    poly = poly.at[50].set(poly[2] * poly[22] - poly[48]) 

    return poly 



