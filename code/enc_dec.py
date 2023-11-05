import numpy as np
from basis import np_add, np_sub, np_mul, np_mul_vec_ele, np_mul_vec, np_mul_vec_mat
import time

def np_encrypt(n, k, q, u, f_id, mu, fp):
    #     arr = np.array(list(bin(message)[2:][::-1]), dtype='int64')
    
#     print(f"message = {mu}", file=fp)
    s = np.random.randint(-1, 2, size=n)
    e0 = np.random.randint(-1, 2, size=n)
     
        
    R = np.random.randint(-1, 2, size=(k, k, n))  
    y = np.random.randint(-1, 2, size=(k, n))
    yR = np_mul_vec_mat(y, R, q)
#     e1 = np.random.randint(-1, 2, size=(2 * k, n))
    e1 = np.concatenate((yR,yR),axis=0)
#     print(f"R={R}",file=fp)
#     print(f"y={y}",file=fp)
#     print(f"z={yR}",file=fp)
#     print(f"s={s}",file=fp)
#     print(f"x={e0}",file=fp)
    
    c0_input = np.zeros(n, dtype=np.int64)
    c0_input[0] = round(q / 2)
    c0 = np_add(
        np_mul(c0_input, mu, q),
        np_add(np_mul(u, s, q), e0, q), 
        q
    )
    
    c1 = ((np_mul_vec_ele(f_id, s, q) + e1) + q // 2) % q - q // 2
    return c0, c1, e1


def np_decrypt(n, q, e_id, c0, c1):
    mes = np_sub(c0, np_mul_vec(c1, e_id, q), q)
#     res = np.around(mes * 2 / q) % 2
#     res = res.astype(np.int32)
#     return res
    return np.array([round(mes[i] * 2 / q) % 2 for i in range(n)])
