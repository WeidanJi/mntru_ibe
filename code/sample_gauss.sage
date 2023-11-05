import numpy as np
from basis import np_mul_vec, np_sub
import time

# def np_sam_gauss(n, k, B, mu, e):
    
    # Qx.<x> = QQ[]
    # S = Qx.quotient(x^n+1)
    # v = vector(S, k)
    # e = vector([S(list(e[i])) for i in range(len(e))])
    # G_Q = matrix(S, B_gs)
    # B_matQ = matrix(S, B)
    # for i in range(k - 1, -1, -1):
    #    ci = e.dot_product(G_Q[i])/(G_Q[i].dot_product(G_Q[i]))
    #    ci = S([round(list(ci)[i]) for i in range(n)])
    #    e = e - ci * B_matQ[i]
    #    v = v + ci * B_matQ[i]
    
    # return np.array([list(v[i]) for i in range(len(v))])
    
    
def round_pol(n, pol):
    # help(type(pol))
    # pol = list(pol)
    return [round(pol[i]) for i in range(n)]
    
def sam_gauss_new(n, k, mu, v):
    Qx.<x> = QQ[]
    S = Qx.quotient(x^n + 1)
    z = vector(S, k + 1)
    for j in range(k, -1, -1):
        if j<k: 
            #sum_mu = S([0] * n)
            #for i in range(j+1, k+1):
            #    sum_mu = (v[i]-z[i])*mu[i][j] + sum_mu
            #z[j] = S(round_pol(n, v[j]+sum_mu))
            
            
            z[j] = S(
                round_pol(
                    n, 
                    v[j]+
                    sum(
                        (v[i]-z[i])*mu[i][j] 
                         for i in range(j+1, k+1)
                    )
                )
            )
        else:    
            z[j] = S(round_pol(n, v[j]))
    return z

def np_sam_pre(n, k, B, mu, inv_B, y):
    # t = np.zeros((k + 1, n), dtype='int64')
    # t[0] = y
    Qx.<x> = QQ[]
    S = Qx.quotient(x^n + 1)
    
    # ts = vector([S(list(t[i])) for i in range(len(t))])
    
    # ts = zero_vector(S, k+1)
    # ts[0] = S(list(y))
    
    
    
    t = np.zeros((k + 1, n), dtype='int64')
    t[0] = y
    
    ts = vector([S(list(t[i])) for i in range(len(t))])
    
    v = ts * inv_B
    
    z = sam_gauss_new(n, k, mu, v)
    samp_vec = (v-z)*B
    
    return np.array([list(samp_vec[i]) for i in range(len(v))])


def np_sam_l(n, k, q, B, mu, inv_B, u, h_id):
    e2 = np.random.randint(-1, 1, size=(k, n))
    h_e2 = np_mul_vec(h_id, e2, q)
    y = np_sub(u, h_e2, q)
    e1 = np_sam_pre(n, k, B, mu, inv_B, y)
    
    e_id = np.vstack((e1, e2))
    return e_id[1:]