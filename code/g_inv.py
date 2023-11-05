import numpy as np
from basis import np_mod_r_inv, np_gen_poly, np_mul_mat


def np_g_inv(n, k, q, u):
    # r = g^{-1}(u), u is a ring element
    u_pos = np_mod_r_inv(u, q)  # (-q/2, q/2)->(0, q)

    def f(x):
        y = bin(x)[2:][::-1]
        return y

    vf = np.vectorize(f)

    bin_decompose = vf(u_pos)

    def deal_str(s, k):
        arr = np.array(list(s), dtype='int64')
        return np.pad(arr, (0, k - len(arr)), mode='constant')

    new = [deal_str(bin_decompose[i], k) for i in range(n)]
    return np.stack(new, axis=1)


def np_g_inv_vec(n, k, q, u_vec):
    # r = g^{-1}(u), u is a ring vector
    m = len(u_vec)
    arr = []
    for i in range(m):
        arr.append(np_g_inv(n, k, q, u_vec[i]))
    return np.stack(arr, axis=1)


def np_g_inv_mat(n, k, q, u_mat):
    new = u_mat.reshape(4, k, n)
    r0 = np_g_inv_vec(n, k, q, new[0])
#     print(np_mul_mat_vec(r0, g, q))
#     print(new[0])
    r1 = np_g_inv_vec(n, k, q, new[1])
    r2 = np_g_inv_vec(n, k, q, new[2])
    r3 = np_g_inv_vec(n, k, q, new[3])
    return np.vstack((np.hstack((r0, r1)), np.hstack((r2, r3))))


def np_gen_g(n, k):
    arr_list = []
    for j in range(k):
        a = np.zeros(n, dtype='int64')
        a[0] = 2 ** j
        arr_list.append(a)
    return np.array(arr_list, ndmin=2)


def test_g_inv_mat(n, k, q, G):
    arr = []
    for j in range(4*k):
        arr.append(np_gen_poly(n, q))
    U = np.array(arr)
    U.reshape((2, 2 * k, n))
    print(U)

    R = np_g_inv_mat(n, k, q, U)
    U_ = np_mul_mat(G, R, q)
    print(U_)