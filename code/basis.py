import numpy as np
from copy import deepcopy
from reedsolo import RSCodec


def np_gen_poly(n, q):
    return np.random.randint(-q // 2, q // 2 + 1, size=n)


def np_mod_r(a, q):
#     b = deepcopy(a)

#     def f(x):
#         return (x + q // 2) % q - q // 2

#     g = np.vectorize(f)
#     return g(b)
    return (a + q // 2) % q - q // 2


def np_mod_r_inv(a, q):
#     b = deepcopy(a)

#     def f_inv(x):
#         return (x + q) % q

#     g = np.vectorize(f_inv)
#     return g(b)
    return (a + q) % q


def np_add(u, v, q):
    return np_mod_r(u + v, q)


def np_sub(u, v, q):
    return np_mod_r(u - v, q)


def np_mul(u, v, q):
    n = len(u)
    result = np.convolve(u, v)
    res = np.pad(result, (0, 2 * n - 1 - len(result)), mode='constant')
    for i in range(n - 1):
        res[i] = res[i] - res[i + n]
    return np_mod_r(res[:n], q)


def np_mul_vec_ele(u, v, q):
    # u is a vector [[1,2],[1,2]]
    return np.apply_along_axis(np_mul, 1, u, v, q)


def np_mul_vec(u, v, q):
    result = np.empty_like(u)
    for i, (x, y) in enumerate(zip(u, v)):
        result[i] = np_mul(x, y, q)
    return np_mod_r(np.sum(result, axis=0), q)


def np_mul_mat_ele(u, v, q):
    # u is a matrix [[[1,2],[1,2]],[[2,3],[2,3]]]
    m = len(u)
    n = len(u[0])
    l = len(u[0][0])
    w = np.zeros((m, n, l), dtype=object)
    for i in range(m):
        for j in range(n):
            w[i][j] = np_mul(u[i][j], v, q)
    return w


def np_mul_vec_mat(u, v, r):
    # vec:u, mat:v
    m = len(v[0])
    n = len(u[0])
    w = np.zeros((m, n), dtype=object)
    v_t = np.transpose(v, (1, 0, 2))
    for i in range(m):
        w[i] = np_mul_vec(u, v_t[i], r)
    return w


def np_mul_mat_vec(u, v, r):
    # mat:u, vec:v
    m = len(u[0])
    n = len(v[0])
    w = np.zeros((m, n), dtype=object)
    for i in range(m):
        w[i] = np_mul_vec(u[i], v, r)
    return w


def np_mul_mat(u, v, q):
    m = len(u)
    n = len(v[0])
    l = len(u[0][0])
    v_t = np.transpose(v, (1, 0, 2))
    w = np.zeros((m, n, l), dtype=object)
    for i in range(m):
        for j in range(n):
            w[i][j] = np_mul_vec(u[i], v_t[j], q)
    return w


def exgcd(a, b):
    if b == 0:
        return 1, 0, a
    else:
        x, y, r = exgcd(b, a % b)
        x, y = y, (x - (a // b) * y)
        return x, y, r


def gen_plain(n, i):
    m = [0] * n
    m[i] = 1
    return m


def gen_plain_inv(n, i):
    # x^{-beta}
    m = [0] * n
    if i == 0:
        m[0] = 1
    else:
        m[n - i] = -1
    return m


def m_ary(q, m, eta):
    # decompose q into m-ary, output a list, length is eta
    div = q  # dividend
    res = []
    while div > 0:
        rem = div % m  # reminder
        res.append(rem)
        div = div // m
    while len(res) < eta:
        res.append(0)
    return res


def func(z, L, p_mod):
    l = len(z)  # l:  id length
    # ECC(id), length = L + 1, ECC(id)[0]=0
    rsc = RSCodec(L - l)
    codeword = rsc.encode(z)
    for i in range(len(codeword)):
        codeword[i] = codeword[i] % p_mod
    return [0] + list(codeword)
