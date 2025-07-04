import random, sys
import multiprocessing

from cmath import cos, exp, pi, sin, sqrt
from jax.scipy.linalg import expm
# from scipy.linalg import expm
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg as ssla
import scipy

import numpy as np
from numpy import log
from numpy.linalg import matrix_power
np.set_printoptions(precision=6)
FLOATING_POINT_PRECISION = 1e-10

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import matplotlib.pyplot as plt

# from quantum_simulation_recipe.bounds import *

def expH(H, t, use_jax=False):
    # # check H is Hermitian
    # if not np.allclose(H, H.conj().T):
    #     raise ValueError('H is not Hermitian')

    if use_jax: 
        if isinstance(H, np.ndarray):
            return jax.scipy.linalg.expm(-1j * t * H)
        else:
            return jax.scipy.linalg.expm(-1j * t * H.to_matrix())
    elif isinstance(H, csr_matrix):
        return scipy.sparse.linalg.expm(-1j * t * H)
    else:
        return scipy.linalg.expm(-1j * t * H)


def pf(h_list, t, r: int, order: int=2, use_jax=False, return_exact=False, verbose=False):
# def pf_r(h_list, t, r: int, order: int=2, use_jax=False, return_exact=False, verbose=False):
    ## if use_jax=False, maybe encounter weired error for n=10. 
    # If your computer support Jax, please set use_jax=True 
    if order == 1:
        list_U = [expH(herm, t/r, use_jax=use_jax) for herm in h_list]
        # appro_U_dt = np.linalg.multi_dot(list_U)
        appro_U_dt = sparse_multi_dot(list_U)
        if isinstance(appro_U_dt, csr_matrix):
            appro_U = appro_U_dt**r
        else:
            if use_jax:
                appro_U = jnp.linalg.matrix_power(appro_U_dt, r)
            else:
                appro_U = np.linalg.matrix_power(appro_U_dt, r)
    elif order == 2:
        list_U = [expH(herm, t/(2*r), use_jax=use_jax) for herm in h_list]
        if verbose: print('----expm Herm finished----')
        appro_U_dt_forward = sparse_multi_dot(list_U)
        appro_U_dt_reverse = sparse_multi_dot(list_U[::-1])
        # appro_U_dt = list_U[0] @ list_U[1]
        if verbose: print('----matrix product finished----')
        if isinstance(appro_U_dt_forward, csr_matrix):
            appro_U = (appro_U_dt_forward @ appro_U_dt_reverse)**r
        else:
            if use_jax:
                appro_U = jnp.linalg.matrix_power(appro_U_dt_reverse @ appro_U_dt_forward, r)
            else:
                appro_U = np.linalg.matrix_power(appro_U_dt_reverse @ appro_U_dt_forward, r)
        if verbose: print('----matrix power finished----')
    else: 
        raise ValueError('higher order is not defined')

    if return_exact:
        exact_U = expH(sum(h_list), t, use_jax=use_jax)
        return appro_U, exact_U
    else:
        return appro_U

def pf_high(h_list, t: float, r: int, order: int, use_jax=False, verbose=False):
    dt = t/r
    if order != 1 and order != 2:
        # print('order: ', order) 
        u_p = 1/(4-4**(1/(order-1)))
        if verbose: print(u_p)
    if order == 1:
        pf1 = pf(h_list, t, r, order=1, use_jax=use_jax)
        return pf1
    elif order == 2:
        pf2 = pf(h_list, t, r, order=2, use_jax=use_jax)
        return pf2
    elif order == 4:
        pf2 = pf(h_list, u_p*dt, 1, use_jax=use_jax)
        if use_jax:
            pf2_2 = jnp.linalg.matrix_power(pf2, 2)
            pf4 = jnp.linalg.matrix_power(pf2_2 @ pf(h_list, (1-4*u_p)*dt, 1) @ pf2_2, r)
        else:
            pf2_2 = np.linalg.matrix_power(pf2, 2)
            pf4 = np.linalg.matrix_power(pf2_2 @ pf(h_list, (1-4*u_p)*dt, 1) @ pf2_2, r)
        # # be careful **r not work as matrix power
        # (pf(H_list, u_4*dt, 1)**2 @ pf(H_list, (1-4*u_4)*dt, 1) @ pf(H_list, u_4*dt, 1)**2)**r  
        return pf4
    elif order == 6:
        pf4 = pf_high(h_list, u_p*dt, 1, order=4, use_jax=use_jax)
        pf4_mid = pf_high(h_list, (1-4*u_p)*dt, 1, order=4, use_jax=use_jax)
        if use_jax:
            pf4_2 = jnp.linalg.matrix_power(pf4, 2)
            pf6 = jnp.linalg.matrix_power(pf4_2 @ pf4_mid @ pf4_2, r)
        else:
            pf4_2 = np.linalg.matrix_power(pf4, 2)
            pf6 = np.linalg.matrix_power(pf4_2 @ pf4_mid @ pf4_2, r)
        return pf6
    elif order == 8:
        pf6 = pf_high(h_list, u_p*dt, 1, order=6, use_jax=use_jax)
        pf6_mid = pf_high(h_list, (1-4*u_p)*dt, 1, order=6, use_jax=use_jax)
        if use_jax:
            pf6_2 = jnp.linalg.matrix_power(pf6, 2)
            pf8 = jnp.linalg.matrix_power(pf6_2 @ pf6_mid @ pf6_2, r)
        else:
            pf6_2 = np.linalg.matrix_power(pf6, 2)
            pf8 = np.linalg.matrix_power(pf6_2 @ pf6_mid @ pf6_2, r)
        return pf8
    else: 
        raise ValueError(f'higher order={order} is not defined')


# def pf(list_herm, order, t):
#     # print('order: ', order)
#     if order == 1:
#         return unitary_matrix_product(list_herm, t)
#     elif order == 2:
#         forward_order_product = unitary_matrix_product(list_herm, t/2) 
#         reverse_order_product = unitary_matrix_product(list_herm[::-1], t/2)
#         return forward_order_product @ reverse_order_product
#         # return second_order_trotter(list_herm, t)
#     elif order > 0 and order!= 1 and order != 2 and order % 2 == 0:
#         p = 1 / (4 - 4**(1/(order-1)))
#         # print('p: ', p)
#         return matrix_power(pf(list_herm, order-2, p*t), 2) @ pf(list_herm, order-2, (1-4*p)*t) @ matrix_power(pf(list_herm, order-2, p*t), 2)
#     else:
#         raise ValueError('k is not defined')

# matrix product of a list of matrices
def unitary_matrix_product(list_herm_matrices, t=1):
    ''' 
    matrix product of a list of unitary matrices exp(itH)
    input: 
        list_herm_matrices: a list of Hermitian matrices
        t: time
    return: the product of the corresponding matrices
    '''
    product = expm(-1j * t * list_herm_matrices[0])
    for i in range(1, len(list_herm_matrices)):
        product = product @ expm(-1j * t * list_herm_matrices[i])

    return product

def matrix_product(list_U, t=1):
    # product = matrix_power(list_U[0], t)
    # for i in range(1, len(list_U)):
    #     product = matrix_power(list_U[i], t) @ product
    #     # product = product @ matrix_power(list_U[i], t)
    product = np.linalg.multi_dot([matrix_power(U, t) for U in list_U])
    return product

# def second_order_trotter(list_herm_matrices, t=1):
#     forward_order_product = unitary_matrix_product(list_herm_matrices, t/2) 
#     reverse_order_product = unitary_matrix_product(list_herm_matrices[::-1], t/2)

#     return forward_order_product @ reverse_order_product

def pf_U(list_U, order, t=1):
    # print('order: ', order)
    if order == 1:
        return matrix_product(list_U, t)
    elif order == 2:
        forward_order_product = matrix_product(list_U, t/2) 
        reverse_order_product = matrix_product(list_U[::-1], t/2)
        return forward_order_product @ reverse_order_product
    elif order > 0 and order != 1 and order != 2 and order % 2 == 0:
        p = 1 / (4 - 4**(1/(order-1)))
        # print('p: ', p)
        return matrix_power(pf_U(list_U, order-2, p*t), 2) @ pf_U(list_U, order-2, (1-4*p)*t) @ matrix_power(pf_U(list_U, order-2, p*t), 2)
    else:
        raise ValueError('k is not defined')


########### jax, mpi, sparse ###########
def jax_matrix_exponential(matrix):
    # return jsl.expm( matrix)
    return ssla.expm(matrix)
jax_matrix_exponential = jax.jit(jax.vmap(jax_matrix_exponential))

def sparse_multi_dot(sparse_matrices):
    '''
    计算一个列表中所有矩阵的乘积
    '''
    product = sparse_matrices[0]
    for matrix in sparse_matrices[1:]:
        product = product.dot(matrix)
    return product
    # return product.toarray()

vectorized_sparse_expm = jax.vmap(ssla.expm)

def mpi_sparse_expm(list_herms, t, r):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_unitaries = pool.map(ssla.expm, -1j * t / r * np.array(list_herms))
    # Close the pool of workers
    pool.close()
    pool.join()

    return list_unitaries
