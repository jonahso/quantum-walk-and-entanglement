############################
# define spin Hamiltonians
############################

import numpy as np
from qiskit.quantum_info import SparsePauliOp
import random
import itertools


class Nearest_Neighbour_1d:
    def __init__(self, n: int, Jx=0, Jy=0, Jz=0, hx=0, hy=0, hz=0, pbc=False, verbose=False, rand_field=[]):
        self.n = n
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.hx, self.hy, self.hz = hx, hy, hz

        self.xx_tuples = [('XX', [i, i + 1], Jx) for i in range(0, n-1)]
        self.yy_tuples = [('YY', [i, i + 1], Jy) for i in range(0, n-1)]
        self.zz_tuples = [('ZZ', [i, i + 1], Jz) for i in range(0, n-1)]

        if len(rand_field) == 0:
            self.rand_field = [0]*n
        elif len(rand_field) >= n:
            self.rand_field = rand_field[:n]
        else:
            raise ValueError(f'Length of random field should be at least {n}!')

        self.x_tuples = [('X', [i], (self.rand_field[i]+1)*hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], (self.rand_field[i]+1)*hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], (self.rand_field[i]+1)*hz) for i in range(0, n)] 

        if pbc: 
            self.xx_tuples.append(('XX', [n-1, 0], Jx))
            self.yy_tuples.append(('YY', [n-1, 0], Jy))
            self.zz_tuples.append(('ZZ', [n-1, 0], Jz))

        self.ham = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, *self.x_tuples, *self.y_tuples, *self.z_tuples], num_qubits=n).simplify() 
        self.xyz_group()
        self.par_group()
        if verbose: 
            print('The Hamiltonian: \n', self.ham)
            print('The xyz grouping: \n', self.ham_xyz)
            print('The parity grouping: \n', self.ham_par)

    def xyz_group(self):
        self.x_terms = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.x_tuples], num_qubits=self.n).simplify()
        self.y_terms = SparsePauliOp.from_sparse_list([*self.yy_tuples, *self.y_tuples], num_qubits=self.n).simplify()
        self.z_terms = SparsePauliOp.from_sparse_list([*self.zz_tuples, *self.z_tuples], num_qubits=self.n).simplify()
        self.ham_xyz = [self.x_terms, self.y_terms, self.z_terms]
        self.ham_xyz = [item for item in self.ham_xyz if not np.all(abs(item.coeffs) == 0)]

    def par_group(self):
        self.even_terms = SparsePauliOp.from_sparse_list([*self.xx_tuples[::2], *self.yy_tuples[::2], *self.zz_tuples[::2], *self.x_tuples[::2], *self.y_tuples[::2], *self.z_tuples[::2]], num_qubits=self.n).simplify()
        self.odd_terms = SparsePauliOp.from_sparse_list([*self.xx_tuples[1::2], *self.yy_tuples[1::2], *self.zz_tuples[1::2], *self.x_tuples[1::2], *self.y_tuples[1::2], *self.z_tuples[1::2]], num_qubits=self.n).simplify()
        self.ham_par = [self.even_terms, self.odd_terms]

    # def lc_group(self, right, left, step):
    #     self.ham_lc = []

