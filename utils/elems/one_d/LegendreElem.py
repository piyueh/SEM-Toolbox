#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of standard element of p-type Legendre polynomials expansions"""

import numpy
from utils.errors.JacobiOrderError import JacobiOrderError
from utils.elems.one_d import JacobiElem
from utils.poly.jacobi_operations import jacobi_orthogonal_constant
from utils.poly.jacobi_operations import jacobi_recr_coeffs


class LegendreElem(JacobiElem):
    """Class for single element using C0 Legendre polynomial modal exapnasion

    This element use Legendre polynomial for interior modes but (1 - x) / 2
    and (1 + x) / 2 for boundary modes in order to keep globally C0 continuity.
    """

    def __init__(self, ends, n):
        """__init__

        Args:
            ends: array of the two end nodes (their locations)
            n: number of nodes in this element
        """

        super().__init__(ends, n, 0, 0)

    def _set_mass_mtx(self, tol):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        # 1st step: handle boundary modes
        self.M[0, 0] = 2. / 3.
        self.M[0, -1] = self.M[-1, 0] = 1. / 3.
        self.M[-1, -1] = 2. / 3.

        if self.n_nodes > 2:
            self.M[0, 1] = self.M[1, 0] = \
                jacobi_orthogonal_constant(0, 0, 0) / 12.
            self.M[1, -1] = self.M[-1, 1] = \
                jacobi_orthogonal_constant(0, 0, 0) / 12.

        if self.n_nodes > 3:
            self.M[0, 2] = self.M[2, 0] = \
                - jacobi_orthogonal_constant(1, 0, 0) / 20.
            self.M[2, -1] = self.M[-1, 2] = \
                jacobi_orthogonal_constant(1, 0, 0) / 20.

        if self.n_nodes > 4:
            self.M[0, 3] = self.M[3, 0] = \
                - jacobi_orthogonal_constant(2, 0, 0) / 12.
            self.M[3, -1] = self.M[-1, 3] = \
                - jacobi_orthogonal_constant(2, 0, 0) / 12.

        if self.n_nodes > 5:
            self.M[0, 4] = self.M[4, 0] = \
                jacobi_orthogonal_constant(3, 0, 0) / 20.
            self.M[4, -1] = self.M[-1, 4] = \
                - jacobi_orthogonal_constant(3, 0, 0) / 20.

        # 2nd step: handle interior modes
        for i, p in enumerate(self.expn[1:-1]):
            for j in range(i+1, min(self.n_nodes-1, i+4+2), 2):
                pi = (p * self.expn[j]).integral()
                self.M[i+1, j] = self.M[j, i+1] = pi(1) - pi(-1)

    def _set_weak_laplacian(self, tol=1e-12):
        """set up the mass matrix"""

        self.wL = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        # boundary-boundary
        self.wL[0, 0] = self.wL[-1, -1] = 0.5
        self.wL[0, -1] = self.wL[-1, 0] = -0.5

        # boundary-interior
        # this part is always zero for p-type Legendre polynomial

        # interior-interior
        def A(i):
            def Ac(i, j):
                return (i + 1) * (j + 1) / 16.

            a1, a2, a3, a4 = jacobi_recr_coeffs(i-1, 0, 0)
            a1 /= a3
            a4 /= a3

            try:
                b1, b2, b3, b4 = jacobi_recr_coeffs(i-3, 0, 0)
                b1 /= b3
                b4 /= b3
            except JacobiOrderError:
                b1 = b3 = b4 = 0
            except:
                raise

            c1, c2, c3, c4 = jacobi_recr_coeffs(i+1, 0, 0)
            c1 /= c3
            c4 /= c3

            try:
                A1 = Ac(i, i-2) * \
                    a4 * b1 * jacobi_orthogonal_constant(i-2, 0, 0)
            except JacobiOrderError:
                A1 = 0
            except:
                raise

            A2 = a1 * a1 * jacobi_orthogonal_constant(i, 0, 0)
            try:
                A2 += a4 * a4 * jacobi_orthogonal_constant(i-2, 0, 0)
            except JacobiOrderError:
                A2 += 0
            except:
                raise

            A2 *= Ac(i, i)

            A3 = Ac(i, i+2) * a1 * c4 * jacobi_orthogonal_constant(i, 0, 0)

            return A1, A2, A3

        def B(i):
            def Bc(i, j):
                return - (i + 1) * (j - 1) / 16.

            a1, a2, a3, a4 = jacobi_recr_coeffs(i-1, 0, 0)
            a1 /= a3
            a4 /= a3

            B1 = 0

            try:
                B2 = Bc(i, i) * a4 * jacobi_orthogonal_constant(i-2, 0, 0)
            except JacobiOrderError:
                B2 = 0
            except:
                raise

            B3 = Bc(i, i+2) * a1 * jacobi_orthogonal_constant(i, 0, 0)

            return B1, B2, B3

        def C(i):
            def Cc(i, j):
                return - (j + 1) * (i - 1) / 16.

            try:
                a1, a2, a3, a4 = jacobi_recr_coeffs(i-2, 0, 0)
                a1 /= a3
                a4 /= a3
            except JacobiOrderError:
                a1 = a4 = 0
            except:
                raise

            try:
                C1 = Cc(i, i-2) * a4 * jacobi_orthogonal_constant(i-3, 0, 0)
            except JacobiOrderError:
                C1 = 0
            except:
                raise
            C2 = Cc(i, i) * a1 * jacobi_orthogonal_constant(i-1, 0, 0)
            C3 = 0

            return C1, C2, C3

        def D(i):
            def Dc(i, j):
                return (j - 1) * (i - 1) / 16.

            D1 = 0

            try:
                D2 = Dc(i, i) * jacobi_orthogonal_constant(i-2, 0, 0)
            except JacobiOrderError:
                D2 = 0
            except:
                raise

            D3 = 0

            return D1, D2, D3

        for i in range(1, self.n_nodes-1):
            A1, A2, A3 = A(i)
            B1, B2, B3 = B(i)
            C1, C2, C3 = C(i)
            D1, D2, D3 = D(i)

            # j = i - 2
            if i >= 3:
                self.wL[i, i-2] = A1 + B1 + C1 + D1

            # j = i
            self.wL[i, i] = A2 + B2 + C2 + D2

            # j = i + 2, the same as i - 2
            if i <= self.n_nodes - 4:
                self.wL[i, i+2] = A3 + B3 + C3 + D3
