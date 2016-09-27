#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of standard element of p-type Legendre polynomials expansions"""

import numpy
from utils.elems.one_d import JacobiElem
from utils.poly.jacobi_operations import jacobi_orthogonal_constant


class LegendreElem(JacobiElem):
    """Class for single element using C0 Legendre polynomial modal exapnasion

    This element use Legendre polynomial for interior modes but (1 - x) / 2
    and (1 + x) / 2 for boundary modes in order to keep globally C0 continuity.
    """

    def __init__(self, nIdx_g, n):
        """__init__

        Args:
            nIdx_g: array of the global indicies of the nodes in this element
            n: number of nodes in this element
        """

        super().__init__(nIdx_g, n, 0, 0)

    def _set_mass_mtx(self, tol):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        # 1st step: handle boundary modes
        self.M[0, 0] = 2. / 3.
        self.M[0, -1] = 1. / 3.
        self.M[-1, -1] = 2. / 3.

        self.M[0, 1] = jacobi_orthogonal_constant(0, 0, 0) / 12.
        self.M[0, 2] = - jacobi_orthogonal_constant(1, 0, 0) / 20.
        self.M[0, 3] = - jacobi_orthogonal_constant(2, 0, 0) / 12.
        self.M[0, 4] = jacobi_orthogonal_constant(3, 0, 0) / 20.

        self.M[1, -1] = jacobi_orthogonal_constant(0, 0, 0) / 12.
        self.M[2, -1] = jacobi_orthogonal_constant(1, 0, 0) / 20.
        self.M[3, -1] = - jacobi_orthogonal_constant(2, 0, 0) / 12.
        self.M[4, -1] = - jacobi_orthogonal_constant(3, 0, 0) / 20.

        # 2nd step: handle interior modes
        for i, p in enumerate(self.expn[1:-1]):
            for j in range(i+1, min(self.n_nodes-1, i+4+2), 2):
                pi = (p * self.expn[j]).integral()
                self.M[i+1, j] = pi(1) - pi(-1)

        # 3rd step: symmetric
        for i in range(1, self.n_nodes):
            for j in range(0, i):
                self.M[i, j] = self.M[j, i]
