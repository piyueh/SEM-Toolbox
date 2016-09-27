#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of commonly used p-type Jacobi polynomials expansions"""

import numpy
from utils.elems.one_d import JacobiElem
from utils.poly.jacobi_operations import jacobi_orthogonal_constant


class CommonJacobiElem(JacobiElem):
    """Class for single element using C0 Jacobi polynomial modal exapnasion

    This element use special Jacobi polynomial, which has alpha = beta = 1, for
    interior modes but (1 - x) / 2 and (1 + x) / 2 for boundary modes in order
    to keep globally C0 continuity.
    """

    def __init__(self, ends, n):
        """__init__

        Args:
            ends: array of the two end nodes (their locations)
            n: number of nodes in this element
        """

        super().__init__(ends, n, 1, 1)

    def _set_mass_mtx(self, tol):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        # 1st step: handle boundary modes
        self.M[0, 0] = 2. / 3.
        self.M[0, -1] = self.M[-1, 0] = 1. / 3.
        self.M[-1, -1] = 2. / 3.

        if self.n_nodes > 2:
            self.M[0, 1] = self.M[1, 0] = \
                jacobi_orthogonal_constant(0, 1, 1) / 8.
            self.M[1, -1] = self.M[-1, 1] = \
                jacobi_orthogonal_constant(0, 1, 1) / 8.

        if self.n_nodes > 3:
            self.M[0, 2] = self.M[2, 0] = \
                - jacobi_orthogonal_constant(1, 1, 1) / 16.
            self.M[2, -1] = self.M[-1, 2] = \
                jacobi_orthogonal_constant(1, 1, 1) / 16.

        # 2nd step: handle interior modes
        for i, p in enumerate(self.expn[1:-1]):
            for j in range(i+1, min(self.n_nodes-1, i+2+2), 2):
                pi = (p * self.expn[j]).integral()
                self.M[i+1, j] = self.M[j, i+1] = pi(1) - pi(-1)
