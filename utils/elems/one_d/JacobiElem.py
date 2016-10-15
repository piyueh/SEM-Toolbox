#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of standard element of p-type Jacobi polynomials expansions"""

import numpy
from utils.poly import Polynomial
from utils.poly import Jacobi
from utils.elems.one_d.BaseElem import BaseElem


class JacobiElem(BaseElem):
    """General class for p-type modal expansion using Jacobi polynomial

    For this element, the boundary and interior modes are

    i = 0, phi(x) = (1-x) / 2
    i = 1 to n-2, phi(x) = (1-x)(1+x)J(i-1, alpha, beta)/4
    i = n-1, phi(x) = (1+x)/2

    The construction of mass matrix is done by direct polynomial integration. So
    it's likely return very small floating numbers that are supposed to be
    zeros. Therefore, for common type of Jacobi polynomials, such as alpha=beta=
    0, 1, 2, it's better to use specialized class.
    """

    def __init__(self, ends, n, alpha, beta, tol=1e-12):
        """__init__

        Args:
            ends: array of the two end nodes (their locations)
            n: number of modes in this element
            alpha: the alpha used for Jacobi polynomial
            beta: the beta used for Jacobi polynomial
            tol: tolerance for entities in mass matrix to be treat as zeros
        """

        self.p_order = n - 1
        self.alpha = alpha
        self.beta = beta

        super().__init__(ends, n, tol)

    def _set_expn(self):
        """set up expansion polynomials"""

        self.expn = numpy.array([None]*(self.n_nodes), dtype=Polynomial)

        self.expn[0] = Polynomial([0.5, -0.5])
        self.expn[-1] = Polynomial([0.5, 0.5])

        for i in range(1, self.p_order):
            self.expn[i] = Jacobi(i-1, self.alpha, self.beta) * \
                Polynomial(roots=[1, -1], leading=-0.25)
