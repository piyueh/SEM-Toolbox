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


class JacobiElem:
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

    def __init__(self, nIdx_g, n, alpha, beta, tol=1e-12):
        """__init__

        Args:
            nIdx_g: array of the global indicies of the nodes in this element
            n: number of modes in this element
            alpha: the alpha used for Jacobi polynomial
            beta: the beta used for Jacobi polynomial
            tol: tolerance for entities in mass matrix to be treat as zeros
        """

        assert isinstance(nIdx_g, (numpy.ndarray, list)), \
            "nIdx_g is neither a numpy array nor a list"
        assert isinstance(n, (int, numpy.int_)), \
            "the number of nodes, n, is not an integer"
        assert n >= 2, \
            "the number of nodes, n, should be >= 2"
        assert n == len(nIdx_g), \
            "the lenth of nIdx_g is not the same as n"

        self.nIdx_g = numpy.array(nIdx_g)

        self.p_order = n - 1
        self.n_nodes = n
        self.alpha = alpha
        self.beta = beta

        self._set_expn()
        self._set_mass_mtx(tol)

    def _set_expn(self):
        """set up expansion polynomials"""

        self.expn = numpy.array([None]*(self.n_nodes), dtype=Polynomial)

        self.expn[0] = Polynomial([0.5, -0.5])
        self.expn[-1] = Polynomial([0.5, 0.5])

        for i in range(1, self.p_order):
            self.expn[i] = Jacobi(i-1, self.alpha, self.beta) * \
                Polynomial(roots=[1, -1], leading=-0.25)

    def _set_mass_mtx(self, tol=1e-12):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                p = (self.expn[i] * self.expn[j]).integral()
                self.M[i, j] = p(1) - p(-1)

        Mmax = numpy.max(self.M)
        self.M = numpy.where(numpy.abs(self.M/Mmax) <= tol, 0, self.M)
