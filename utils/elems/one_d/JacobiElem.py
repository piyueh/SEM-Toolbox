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

    def __init__(self, nIdx_g, n, alpha, beta):
        """__init__

        Args:
            nIdx_g: array of the global indicies of the nodes in this element
            n: number of nodes in this element
            alpha: the alpha used for Jacobi polynomial
            beta: the beta used for Jacobi polynomial
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
        self._set_mass_mtx()

    def _set_expn(self):
        """set up expansion polynomials"""

        self.expn = numpy.array([None]*(self.n_nodes), dtype=Polynomial)

        self.expn[0] = Polynomial([0.5, -0.5])
        self.expn[-1] = Polynomial([0.5, 0.5])

        for i in range(1, self.p_order):
            self.expn[i] = Jacobi(i-1, self.alpha, self.beta) * \
                Polynomial(roots=[1, -1], leading=0.25)

    def _set_mass_mtx(self):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                print(i, j)
                p = self.expn[i] * self.expn[j]
                pi = p.integral()
                self.M[i, j] = pi(1) - pi(-1)

                # TODO: this is silly... find analytical solutions to build this
                #       "sparse" matrix!!
                if numpy.abs(self.M[i, j]) <= 1e-14:
                    self.M[i, j] = 0
