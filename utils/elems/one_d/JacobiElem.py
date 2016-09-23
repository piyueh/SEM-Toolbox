#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of element using Jacobi polynomials as expansions"""

import numpy
from utils.poly import Polynomial
from utils.poly import Jacobi
from utils.quadrature import GaussLobattoJacobi


class JacobiElem:

    def __init__(self, nIdx_g, P, alpha, beta, Q=10):

        # TODO: check whether nIdx_g is a list or an array
        # TODO: check whether P, alpha, beta, Q are integers
        # TODO: check whether P >= 1
        # TODO: check whether len(nIdx_g) == P + 1

        self.p_order = P
        self.n_nodes = self.p_order + 1
        self.alpha = alpha
        self.beta = beta

        self.Q = Q
        self.quad = GaussLobattoJacobi(self.Q)

        self.nIdx_g = nIdx_g

        self._set_expn()
        self._set_mass_mtx()

    def _set_expn(self):

        self.expn = numpy.array([None]*(self.n_nodes), dtype=Polynomial)

        self.expn[0] = Polynomial([0.5, -0.5])
        self.expn[-1] = Polynomial([0.5, 0.5])

        for i in range(1, self.p_order):
            self.expn[i] = \
                Polynomial(roots=[1, -1], leading=0.25) * \
                Jacobi(i-1, self.alpha, self.beta)

    def _set_mass_mtx(self):

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.M[i, j] = self.quad(self.expn[i] * self.expn[j])
                '''
                if numpy.abs(self.M[i, j]) <= 1e-12:
                    self.M[i, j] = 0
                '''
