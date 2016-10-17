#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of standard element of moment expansions"""

import numpy
from utils.poly import Polynomial
from utils.elems.one_d.BaseElem import BaseElem


class MomentElem(BaseElem):
    """General class for moment expansion

    That is,

        phi_p(x) = x**p

    This expansion is just for test purpose.
    """

    def _set_expn(self):
        """set up expansion polynomials"""

        self.expn = numpy.array([None]*(self.n_nodes), dtype=Polynomial)

        self.expn[0] = Polynomial([1])
        for i in range(1, self.n_nodes):
            self.expn[i] = Polynomial(roots=[0]*i)

    def _set_mass_mtx(self, tol=1e-12):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if (i + j + 1) % 2 != 0:
                    self.M[i, j] = 2. / (i + j + 1)

    def _set_weak_laplacian(self, tol=1e-12):
        """set up the weak laplacian"""

        self.wL = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if (i + j - 1) % 2 != 0:
                    self.wL[i, j] = i * j * 2. / (i + j - 1)
