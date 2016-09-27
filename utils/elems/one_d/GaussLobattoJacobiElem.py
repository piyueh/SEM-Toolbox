#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Definition of specialized element using Lagrange expansion with roots of
Gauss-Lobatto-Jacobi polynomials
"""

import numpy
from utils.elems.one_d import LagrangeElem
from utils.quadrature import GaussLobattoJacobi


class GaussLobattoJacobiElem(LagrangeElem):
    """class for p-type modal expansion using Gauss-Lobatto-Jacobi polynomial

    Note:
        the mass matrix is calculated through Gauss-Lobatto-Jacobi quadrature,
        so the mass matrix is lumpped.
    """

    def __init__(self, ends, n, alpha=0, beta=0):
        """__init__

        Args:
            ends: array of the two end nodes (their locations)
            n: number of modes in this element
            alpha, beta: parameters for Jacobi polynomial
            tol: tolerance for entities in mass matrix to be treat as zeros
        """

        self.alpha = alpha
        self.beta = beta
        self.qd = GaussLobattoJacobi(n, alpha, beta)

        super().__init__(ends, n, self.qd.nodes)

    def _set_mass_mtx(self, tol=1e-12):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            self.M[i, i] = self.qd.weights[i]
