#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the class for Gauss-Lobatto-Jacobi quadrature"""

import numpy
from utils.quadrature.GaussJacobi import GaussJacobi
from utils.poly.Jacobi import Jacobi
from utils.misc import factorial_division


class GaussLobattoJacobi(GaussJacobi):
    """Definition of the class for Gauss-Lobatto-Jacobi quadrature

    For usage and attributes, please refer to Gauss-Jacobi class.
    """
    def _quad_points(self):
        """_quad_points

        Calculate the locations and weights of quadrature points
        """

        def C():
            p = Jacobi(self.n-1, self.alpha, self.beta)

            ans = numpy.power(2, self.alpha+self.beta+1)
            ans /= (self.n-1)
            ans *= factorial_division(self.n-1, self.n+self.alpha-1)
            ans /= factorial_division(
                self.n+self.beta-1, self.n+self.alpha+self.beta)
            ans /= numpy.power(p(self.nodes), 2)

            return ans

        # init quadrature points, xi, and weights, wi
        self.nodes = numpy.zeros(self.n, dtype=numpy.float64)
        self.weights = numpy.zeros(self.n, dtype=numpy.float64)

        # quadrature points
        p = Jacobi(self.n-2, self.alpha+1, self.beta+1)
        self.nodes[0] = -1.
        self.nodes[-1] = 1.
        self.nodes[1:-1] = p.roots

        # weights
        self.weights = C()
        self.weights[0] *= (self.beta + 1)
        self.weights[-1] *= (self.alpha + 1)

    def _check_order(self):
        """Check the order of quadrature"""

        if self.n < 2:
            raise ValueError(
                "Lobatto requires the order to be greater than 1. " +
                "Current input n: {0}".format(self.n))
