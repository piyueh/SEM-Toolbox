#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the class for Gauss-Radau-Jacobi quadrature"""

import numpy
from utils.quadrature.GaussJacobi import GaussJacobi
from utils.poly.Jacobi import Jacobi
from utils.misc import factorial_division


class GaussRadauJacobi(GaussJacobi):
    """Definition of the class for Gauss-Radau-Jacobi quadrature

    For usage and attributes, please refer to Gauss-Jacobi class.
    """

    def __init__(self, n, end=-1, alpha=0, beta=0):
        """initialize the Gauss-Radau-Jacobi quadrature instance

        Args:
            n: the order of Gauss-Jacobi quadrature
            end: either -1 or 1, indicate the included end point
            alpha: the alpha for the Jacobi polynomial
            beta: the beta for the Jacobi polynomial
        """
        if end not in [-1, 1]:
            raise ValueError(
                "end should indicate if the included end point is -1 or 1. " +
                "The input end is: {0}, and type: {1}".format(end, type(end)))

        self.end = end

        super().__init__(n, alpha, beta)

    def _quad_points(self):
        """_quad_points

        Calculate the locations and weights of quadrature points
        """

        def B():
            p = Jacobi(self.n-1, self.alpha, self.beta)

            ans = numpy.power(2, self.alpha+self.beta)
            ans *= factorial_division(self.n-1, self.n+self.alpha-1)
            ans /= (self.beta + self.n)
            ans /= factorial_division(
                self.n+self.beta-1, self.n+self.alpha+self.beta)

            ans *= (1 - self.nodes)
            ans /= numpy.power(p(self.nodes), 2)
            return ans

        # init quadrature points, xi, and weights, wi
        self.nodes = numpy.zeros(self.n, dtype=numpy.float64)
        self.weights = numpy.zeros(self.n, dtype=numpy.float64)

        # quadrature points
        p = Jacobi(self.n-1, self.alpha, self.beta+1)
        self.nodes[0] = -1.
        self.nodes[1:] = p.roots

        # weights
        self.weights = B()
        self.weights[0] *= (self.beta + 1)

        # inverse, if the included end is x=1
        if self.end == 1:
            self.nodes = - self.nodes[::-1]
            self.weights = self.weights[::-1]
