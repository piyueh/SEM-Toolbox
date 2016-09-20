#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""
Definition of Gauss-Jacobi quadrature class. It's also the base class for all
Gauss-based quadrature methods.
"""

import numpy
from utils.poly.Jacobi import Jacobi
from utils.misc import gamma, factorial


class GaussJacobi(object):
    """Definition of Gauss-Jacobi quadrature class

    Usage:

        q = GaussJacobi(7, 0, 0)
        f = lambda x: x**6
        print(q(f, -1., 2.))

        >> 18.42857....

    Attributes:
        nodes: quadrature points
        weights: weights for quadrature points
    """

    def __init__(self, _n, _alpha=0, _beta=0):
        """Initialization of Gauss-Jacobi quafrature instance

        Args:
            _n: the order of Gauss-Jacobi quadrature
            _alpha: the alpha for the Jacobi polynomial
            _beta: the beta for the Jacobi polynomial
        """

        self.n = _n
        self.__check_order()

        self.alpha = _alpha
        self.beta = _beta

        self.__quad_points()

    def __call__(self, f, xmin=-1, xMax=1):
        """Carry out quadrature integration on the one-variable function f

        Linear shape function is assumed.

        Args:
            f: the one-variable function
            xmin: the lower limit of the integral
            xMax: the upper limit of the integral

        Returns:
            the result of integration
        """

        dx_dxi = (xMax - xmin) / 2
        x = (self.nodes + 1.) * dx_dxi + xmin

        return numpy.sum(f(x) * dx_dxi * self.weights)

    def __repr__(self):
        """__repr__"""

        return "{0}({1}, {2}, {3})".format(
            self.__class__, self.n, self.alpha, self.beta)

    def __str__(self):
        """__str__"""

        s = "Nodes  : " + str(self.nodes) + "\n"
        s += "Weights: " + str(self.weights)

        return s

    def __quad_points(self):
        """__quad_points__

        Calculate the locations and weights of quadrature points
        """

        if self.n == 1:
            self.nodes = numpy.array([0.])
            self.weights = numpy.array([2.])
        else:
            p = Jacobi(self.n, self.alpha, self.beta)
            self.nodes = p.roots

            c1 = 2**(self.alpha+self.beta+1)
            c1 *= gamma(self.alpha+self.n+1)
            c1 *= gamma(self.beta+self.n+1)
            c1 /= factorial(self.n)
            c1 /= gamma(self.alpha+self.beta+self.n+1)
            c1 /= (1. - self.nodes**2)

            c2 = p.derive()(self.nodes)
            self.weights = c1 / c2 / c2

    def __check_order(self):
        """Check the order of quadrature"""

        assert self.n >= 1, \
            "Gauss-Jacobi quadrature requires the order to be greater than 0"
