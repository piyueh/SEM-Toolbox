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
from utils.misc import factorial_division


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
        self._check_order()

        self.alpha = _alpha
        self.beta = _beta

        self._quad_points()

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

        if callable(f):
            return numpy.sum(f(x) * dx_dxi * self.weights)
        else:
            # TODO: check the type & size of f is a list/ndarray
            return numpy.sum(f * dx_dxi * self.weights)

    def __repr__(self):
        """__repr__"""

        return "{0}({1}, {2}, {3})".format(
            self.__class__, self.n, self.alpha, self.beta)

    def __str__(self):
        """__str__"""

        s = "Nodes  : " + str(self.nodes) + "\n"
        s += "Weights: " + str(self.weights)

        return s

    def _quad_points(self):
        """_quad_points

        Calculate the locations and weights of quadrature points
        """

        def H():
            p = Jacobi(self.n, self.alpha, self.beta)

            ans = numpy.power(2, self.alpha+self.beta+1)
            ans *= factorial_division(self.n, self.n+self.alpha)
            ans /= factorial_division(
                self.n+self.beta, self.n+self.alpha+self.beta)

            ans /= (1. - numpy.power(self.nodes, 2))
            ans /= numpy.power(p.derive()(self.nodes), 2)

            return ans

        # quadrature points
        p = Jacobi(self.n, self.alpha, self.beta)
        self.nodes = p.roots

        # weights
        self.weights = H()

    def _check_order(self):
        """Check the order of quadrature"""

        if self.n < 1:
            raise ValueError(
                "Requires the order to be greater than 0. " +
                "Current input n: {0}".format(self.n))
