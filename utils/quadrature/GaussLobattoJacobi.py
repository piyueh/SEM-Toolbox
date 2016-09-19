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
from utils.misc import gamma, factorial


class GaussLobattoJacobi(GaussJacobi):
    """Definition of the class for Gauss-Lobatto-Jacobi quadrature

    For usage and attributes, please refer to Gauss-Jacobi class.
    """
    def __quad_points__(self):
        """__quad_points__

        Calculate the locations and weights of quadrature points
        """

        if self.n == 2:
            self.nodes = numpy.array([-1., 1.])
            self.weights = numpy.ones(2)
        else:
            self.nodes = numpy.array([-1.] + [0]*(self.n-2) + [1.])
            p = Jacobi(self.n-2, self.alpha+1, self.beta+1)
            self.nodes[1: -1] = p.roots

        c1 = 2**(self.alpha+self.beta+1)
        c1 *= gamma(self.alpha+self.n)
        c1 *= gamma(self.beta+self.n)
        c1 /= (self.n - 1)
        c1 /= factorial(self.n-1)
        c1 /= gamma(self.alpha+self.beta+self.n+1)

        c2 = Jacobi(self.n-1, self.alpha, self.beta)(self.nodes)
        self.weights = c1 / c2 / c2
        self.weights[0] *= (self.beta + 1.)
        self.weights[-1] *= (self.alpha + 1.)

    def __check_order__(self):
        """Check the order of quadrature"""

        assert self.n >= 2, \
            "Gauss-Lobatto-Jacobi quadrature requires " +\
            "the order to be greater than or equal to 2"
