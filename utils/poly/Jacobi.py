#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of Jacobi polynomial class"""

import numpy
from utils.poly.Polynomial import Polynomial
from utils.poly.jacobi_operations import jacobi_coef


class Jacobi(Polynomial):
    """Jacobi polynomial

    Its attributions are the same as the Polynomial class.
    """

    def __init__(self, _n, _alpha, _beta):
        """initialize the instance of Jacobi polynomial

        Args:
            _n: the order of the Jacobi polynomial
            _alpha: the alpha of the Jacobi polynomial
            _beta: the beta of the Jacobi polynomial
        """

        c = jacobi_coef(_n, _alpha, _beta)
        super().__init__(c)

        self.n = _n
        self.alpha = _alpha
        self.beta = _beta

        assert self.roots.dtype == numpy.float64, \
            "The roots of a Jacobi polynomial should be real numbers. " +\
            "Please check the source code of polynomial operations."

        self.roots = numpy.sort(self.roots)

    def __repr__(self):
        """__repr__"""

        return "{0}({1}, {2}, {3})".format(
            self.__class__, self.n, self.alpha, self.beta)
