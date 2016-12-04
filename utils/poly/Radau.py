#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of Radau polynomial class"""

import numpy
from utils.poly.Polynomial import Polynomial
from utils.poly.Legendre import Legendre


class Radau(Polynomial):
    """Radau polynomial

    Its attributions are the same as the Polynomial class.
    """

    def __init__(self, n, end=1):
        """initialize the instance of Jacobi polynomial

        Args:
            n: the order of the Radau polynomial
            end: which end of interval -1~1 is included
        """
        # TODO: check the type and vlaue of end

        self.end = end

        # right Radau; R(x) = (-1)^n * 0.5 * (L_n - L_{n-1})
        if end == 1:
            coeffs = (Legendre(n) - Legendre(n-1)).coeffs
            if n % 2 == 1:
                coeffs *= (-1)
        # left Radau;  R(x) = 0.5 * (L_n + L_{n-1})
        elif end == -1:
            coeffs = (Legendre(n) + Legendre(n-1)).coeffs

        coeffs *= 0.5

        super().__init__(coeffs)

        if self.n > 0:
            assert self.roots.dtype == numpy.float64, \
                "The roots of a Jacobi polynomial should be real numbers. " +\
                "Please check the source code of polynomial operations."

            self.roots = numpy.sort(self.roots)

    def __repr__(self):
        """__repr__"""

        return "{0}({1}, end={2})".format(self.__class__, self.n, self.end)
