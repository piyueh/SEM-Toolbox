#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the Polynomial class"""

import numpy
from poly_operations import eval_poly, der_poly


class Polynomial(object):
    """The base class for all kinds of polynomial.

    Attrs:
        n: the order of the polynomial
        coeffs: coefficient vector of the polynomial
        defined: a boolean indicating whether the polynomial is defined
    """

    def __init__(self, _coeffs=None):
        """__init__

        Args:
            _coeffs: coefficient vector of the polynomial

        Returns: None
        """

        if _coeffs is not None:
            self.set(_coeffs)
        else:
            self.n = None
            self.coeffs = None
            self.defined = False

    def __call__(self, x):
        """__call__

        Args:
            x: the location at where the value will be evaluated

        Returns:
            the value at x of the polynomial
        """
        return eval_poly(x, self.coeffs)

    def set(self, _coeffs):
        """set

        Args:
            _coeffs: coefficient vector of the polynomial

        Returns:
        """
        assert isinstance(_coeffs, numpy.ndarray), \
            "coeffs is not a numpy.ndarray"
        assert len(_coeffs.shape) == 1, "_coeffs is not a 1D array"

        self.n = _coeffs.size
        self.coeffs = _coeffs.copy()
        self.defined = True

    def derive(self, o=1):
        assert o >= 1, "The order of derivative should >= 1"
        assert o <= (self.n + 1), "The order of derivative should <= n+1"

        if o == 1:
            return Polynomial(der_poly(self.coeffs))
        else:
            return Polynomial(der_poly(self.coeffs)).derive(o-1)
