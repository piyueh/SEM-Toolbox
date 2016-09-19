#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the Polynomial class"""

import numpy
from utils.poly.poly_operations import eval_poly, der_poly, find_roots
from utils.poly.poly_operations import add_polys, mul_poly


class Polynomial(object):
    """The base class for all kinds of polynomial.

    Attrs:
        n: the order of the polynomial
        coeffs: coefficient vector of the polynomial
        defined: a boolean indicating whether the polynomial is defined
        roots: the roots of the polynomial
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
            self.roots = None

    def __call__(self, x):
        """__call__

        Args:
            x: the location at where the value will be evaluated

        Returns:
            the value at x of the polynomial
        """
        return eval_poly(x, self.coeffs)

    def __repr__(self):
        """__repr__"""

        return "{0}({1})".format(self.__class__, self.coeffs)

    def __str__(self):
        """__str__"""

        s = "{0}".format(self.coeffs[0])
        for i, c in enumerate(self.coeffs[1:]):
            s += " + "
            s += "{0} x^{1}".format(c, i)

        return s

    def __add__(self, other):
        """overloading the + operator"""

        return Polynomial(add_polys(self.coeffs, other.coeffs))

    def __sub__(self, other):
        """overloading the - operator"""

        return Polynomial(add_polys(self.coeffs, - other.coeffs))

    def __mul__(self, other):
        """overloading the * operator"""

        return Polynomial(mul_poly(self.coeffs, other.coeffs))

    def __find_roots(self):
        """calculate the roots and store them in self.root"""
        self.roots = find_roots(self.coeffs)

    def set(self, _coeffs):
        """set

        Args:
            _coeffs: coefficient vector of the polynomial

        Returns:
        """
        assert isinstance(_coeffs, numpy.ndarray), \
            "coeffs is not a numpy.ndarray"
        assert len(_coeffs.shape) == 1, "_coeffs is not a 1D array"

        self.n = _coeffs.size - 1
        self.coeffs = _coeffs.copy()
        self.defined = True
        self.__find_roots()

    def derive(self, o=1):
        assert o >= 1, "The order of derivative should >= 1"
        assert o <= (self.n + 1), "The order of derivative should <= n+1"

        if o == 1:
            return Polynomial(der_poly(self.coeffs))
        else:
            return Polynomial(der_poly(self.coeffs)).derive(o-1)
