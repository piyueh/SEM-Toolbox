#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the Polynomial class"""

import numpy
from utils.poly.poly_operations import eval_poly_coeffs, eval_poly_roots
from utils.poly.poly_operations import der_poly
from utils.poly.poly_operations import find_roots, find_coeffs
from utils.poly.poly_operations import add_polys, mul_poly


class Polynomial(object):
    """The base class for all kinds of polynomial.

    User can create a Polynomial instance with either providing a coefficient
    array or an array of roots. If both provided, using roots is the top choice.
    If both provided, currently the code will not check whether the roots
    match the coefficients. It simply use roots.

    Creating an instance with roots, the coefficient of the highest order term
    is 1 by default. Users can scale the coefficients by prviding the value of
    `leading` when instantializing.

    Attrs:
        n: the order of the polynomial
        coeffs: coefficient vector of the polynomial
        defined: a boolean indicating whether the polynomial is defined
        roots: the roots of the polynomial
    """

    def __init__(self, coeffs=None, roots=None, leading=1):
        """__init__

        Args:
            coeffs: coefficient vector of the polynomial
            roots: the roots of the polynomial
            leading: the leading (scaling) coefficient of the polynomial,
                     only works when create instance with roots

        Returns: None
        """

        if (roots is not None):
            self.set_from_roots(roots, leading)
        elif (coeffs is not None):
            self.set_from_coeffs(coeffs)
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
        return self.__eval(x)

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

    def __find_coeffs(self, _leading):
        """calculate the coefficients using its roots"""
        self.coeffs = find_coeffs(self.roots) * _leading

    def set_from_coeffs(self, _coeffs):
        """set the polynomial instance using its coefficients

        Args:
            _coeffs: coefficient vector of the polynomial
        """
        assert isinstance(_coeffs, numpy.ndarray), \
            "coeffs is not a numpy.ndarray"
        assert len(_coeffs.shape) == 1, "_coeffs is not a 1D array"

        self.n = _coeffs.size - 1
        self.coeffs = _coeffs.copy()
        self.defined = True
        self.__find_roots()
        self.eval_method = "C"
        self.__eval = lambda x: eval_poly_coeffs(x, self.coeffs)

    def set_from_roots(self, _roots, _leading):
        """set the polynomial instance using its roots

        Args:
            _roots: array of roots
            _leading: the leading (scaling) coefficient of the polynomial
        """
        assert isinstance(_roots, numpy.ndarray), \
            "coeffs is not a numpy.ndarray"
        assert len(_roots.shape) == 1, "_roots is not a 1D array"

        self.n = _roots.size
        self.roots = _roots.copy()
        self.defined = True
        self.__find_coeffs(_leading)
        self.eval_method = "R"
        self.__eval = lambda x: eval_poly_roots(x, self.roots, self.coeffs[-1])

    def derive(self, o=1):
        assert o >= 1, "The order of derivative should >= 1"
        assert o <= (self.n + 1), "The order of derivative should <= n+1"

        if o == 1:
            return Polynomial(der_poly(self.coeffs))
        else:
            return Polynomial(der_poly(self.coeffs)).derive(o-1)
