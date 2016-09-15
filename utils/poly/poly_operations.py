#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Operations of polynomial class"""

import numpy
from utils.errors import InfLoopError


def eval_poly(x, C):
    """evaluate the value of the polynomial at x using recursion

    Args:
        x: the point at where the value will be evaluated
        C: coefficient array

    Returns:
        value
    """
    assert isinstance(C, numpy.ndarray), \
        "The coefficient array is not a NumPy array"
    assert len(C.shape) == 1, "_coeffs is not a 1D array"

    if C.size == 2:
        return C[0] + C[1] * x
    else:
        return C[0] + x * eval_poly(x, C[1:])


def der_poly(C):
    """der_poly

    Args:
        C: coefficient array

    Returns:
        the coefficient array of the derived polynomial
    """
    assert isinstance(C, numpy.ndarray), \
        "The coefficient array is not a NumPy array"
    assert len(C.shape) == 1, "C is not a 1D array"

    if C.size == 1:
        return numpy.zeros(1)

    dC = numpy.zeros(C.size - 1)
    for i, c in enumerate(C[1:]):
        dC[i] = c * (i + 1)

    return dC


def find_roots(C, z=None):
    """find the roots of the polynomial, using Laguerre's method

    Args:
        C: coefficient array
        z: user-provided initial guesses for all roots

    Returns:
        roots
    """
    assert isinstance(C, numpy.ndarray), \
        "The coefficient array is not a NumPy array"
    assert len(C.shape) == 1, "C is not a 1D array"
    assert C.size > 1, \
        "The order of the polynomial is less than 1, no root exists"

    C = C / C[-1]
    n = C.size - 1
    stop = numpy.array([False]*n)

    if z is None:
        z = numpy.power(0.5+0.5j, numpy.arange(n))
    else:
        assert isinstance(z, numpy.ndarray), "z is not a NumPy array"
        assert len(z.shape) == 1,  "z is not a 1D array"
        assert z.size == n, "z is not in the same size with polynomial order"

    N = 0
    while not stop.all():

        for i in range(n):
            delta = eval_poly(z[i], C) / \
                (numpy.prod(z[i]-z[:i]) * numpy.prod(z[i]-z[i+1:]))
            z[i] -= delta

            if numpy.abs(delta) < 1e-14:
                stop[i] = True

        N += 1
        if N > 1000:
            raise InfLoopError(1000)

    z = numpy.where(numpy.abs(z.real) < 1e-14, z.imag*1j, z)
    z = numpy.where(numpy.abs(z.imag) < 1e-14, z.real, z)
    z = numpy.real(z) if (z.imag == 0).all() else z

    return z
