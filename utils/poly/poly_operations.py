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
    check_C(C)

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
    check_C(C)

    if C.size == 1:
        return numpy.zeros(1)

    dC = numpy.zeros(C.size - 1)
    for i, c in enumerate(C[1:]):
        dC[i] = c * (i + 1)

    return dC


def find_roots(C, z=None):
    """find the roots of the polynomial, using Durand-Kerner method

    Args:
        C: coefficient array
        z: user-provided initial guesses for all roots

    Returns:
        roots
    """
    check_C(C)
    assert C.size > 1, \
        "The order of the polynomial is less than 1, no root exists"

    C = C / C[-1]
    C = C.astype(numpy.complex256)
    n = C.size - 1
    stop = numpy.array([False]*n)

    if z is None:
        z = numpy.power(0.5+0.5j, numpy.arange(n))
    else:
        assert isinstance(z, numpy.ndarray), "z is not a NumPy array"
        assert len(z.shape) == 1,  "z is not a 1D array"
        assert z.size == n, "z is not in the same size with polynomial order"

    z = z.astype(numpy.complex256)

    N = 0
    while not stop.all():

        for i in range(n):
            delta = eval_poly(z[i], C) / \
                (numpy.prod(z[i]-z[:i]) * numpy.prod(z[i]-z[i+1:]))
            z[i] -= delta

            if numpy.abs(delta) < 1e-14:
                stop[i] = True

        N += 1
        if N > 100000:
            print(z)
            raise InfLoopError(100000)

    z = z.astype(numpy.complex128)
    z = numpy.where(numpy.abs(z.real) < 1e-8, z.imag*1j, z)
    z = numpy.where(numpy.abs(z.imag) < 1e-8, z.real, z)
    z = numpy.real(z) if (z.imag == 0).all() else z

    return z


def comp_matrix(C):
    """construct the companion matrix of the polynomial

    Args:
        C: coefficient array

    Returns:
        companion matrix
    """
    check_C(C)
    assert C.size > 1, \
        "The order of the polynomial is less than 1, no companion matrix"

    C = C / C[-1]
    n = C.size - 1
    m = numpy.diag([1.]*(n-1), -1)

    m[:, -1] -= C[:-1]

    return m


def add_polys(C1, C2):
    """add two polynomials

    Args:
        C1: coefficient array
        C2: coefficient array

    Returns:
        a new coefficient array representing the sum of the two polynomials
    """
    check_C(C1)
    check_C(C2)

    if C1.size < C2.size:
        C1 = numpy.pad(C1, (0, C2.size-C1.size), 'constant', constant_values=0)
    if C2.size < C1.size:
        C2 = numpy.pad(C2, (0, C1.size-C2.size), 'constant', constant_values=0)

    return C1 + C2


def mul_poly(C1, C2):
    """multiply two polynomials

    Args:
        C1: coefficient array
        C2: coefficient array

    Returns:
        a new coefficient array representing the multiplication of the
        two polynomials
    """
    check_C(C1)
    check_C(C2)

    nSize = (C1.size - 1) + (C2.size - 1)  # the order of resulting polynomial
    nC = numpy.zeros(nSize + 1)  # coeff. array of resulting polynomial

    for i, c in enumerate(C1):
        nC[i:i+C2.size] += c * C2

    return nC


def check_C(C):
    """check the type and order of the polynomial

    Args:
        C: coefficient array
    """
    assert isinstance(C, numpy.ndarray), \
        "The coefficient array is not a NumPy array"
    assert len(C.shape) == 1, "C is not a 1D array"
    assert C.size >= 1, \
        "The order of the polynomial should >= 0"
