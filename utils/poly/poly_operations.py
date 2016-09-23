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
from utils.misc import strip_trivial


def eval_poly_coeffs(x, C):
    """providing polynomial coefficients, evaluate the value of the polynomial
    at x using recursion

    Args:
        x: the point at where the value will be evaluated
        C: coefficient array

    Returns:
        value
    """
    check_C(C)

    if C.size == 1:
        return C[0]
    else:
        return C[0] + x * eval_poly_coeffs(x, C[1:])


def eval_poly_roots(x, R, leading=1):
    """providing polynomial roots, evaluate the value of the polynomial at x

    Args:
        x: the point at where the value will be evaluated
        C: coefficient array

    Returns:
        value
    """
    check_R(R)

    if R.size == 1:
        return (x - R[0]) * leading
    else:
        return (x - R[0]) * eval_poly_roots(x, R[1:], leading)


def der_poly(C):
    """derivative of a polynomial

    Args:
        C: coefficient array

    Returns:
        the coefficient array of the derived polynomial
    """
    check_C(C)

    if C.size == 1:
        return numpy.zeros(1)

    dC = C[1:] * numpy.arange(1, C.size)

    return dC


def int_poly(C, C0=0):
    """undeterministic integral of a polynomial

    Args:
        C: the coefficient array of the polynomial
        C0: the constant term generated from integration, default 0

    Returns:
        the integrated polynomial with constant term C0 (dafault 0)
    """

    check_C(C)

    C = numpy.append([C0], C / numpy.arange(1, C.size+1))

    return C


def find_roots(C, z=None):
    """find the roots of the polynomial, using Durand-Kerner method

    Args:
        C: coefficient array
        z: user-provided initial guesses for all roots

    Returns:
        roots
    """
    # TODO: handle multiple roots

    check_C(C)

    if C.size == 1:
        return None

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
            delta = eval_poly_coeffs(z[i], C) / \
                (numpy.prod(z[i]-z[:i]) * numpy.prod(z[i]-z[i+1:]))
            z[i] -= delta

            if numpy.abs(delta) < 1e-14 or \
                    numpy.abs(eval_poly_coeffs(z[i], C)) < 1e-14:
                stop[i] = True

        N += 1
        if N > 100000:
            print(z)
            raise InfLoopError(100000)

    z = strip_trivial(z)

    return z


def find_coeffs(R):
    """construct/find the coefficients of a polynomial by its roots

    Args:
        R: array of roots

    Returns:
        the coefficient array
    """

    check_R(R)
    n = R.size
    pi = numpy.ones(n, dtype=numpy.complex256)
    pi = numpy.hstack((-R[:, None], pi[:, None]))

    p = numpy.ones(1, dtype=numpy.complex256)
    for i in range(n):
        p = mul_poly(p, pi[i, :])

    p = strip_trivial(p)

    return p


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
        try:
            nC[i:i+C2.size] += c * C2
        except TypeError:
            nC = nC.astype(numpy.complex256)
            nC[i:i+C2.size] += c * C2

    nC = strip_trivial(nC)
    nC = numpy.append(nC[0], numpy.trim_zeros(strip_trivial(nC[1:]), 'b'))

    return nC


def div_poly(C1, C2):
    """polynomial C1 divides by C2, i.e., C1 / C2 = Q + R

    Requirement: the degree of C2 must be lower or equal to C1

    Args:
        C1: coefficient array
        C2: coefficient array

    Returns:
        Q: quotient
        R: remainder
    """
    check_C(C1)
    check_C(C2)

    assert C2.size <= C1.size, \
        "The degree of polynomial C2 must be <= C1"

    Q = numpy.zeros(C1.size - C2.size + 1, dtype=numpy.complex256)
    R = C1.copy()

    i = Q.size
    while R.size >= C2.size:

        Q[i-1] = R[-1] / C2[-1]
        R = numpy.delete(R-mul_poly(Q[:i], C2), -1)
        i -= 1

    Q = strip_trivial(Q)
    R = numpy.append(R[0], numpy.trim_zeros(strip_trivial(R[1:]), 'b'))

    return Q, R


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


def check_R(R):
    """check the type and number of the polynomial root

    Args:
        R: polynomial roots
    """
    assert isinstance(R, numpy.ndarray), \
        "The array of the roots is not a NumPy array"
    assert len(R.shape) == 1, "R is not a 1D array"
    assert R.size > 0, \
        "The number of the polynomial roots should > 0"
