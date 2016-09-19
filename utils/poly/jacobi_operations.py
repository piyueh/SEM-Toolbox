#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""poly.py includes functions related to Jacobi polynomials."""

import numpy
from utils.misc import gamma, factorial
from utils.errors.InfLoopError import InfLoopError


def jacobi_r(x, n, alpha, beta):
    """Evaluate values of Jacobi polynomials at x using recursion

    Args:
        x: the location where the value will be evaluated
        n: the order of Jacobi polynomial
        alpha: the alpha parameter of Jacobi polynomial
        beta: the beta parameter of Jacobi polynomial
    Returns:
        the values of Jacobi polynomial at x
    Raises:
        None
    """

    jacobi_check(n, alpha, beta)

    if n == 0:
        return 1.
    elif n == 1:
        return 0.5 * (alpha - beta + (alpha + beta + 2.) * x)
    else:
        n -= 1

        c1 = alpha + beta
        np1 = n + 1
        nt2 = 2 * n
        np1t2 = 2 * np1
        nt2p1 = nt2 + 1

        a1 = np1t2 * (np1 + c1) * (nt2 + c1)
        a2 = (nt2p1 + c1) * c1 * (alpha - beta)
        a3 = (nt2 + c1) * (nt2p1 + c1) * (np1t2 + c1)
        a4 = 2 * (n + alpha) * (n + beta) * (np1t2 + c1)

        return ((a2 + a3 * x) * jacobi_r(x, n, alpha, beta) -
                a4 * jacobi_r(x, n - 1, alpha, beta)) / a1


def jacobi_coef(n, alpha, beta):
    """Calculate the coefficients of terms in a Jacobi polynomial

    That is, if we write the polynomial as the vector form:
        P(x, n, a, b) = [C0, C1, ..., Cn] x [1, x, ..., x**n]^T
    then [C0, C1, ..., Cn] is right the array returned by this function.

    Args:
        n: the order of Jacobi polynomial
        alpha: the alpha parameter of Jacobi polynomial
        beta: the beta parameter of Jacobi polynomial
    Returns:
        the coefficient array
    """
    if n == 0:
        return numpy.array([1.], dtype=numpy.float64)
    elif n == 1:
        return numpy.array([0.5*(alpha-beta), 0.5*(alpha+beta+2.)],
                           dtype=numpy.float64)
    else:
        n -= 1

        c1 = alpha + beta
        np1 = n + 1
        nt2 = 2 * n
        np1t2 = 2 * np1
        nt2p1 = nt2 + 1

        a1 = np1t2 * (np1 + c1) * (nt2 + c1)
        a2 = (nt2p1 + c1) * c1 * (alpha - beta)
        a3 = (nt2 + c1) * (nt2p1 + c1) * (np1t2 + c1)
        a4 = 2 * (n + alpha) * (n + beta) * (np1t2 + c1)

        return (a3 * numpy.append([0.], jacobi_coef(n, alpha, beta)) +
                a2 * numpy.append(jacobi_coef(n, alpha, beta), [0.]) -
                a4 * numpy.append(jacobi_coef(n-1, alpha, beta), [0., 0.])) / a1


def jacobi_r_d1(x, n, alpha, beta):
    """Evaluate the first derivative of Jacobi polynomial at x using resursion

    Args:
        x: the location where the value will be evaluated
        n: the order of Jacobi polynomial
        alpha: the alpha parameter of Jacobi polynomial
        beta: the beta parameter of Jacobi polynomial
    Returns:
        the first derivative of Jacobi polynomial at x
    Raises:
        None
    """
    jacobi_check(n, alpha, beta)

    if n == 0:
        return 0.
    elif n == 1:
        return 0.5 * (alpha + beta + 2.)
    else:
        c1 = 2 * n + alpha + beta
        c2 = alpha - beta

        b1 = c1 * (1. - x * x)
        b2 = n * (c2 - c1 * x)
        b3 = 2 * (n + alpha) * (n + beta)

        return (b2 * jacobi_r(x, n, alpha, beta) +
                b3 * jacobi_r(x, n - 1, alpha, beta)) / b1


def jacobi_d1(x, n, alpha, beta):
    """Evaluate the first derivative of Jacobi polynomial at x using eq. A.1.8

    Args:
        x: the location where the value will be evaluated
        n: the order of Jacobi polynomial
        alpha: the alpha parameter of Jacobi polynomial
        beta: the beta parameter of Jacobi polynomial
    Returns:
        the first derivative of Jacobi polynomial at x
    Raises:
        None
    """
    jacobi_check(n, alpha, beta)

    if n == 0:
        return 0.
    else:
        return 0.5 * (alpha + beta + n + 1) * \
            jacobi_r(x, n - 1, alpha + 1, beta + 1)


def jacobi_roots(n, alpha, beta):
    """Return the roots of Jacobi polynomial

    Args:
        n: the order of Jacobi polynomial
        alpha: the alpha parameter of Jacobi polynomial
        beta: the beta parameter of Jacobi polynomial
    Returns:
        a numpy array of roots
    Raises:
        InfLoopError: the iteration number exceeds 1000
    """
    jacobi_check(n, alpha, beta)

    z = numpy.array([], dtype=numpy.float64)

    def s(x): return numpy.sum(1. / (x - z))

    if n == 1:
        z = numpy.append(z, 0.)
    else:
        for k in range(n):

            r = - numpy.cos((2 * k + 1) * numpy.pi * 0.5 / n)

            if (k > 0):
                r = (r + z[k - 1]) / 2.

            N = 0
            delta = 1e10
            while delta > 1e-14:
                delta = - jacobi_r(r, n, alpha, beta) /\
                    (jacobi_d1(r, n, alpha, beta) -
                     jacobi_r(r, n, alpha, beta) * s(r))
                r += delta
                N += 1
                if N > 1000:
                    raise InfLoopError(1000)

            if numpy.abs(r) < 1e-14:
                r = 0.
            z = numpy.append(z, r)

    return z


def jacobi_weights(roots, n, alpha, beta):
    """Calculate the weight of each root

    Args:
        n: the order of the polynomial
        alpha: the alpha parameter in Jacobi polynomials
        beta: the beta parameter in Jacobi polynomials
    """
    jacobi_check(n, alpha, beta)
    jacobi_roots_check(roots, n, alpha, beta)

    c1 = alpha + beta + 1
    ans = (2**c1) * gamma(alpha+n+1) * gamma(beta+n+1)
    ans /= (factorial(n) * gamma(c1+n))
    ans /= (1-(roots**2))
    ans /= (jacobi_d1(roots, n, alpha, beta)**2)

    return ans


def jacobi_check(n, alpha, beta):
    """Check the parameters used in Jacobi polynomials

    Args:
        n: the order of the polynomial
        alpha: the alpha parameter in Jacobi polynomials
        beta: the beta parameter in Jacobi polynomials
    Returns:
        None
    Raises:
        None
    """
    assert alpha >= -1, "alpha should be larger than/equal to -1"
    assert beta >= -1, "beta should be larger than/equal to -1"
    assert n >= 0, "n should be larger than/equal to 1"


def jacobi_roots_check(roots, n, alpha, beta):
    """Check whether the roots of a Jacobi polynomial are correct.

    Args:
    """
    pass
