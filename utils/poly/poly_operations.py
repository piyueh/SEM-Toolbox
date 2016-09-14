#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Operations of polynomial class"""

import numpy


def eval_poly(x, C, method="direct"):
    """Evaluate the value of a polynomial at x

    Args:
        x: the point at where the value will be evaluated
        C: coefficient array
        method: direct of recursive

    Returns:
        value
    """

    if method == "direct":
        return eval_poly_direct(x, C)
    elif method == "recursive":
        return eval_poly_recursive(x, C)
    else:
        raise ValueError("method should be either direct or recursive")


def eval_poly_direct(x, C):
    """evaluate the value of the polynomial at x using direct calculation

    Args:
        x: the point at where the value will be evaluated
        C: coefficient array

    Returns:
        value
    """
    assert isinstance(C, numpy.ndarray), \
        "The coefficient array is not a NumPy array"
    assert len(C.shape) == 1, "_coeffs is not a 1D array"

    return numpy.sum(numpy.array([x**i for i in range(C.size)]) * C)


def eval_poly_recursive(x, C):
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
    assert len(C.shape) == 1, "_coeffs is not a 1D array"

    if C.size == 1:
        return numpy.zeros(1)
    dC = numpy.zeros(C.size - 1)
    for i, c in enumerate(C[1:]):
        dC[i] = c * (i + 1)

    return dC
