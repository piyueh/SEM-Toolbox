#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Some misc functions"""

import numpy
import functools


def factorial(n):
    """Naive implementation of factorial

    For serious use, please consider scipy.special.factorial

    Args:
        n: an integer
    Returns:
        n!
    """

    assert isinstance(n, int), "input is not an integer"
    if n == 0:
        return 1
    else:
        return functools.reduce(lambda x, y: x * y, range(1, n+1))


def gamma(n):
    """Naive implementation of gamma function (integer input)

    For serious use, please consider scipy.special.gamma

    Args:
        n: the integer
    Returns:
        (n-1)!
    """
    return factorial(n-1)


def strip_trivial(z, tol=1e-8):
    """if any element in array z is smaller than tol, we set it to zero

    Args:
        z: the array to be cleaned
        tol: the tolerance

    Returns:
    """
    # TODO implement different way to lower the dependence of numpy
    z = z.astype(numpy.complex128)
    z = numpy.where(numpy.abs(z.real) < tol, z.imag*1j, z)
    z = numpy.where(numpy.abs(z.imag) < tol, z.real, z)
    z = numpy.real(z) if (z.imag == 0).all() else z

    return z
