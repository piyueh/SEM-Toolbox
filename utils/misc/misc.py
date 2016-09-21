#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Some misc functions"""

import numpy
import numbers
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


def check_array(arry, msg="Can't convert input to numpy.ndarray"):
    """check whether the input is a numpy array, and try to convert it

    Args:
        arry: the data to be checked
        msg: the message to be passed to error instance

    Returns:
        arry as a numpy.ndarray

    Raise:
        TypeError, if it fail to convert the input to a numpy array
    """

    if isinstance(arry, (numbers.Number, numpy.number)):
        return numpy.array([arry])
    elif isinstance(arry, list):
        return numpy.array(arry)
    elif isinstance(arry, numpy.ndarray):
        return arry
    else:
        raise TypeError(msg)
