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

# TODO: replace assertion with if ... raise


def factorial(n):
    """Naive implementation of factorial

    For serious use, please consider scipy.special.factorial

    Args:
        n: an integer
    Returns:
        n!
    """

    if not isinstance(n, (int, numpy.int_)):
        raise ValueError(
            "n is not an integer: {0}, {1}".format(n, type(n)))

    if n == 0:
        return 1
    else:
        return functools.reduce(lambda x, y: x * y, range(1, n+1))


def factorial_division(bg, end):
    """Naive implementation of factorial division: end! / bg!

    This function is to avoid integer overflow. If end and bg are big, it is
    dangerous to use fractional(end) / fractional(bg) due to the potential of
    integer overflow.

    For serious use, please consider scipy.special.factorial

    Args:
        bg: the beginning integer
        end: the endding integer
    Returns:
        end! / bg!
    """

    if not isinstance(bg, (int, numpy.int_)):
        raise ValueError(
            "bg is not an integer: {0}, {1}".format(bg, type(bg)))
    if not isinstance(end, (int, numpy.int_)):
        raise ValueError(
            "end is not an integer: {0}, {1}".format(end, type(end)))
    if bg < 0:
        raise ValueError("bg can not be smaller than zero!")
    if end < bg:
        raise ValueError(
            "end should larger than or equal to bg: " +
            "bg={0}, end={1}".format(bg, end))

    if end == bg:
        return 1
    else:
        return functools.reduce(lambda x, y: x * y, range(bg+1, end+1))


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
