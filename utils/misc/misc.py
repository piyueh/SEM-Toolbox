# -*- coding: utf-8 -*-

"""Some misc functions"""

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
