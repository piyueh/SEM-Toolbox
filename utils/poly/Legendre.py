#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of Legendre polynomial class"""

from utils.poly.Jacobi import Jacobi


class Legendre(Jacobi):
    """Legendre polynomial

    Its attributions are the same as the Polynomial class.
    """

    def __init__(self, _n):
        """initialize the instance of Jacobi polynomial

        Args:
            _n: the order of the Legendre polynomial
        """

        super().__init__(_n, 0, 0)

    def __repr__(self):
        """__repr__"""

        return "{0}({1})".format(self.__class__, self.n)
