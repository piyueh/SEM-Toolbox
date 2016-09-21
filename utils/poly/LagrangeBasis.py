#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the class representing Lagrange basis"""
# TODO: complete documentation
# TODO: find better way to carry out calculations. Current way is pretty naieve.

import numpy
import numbers
from utils.poly import Polynomial


class LagrangeBasisDenominator(object):
    """This class should be private"""

    def __init__(self, node, dp):
        """__init__

        Args:
            node: a node used in Lagrange polynomial
        """
        self.node = node
        self.dp = dp
        self.denominator = Polynomial(roots=self.node) * self.dp
        self.denominator2 = Polynomial(roots=[self.node, self.node]) * self.dp

    def __call__(self, x):
        """__call__

        Args:
            x: the location at where the value will be evaluated

        Returns:
            inf is x is exactly the node
            else, the value

        Note:
            no matter how many returned values there are, this function always
            return a numpy.ndarray
        """

        return numpy.vectorize(self._call_single)(x)

    def _call_single(self, x):
        """__call_single

        Args:
            x: the location at where the value will be evaluated

        Returns:
            inf is x is exactly the node
            else, the value
        """

        if numpy.abs(self.denominator(x)) <= 1e-12:
            return numpy.inf
        else:
            return 1. / self.denominator(x)

    def derivative(self, x):

        if numpy.abs(self.denominator(x)) <= 1e-12:
            return numpy.inf
        else:
            return 1. / self.denominator2(x)


class LagrangeBasis(object):
    """LagrangeBasis"""

    def __init__(self, nodes):
        """__init__

        Args:
            nodes:

        Returns:
        """
        # TODO: check nodes type and shape
        # TODO: check duplicated nodes

        self.nodes = nodes
        self.p = Polynomial(roots=self.nodes)
        self.dp = self.p.derive()
        self.ddp = self.dp.derive()
        self.dpi = self.dp(self.nodes)
        self.ddpi = self.ddp(self.nodes)

        self.basis = numpy.array(
            [LagrangeBasisDenominator(nd, dpi)
             for nd, dpi in zip(self.nodes, self.dpi)],
            dtype=LagrangeBasisDenominator)

    def __call__(self, x):
        """__call__

        Args:
            x:

        Returns:
        """

        if isinstance(x, (numbers.Number, numpy.number)):
            return self._call_single(x)
        elif isinstance(x, (numpy.ndarray, list)):
            # TODO: check whether x is 1D array
            return numpy.array([self._call_single(xi) for xi in x])

    def _call_single(self, x):
        """__call_single

        Args:
            x:

        Returns:
        """

        result = numpy.array([f(x) for f in self.basis])
        if numpy.inf in result:
            numpy.place(result, result != numpy.inf, 0.)
            numpy.place(result, result == numpy.inf, 1.)
        else:
            result *= self.p(x)

        return result

    def derivative(self, x):
        """derivative

        Args:
            x:

        Returns:
        """

        if isinstance(x, (numbers.Number, numpy.number)):
            return self._derivative_single(x)
        elif isinstance(x, (numpy.ndarray, list)):
            # TODO: check whether x is 1D array
            return numpy.array([self._derivative_single(xi) for xi in x])

    def _derivative_single(self, x):
        """__derivative_single

        Args:
            x:

        Returns:
        """

        result = numpy.array([f.derivative(x) for f in self.basis])

        if numpy.inf in result:
            result = numpy.array([f(x) for f in self.basis])
            result = numpy.where(
                result == numpy.inf, 0.5*self.ddpi/self.dpi, self.dp(x)*result)
        else:
            result *= (self.dp(x) * (x - self.nodes) - self.p(x))

        return result
