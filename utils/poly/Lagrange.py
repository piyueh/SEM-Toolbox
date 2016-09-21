#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the class of Lagrange polynomial"""
# TODO: documentation
# TODO: __repr__ and __str__

import numpy
from utils.poly import LagrangeBasis
from utils.misc import check_array


class Lagrange(object):
    """Lagrange:"""

    def __init__(self, nodes, values):
        """__init__

        Args:
            nodes: interpolation node
            values: values at interpolation nodes
        """

        check_array(nodes)
        check_array(values)

        assert len(nodes.shape) == 1, "The node array is not a 1D array"
        assert len(values.shape) == 1, "The node array is not a 1D array"
        assert nodes.size == values.size, "nodes.size != values.size"
        # TODO: check whether there are duplicated nodes

        self.n = nodes.size
        self.nodes = nodes
        self.values = values

        self.basis = LagrangeBasis(self.nodes)

    def __call__(self, x):
        """__call__

        Args:
            x: locations at where values will be interpolated

        Returns:
            values from interpolation
        """

        check_array(x)
        return numpy.dot(self.basis(x), self.values)

    def derivative(self, x):
        """derivative

        Args:
            x: locations at where derivatives will be interpolated

        Returns:
            derivatives from interpolation
        """

        check_array(x)
        return numpy.dot(self.basis.derivative(x), self.values)
