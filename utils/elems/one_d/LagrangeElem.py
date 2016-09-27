#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of standard element of p-type Lagrange polynomials expansions"""

import numpy
from utils.poly import Polynomial


class LagrangeElem:
    """General class for p-type modal expansion using Lagrange polynomial

    If user only provide number of nodes and end nodes, this will create an
    element based on equally spaced lagrange polynomial.

    Note:
    The mass matrix is calculated through direct polynomial integration. So
    the mass matrix is full no matter what kind of nodal distribution used.
    To obtain diagonal mass matrix, users have to perform lumping by themselves.
    """

    def __init__(self, ends, n, nodes=None, tol=1e-12):
        """__init__

        Args:
            ends: array of the two end nodes (their locations)
            n: number of modes in this element
            nodes: the locations of nodes, can be omitted if equally spaced
                   nodes is desired
            tol: tolerance for entities in mass matrix to be treat as zeros
        """

        assert isinstance(ends, (numpy.ndarray, list)), \
            "ends is neither a numpy array nor a list"
        assert len(ends) == 2, \
            "the size of end nodes array should be two"
        assert isinstance(n, (int, numpy.int_)), \
            "the number of nodes, n, is not an integer"
        assert n >= 2, \
            "the number of nodes, n, should be >= 2"

        self.ends = numpy.array(ends, dtype=numpy.float64)
        self.L = numpy.abs(ends[1] - ends[0])

        self.n_nodes = n

        if nodes is not None:
            assert isinstance(ends, (numpy.ndarray, list)), \
                "nodes is neither a numpy array nor a list"
            assert len(nodes) == n, \
                "the size of nodes array should be equal to n"

            self.nodes = numpy.array(nodes)
        else:
            self.nodes = numpy.linspace(ends[0], ends[1], n)

        self._set_expn()
        self._set_mass_mtx(tol)

    def _set_expn(self):
        """set up expansion polynomials"""

        self.expn = numpy.array([None]*(self.n_nodes), dtype=Polynomial)

        for i in range(self.n_nodes):
            self.expn[i] = Polynomial(roots=numpy.delete(self.nodes, i))
            self.expn[i] /= self.expn[i](self.nodes[i])

    def _set_mass_mtx(self, tol=1e-12):
        """set up the mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                p = (self.expn[i] * self.expn[j]).integral()
                self.M[i, j] = p(1) - p(-1)

        Mmax = numpy.max(self.M)
        self.M = numpy.where(numpy.abs(self.M/Mmax) <= tol, 0, self.M)
