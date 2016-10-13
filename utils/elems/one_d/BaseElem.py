#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of base class of all 1D element"""

import numpy


class BaseElem:
    """Base class for all 1D expansion"""

    def __init__(self, ends, n, tol=1e-12):
        """__init__

        Args:
            ends: array of the two end nodes (their locations)
            n: number of modes in this element
            alpha: the alpha used for Jacobi polynomial
            beta: the beta used for Jacobi polynomial
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
        # TODO: the order of the ends given?

        self.ends = numpy.array(ends, dtype=numpy.float64)
        self.L = ends[1] - ends[0]

        self.n_nodes = n
        self.ui = None

        self._set_expn()
        self._set_mass_mtx(tol)
        self._set_weak_laplacian(tol)

    def __call__(self, x):
        """__call__

        Args:
            x: the location to evaluate values

        Returns:
            values
        """

        return numpy.array([self.expn[i](self.x_to_xi(x)) * self.ui[i]
                            for i in range(self.n_nodes)]).sum(axis=0)

    def _set_expn(self):
        """setting up expansions"""

        self.expn = None

    def _set_mass_mtx(self, tol=1e-12):
        """compute mass matrix"""

        self.M = None

    def _set_weak_laplacian(self, tol=1e-12):
        """compute weak-form laplacian"""

        self.wL = None

    def x_to_xi(self, x):
        """map physical coordinate x to standard coordinate xi

        Args:
            x: physical coordinate, x in [ends[0], ends[1]]

        Returns:
            xi, and xi in [-1, 1]
        """

        return 2. * (x - self.ends[0]) / self.L - 1.

    def xi_to_x(self, xi):
        """map standard coordinate xi to physical coordinate x

        Args:
            xi: standard coordinate, xi in [-1, 1]

        Returns:
            x, and x in [ends[0], ends[1]]
        """

        return (xi + 1) * self.L / 2. + self.ends[0]

    def set_ui(self, ui):
        """setting up the coefficients (ui) of each modes

        After the coefficients ui are solved and known, setting up these
        coefficients to the element so that we cal use the elements to evaluate
        the solution at the locations desired.

        Args:
            ui: the coefficients of modes
        """
        assert isinstance(ui, (list, numpy.ndarray)), \
            "ui is neither a list nor a numpy array"
        assert len(ui) == self.n_nodes, \
            "the lenth of ui is not the same as that of the modes"

        self.ui = numpy.array(ui, dtype=numpy.float64)
