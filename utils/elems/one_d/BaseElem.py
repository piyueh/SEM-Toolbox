#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of base class of all 1D element"""

import numpy
import utils.quadrature as quad


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
        self._set_derivative()
        self._set_mass_mtx(tol)
        self._set_weak_laplacian(tol)
        self._scale_weak_laplacian()

    def __call__(self, x):
        """__call__

        Args:
            x: the location to evaluate values

        Returns:
            values
        """

        if self.ui is not None:
            try:
                return numpy.array(
                    [self._call_single(xi) for i, xi in enumerate(x)],
                    dtype=numpy.float64)
            except TypeError:
                try:
                    return self._call_single(x)
                except:
                    raise
            except:
                raise
        else:
            raise ValueError("the coeffs has not been set")

    def _call_single(self, x):
        """__call__

        Args:
            x: the location to evaluate values

        Returns:
            values
        """
        # TODO: check the type of x

        if x < self.ends[0] or x > self.ends[1]:
            raise ValueError(
                "the input location is outside the domain. " +
                "The input is {0}, where the domain is ".format(x) +
                "[{0}, {1}].".format(self.ends[0], self.ends[1]))

        if x == self.ends[0]:
            return self.ui[0]

        if x == self.ends[-1]:
            return self.ui[-1]

        return numpy.array([self.expn[i](self.x_to_xi(x)) * self.ui[i]
                            for i in range(self.n_nodes)]).sum(axis=0)

    def _set_expn(self):
        """setting up expansions"""

        self.expn = None

    def _set_mass_mtx(self, tol=1e-12):
        """compute mass matrix"""

        self.M = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                p = (self.expn[i] * self.expn[j]).integral()
                self.M[i, j] = p(1) - p(-1)

        Mmax = numpy.max(self.M)
        self.M = numpy.where(numpy.abs(self.M/Mmax) <= tol, 0, self.M)

    def _set_weak_laplacian(self, tol=1e-12):
        """compute weak-form laplacian"""

        self.wL = numpy.matrix(numpy.zeros((self.n_nodes, self.n_nodes)))

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                p = (self.expn[i].derive() * self.expn[j].derive()).integral()
                self.wL[i, j] = p(1) - p(-1)

        wLmax = numpy.max(self.wL)
        self.wL = numpy.where(numpy.abs(self.wL/wLmax) <= tol, 0, self.wL)

    def _scale_weak_laplacian(self):
        """_scale_weak_laplacian scales the weak Laplacian"""

        self.wL *= 4
        self.wL /= (self.L * self.L)

    def weak_rhs(self, f, Q=None, M="GaussLobattoJacobi", **kwargs):
        """weak_rhs calculates the weak RHS with given RHS function f

        Args:
            f: RHS function in the tergeting problem
            Q: the number of quadrature points
            M: the quadrature method. Available methods: Gauss-Jacobi, Gauss-
                Radau-Jacobi, and Gauss-Lobatto-Jacobi
            kwargs: optional arguments for chosen quadrature method

        Returns:
            a ndarray representing elemental weak RHS
        """

        if Q is None:
            Q = self.n_nodes + 1  # use P + 2 quadrature points for safety

        qd = quad.__methods__[M](Q, **kwargs)

        fi = numpy.zeros(self.n_nodes, dtype=numpy.float64)

        for i, expn in enumerate(self.expn):
            def integrand(x):
                return self.expn[i](x) * f(self.xi_to_x(x))
            fi[i] = qd(integrand)

        return fi

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

    def derivative(self, x):
        """calculate derivatives at locations x

        Args:
            x: the location to evaluate derivatives

        Returns:
            derivatives at x
        """

        if self.ui is not None:
            try:
                return numpy.array(
                    [self._derivative_single(xi) for i, xi in enumerate(x)],
                    dtype=numpy.float64)
            except TypeError:
                try:
                    return self._derivative_single(x)
                except:
                    raise
            except:
                raise
        else:
            raise ValueError("the coeffs has not been set")

    def _derivative_single(self, x):
        """_derivative_single calculates the derivative at a single location x

        Args:
            x: the location at where the derivative to be calculated

        Returns: the derivative
        """
        # TODO: check the type of x

        if x < self.ends[0] or x > self.ends[1]:
            raise ValueError(
                "the input location is outside the domain. " +
                "The input is {0}, where the domain is ".format(x) +
                "[{0}, {1}].".format(self.ends[0], self.ends[1]))

        return 2 * numpy.array([self.d_expn[i](self.x_to_xi(x)) * self.ui[i]
                                for i in range(self.n_nodes)]).sum(0) / self.L

    def _set_derivative(self):
        """_set_derivative compute the derivatives of expansions"""

        self.d_expn = numpy.array([e.derive() for e in self.expn])
