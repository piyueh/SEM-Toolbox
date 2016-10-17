#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Base class of all 1D global assembly"""

import numpy
import utils.elems.one_d as elem


class BaseAssembly(object):
    """Base class of all 1D global assembly"""

    def __init__(self, nElems, ends, expn, P, endNodes=None):
        """__init__ initialize the class

        Args:
            nElems: number of elements
            ends: a list/ndarray of two floating numbers;the ends of this one-D
                domain. It should be sorted, i.e., ends[0] < ends[1]
            expn: either a string or a list/ndarray of strings indicating the
                expansions used for all elements. If a single string is
                provides, all elements will use the same expansion.
            P: either a single integer or a list/ndarray of integers indicating
                the polynomial orders of all elements. If a single integer is
                provided, all elements will use the same order.
            endNodes: an optional argument; a list/ndarray of floating numbers.
                This list/ndarray indicates the location of end nodes of each
                element. This endNodes should be sorted. If endNodes is not
                provided, the elements will be assumed to have equal length
                in the interval [ends[0], ends[1]]
        """
        assert isinstance(nElems, (int, numpy.int_)), \
            "the number of elements, nElems, is not an integer"
        assert nElems >= 1, \
            "the number of elements, nElems, should be larger than 1"
        assert isinstance(ends, (numpy.ndarray, list)), \
            "ends is neither a numpy array nor a list"
        assert len(ends) == 2, \
            "the size of end nodes array should be two"

        self.nElems = nElems
        self.ends = numpy.array(ends)

        self._set_P(P)
        self._set_endNodes(endNodes)
        self._set_expn(expn)
        self._set_elems()
        self._set_miscs()
        self._set_mapping()
        self._set_global_mass_mtx()
        self._set_global_weak_laplacian()

    def _set_P(self, P):
        """_set_P sets up the order of each element

        Args:
            P: an integer or a list/ndarray of integers

        Results:
            self.P: a ndarray representing the orders of the all elements
        """
        if isinstance(P, (int, numpy.int_)):
            # TODO: check P is larger than 0
            self.P = numpy.ones(self.nElems, dtype=int) * P
        elif isinstance(P, (list, numpy.ndarray)):
            # TODO: check len(P) == self.nElems
            # TODO: check the type of each entity in P is int
            self.P = numpy.array(P, dtype=int)
        else:
            raise ValueError(
                "P should be a single integer or a list/array of integers")

    def _set_endNodes(self, endNodes):
        """_set_nodes sets up the end nodes of each element

        Args:
            nodes: None or a list/ndarray of sorted nodes. The i-th element will
                   be defined by i-th and i+1 -th entities in nodes.

        Results:
            self.endNodes: a ndarray of end nodes of each elements
        """
        if endNodes is None:
            self.endNodes = numpy.linspace(self.ends[0], self.ends[1],
                                           self.nElems+1, dtype=numpy.float64)
        else:
            # TODO: check len(endNodes) == self.nElems + 1
            # TODO: check if endNodes is sorted
            # TODO: check if the end entities of endNodes are self.ends
            self.endNodes = numpy.array(endNodes, dtype=numpy.float64)

    def _set_expn(self, expn):
        """_set_expn sets up self.expns

        Args:
            expn: the type of expansion. It is either a string or a list/ndarray
            of strings.

        Results:
            self.expns: a ndarray of the type of expansion in each element
        """
        if isinstance(expn, str):
            if expn in elem.__expns__:
                self.expns = numpy.array([expn]*self.nElems, dtype=str)
            else:
                raise ValueError("the chosen expansion is invalid")
        elif isinstance(expn, (list, numpy.ndarray)):
            # TODO: check len(expn) == self.nElems
            # TODO: check each entity in expn should be a valid expansion
            self.expns = numpy.array(expn, dtype=str)
        else:
            raise ValueError(
                "expn should be a string or a list/ndarray of strings")

    def _set_elems(self):
        """_set_elems sets up each element

        Results:
            self.elems: a ndarray holding element classes
        """
        # TODO: check len(expn) == self.nElems
        # TODO: check each entity in expn should be a valid expansion
        self.elems = numpy.array(
            [elem.__expns__[self.expns[i]](self.endNodes[i:i+2], self.P[i]+1)
             for i in range(self.nElems)],
            dtype=object)

    def _set_miscs(self):
        """_set_miscs sets up miscs attributes

        Results:
            self.nModes: an integer of the total number of modes
            self.coeffs: a ndarray holding coefficients of all modes. It is
                initialized with zeros here. Users should use set_coeffs to set
                it up after solving the coefficients.
        """

        self.nModes = self.P.sum() + 1
        self.coeffs = None

    def _set_mapping(self):
        """_set_mapping sets up local to global index dict

        Results:
            self.l2g: a nested ndarray for looking up the global index of a
                local index e.g. l2g[0][3] will return the global index of the
                3rd node/mode in the 0th element.
        """

        self.l2g = numpy.empty(self.nElems, dtype=numpy.ndarray)

    def _set_global_mass_mtx(self):
        """_set_global_mass_mtx sets up the global mass matrix

        Results:
            self.M: a numpy matrix of the global mass matrix
        """
        # TODO: use sparse matrix in Scipy instead

        self.M = numpy.matrix(numpy.zeros((self.nModes, self.nModes)),
                              dtype=numpy.float64)

        for i, e in enumerate(self.elems):
            i, j = numpy.meshgrid(self.l2g[i], self.l2g[i])
            self.M[j, i] += e.M

    def _set_global_weak_laplacian(self):
        """_set_global_weak_laplacian sets up global weak Laplacian

        Results:
            self.wL: a numpy matrix of the global mass matrix
        """
        # TODO: use sparse matrix in Scipy instead

        self.wL = numpy.matrix(numpy.zeros((self.nModes, self.nModes)),
                               dtype=numpy.float64)

        for i, e in enumerate(self.elems):
            i, j = numpy.meshgrid(self.l2g[i], self.l2g[i])
            self.wL[j, i] += e.wL

    def set_coeffs(self, coeffs):
        """set_coeffs sets up the coefficients after solving

        Args:
            coeffs: a list/ndarray of the coefficients

        Results:
            self.coeffs: a list/ndarray of the coefficients
        """
        # TODO: check the length and type of coeffs

        self.coeffs = numpy.array(coeffs, dtype=numpy.float64)

        for i in range(self.nElems):
            self.elems[i].set_ui(self.coeffs[self.l2g[i]])

    def weak_rhs(self, f, Q=None):
        """weak_rhs returns global weak RHS with given rhs function f

        Args:
            f: rhs function in targeting problems
            Q: number of quadrature points

        Returns:
            a ndarray representing global weak RHS vector
        """

        fi = numpy.zeros(self.nModes, dtype=numpy.float64)

        for i, e in enumerate(self.elems):
            fi[self.l2g[i]] += e.weak_rhs(f, Q)

        return fi

    def __call__(self, x):
        """__call__ calculate values at locations x

        Args:
            x: a single floating number or a list/ndarray of floating numbers.
                x represents locations at where the values will be calculated.

        Returns:
            values
        """

        if self.coeffs is not None:
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

    def derivative(self, x):
        """__call__ calculate values at locations x

        Args:
            x: a single floating number or a list/ndarray of floating numbers.
                x represents locations at where the values will be calculated.

        Returns:
            values
        """

        if self.coeffs is not None:
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

    def _call_single(self, x):
        """_call_single calculate the value at location x

        Args:
            x: the location of the value at where to be calculated

        Returns: the value
        """
        # TODO: check the type of x

        if x < self.ends[0] or x > self.ends[1]:
            raise ValueError(
                "the input location is outside the domain. " +
                "The input is {0}, where the domain is ".format(x) +
                "[{0}, {1}].".format(self.ends[0], self.ends[1]))

        if x == self.ends[0]:
            return self.elems[0](x)

        i = numpy.searchsorted(self.endNodes, x) - 1
        return self.elems[i](x)

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

        if x == self.ends[0]:
            i = 0
        else:
            i = numpy.searchsorted(self.endNodes, x) - 1

        return self.elems[i].derivative(x)

    def __str__(self):
        """__str__"""

        s = "Dimension: 1\n"
        s += "Ordering: baisc (sequential)\n"
        s += "Number of elements: {0}\n".format(self.nElems)
        s += "Coefficients are known: {0}\n".format(self.coeffs is not None)

        return s

    def __repr__(self):
        """__repr__"""

        return "{0}({1}, {2}, {3}, {4})".format(self.__class__, self.nElems,
                                                self.ends, self.expns, self.P)
