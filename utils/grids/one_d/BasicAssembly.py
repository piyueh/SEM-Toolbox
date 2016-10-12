#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Order the nodes or modes sequentially"""

import numpy
import utils.elems.one_d as elem


class BasicAssembly(object):

    def __init__(self, nElems, ends, expn, P, nodes=None):
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
        self._set_nodes(nodes)
        self._set_expn(expn)
        self._set_mapping()

    def _set_P(self, P):
        """_set_P sets up the order of each element

        Args:
            P: an integer or a list/ndarray of integers
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

    def _set_nodes(self, nodes):
        """_set_nodes sets up the end nodes of each element

        Args:
            nodes: None or a list/ndarray of sorted nodes. The i-th element will
                   be defined by i-th and i+1 -th entities in nodes.
        """
        if nodes is None:
            self.nodes = numpy.linspace(self.ends[0], self.ends[1],
                                        self.nElems+1, dtype=numpy.float64)
        else:
            # TODO: check len(nodes) == self.nElems + 1
            # TODO: check if nodes is sorted
            # TODO: check if the end entities of nodes are self.ends
            self.nodes = numpy.array(nodes, dtype=numpy.float64)

    def _set_expn(self, expn):
        """_set_expn sets up each element

        Args:
            expn: the type of expansion. It is either a string or a list/ndarray
            of strings.

        Returns:
        """
        if isinstance(expn, str):
            if expn in elem.__expns__:
                self.elems = numpy.array(
                    [elem.__expns__[expn](self.nodes[i:i+2], self.P[i]+1)
                     for i in range(self.nElems)],
                    dtype=object)
            else:
                raise ValueError("the chosen expansion is invalid")
        elif isinstance(expn, (list, numpy.ndarray)):
            # TODO: check len(expn) == self.nElems
            # TODO: check each entity in expn should be a valid expansion
            self.elems = numpy.array(
                [elem.__expns__[expn[i]](self.nodes[i:i+2], self.P[i]+1)
                    for i in range(self.nElems)],
                dtype=object)
        else:
            raise ValueError(
                "expn should be a string or a list/ndarray of strings")

    def _set_mapping(self):
        pass
