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

    def __init__(self, n, ends, expn, nodes=None, expnList=None):
        assert isinstance(n, (int, numpy.int_)), \
            "the number of elements, n, is not an integer"
        assert n >= 1, \
            "the number of elements, n, should be larger than 1"
        assert isinstance(ends, (numpy.ndarray, list)), \
            "ends is neither a numpy array nor a list"
        assert len(ends) == 2, \
            "the size of end nodes array should be two"
        assert expn in elem.__expns__, \
            "the type of chosen local expanstion is invalid"

        self.n = n
        self.ends = numpy.array(ends)

        if nodes is None:
            pass
        else:
            assert isinstance(nodes, (numpy.ndarray, list)), \
                "nodes is neither a numpy array nor a list"
            assert len(nodes) == n, \
                "the size of nodes array should be the same as n"

        if expnList is None:
            pass
        else:
            assert isinstance(expnList, (numpy.ndarray, list)), \
                "expnList is neither a numpy array nor a list"
            assert len(expnList) == n, \
                "the size of expnList array should be the same as n"
