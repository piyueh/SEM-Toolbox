#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Order the nodes or modes sequentially"""

import numpy
from utils.grids.one_d.BaseAssembly import BaseAssembly


class SequentialAssembly(BaseAssembly):
    """1D global assembly using sequential nodal ordering"""

    def _set_mapping(self):
        """_set_mapping sets up local to global index dict

        Results:
            self.l2g: a nested ndarray for looking up the global index of a
                local index e.g. l2g[0][3] will return the global index of the
                3rd node/mode in the 0th element.
        """

        self.l2g = numpy.empty(self.nElems, dtype=numpy.ndarray)

        self.l2g[0] = numpy.arange(0, self.elems[0].n_nodes)
        cnt = self.elems[0].n_nodes - 1
        for i, e in enumerate(self.elems[1:]):
            self.l2g[i+1] = numpy.arange(cnt, cnt+e.n_nodes)
            cnt += (e.n_nodes - 1)

        assert cnt == self.nModes - 1
