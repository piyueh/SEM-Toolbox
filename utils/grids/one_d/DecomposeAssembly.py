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


class DecomposeAssembly(BaseAssembly):
    """1D global assembly that decomposes boundary and interior nodes"""

    def _set_mapping(self):
        """_set_mapping sets up local to global index dict

        Results:
            self.l2g: a nested ndarray for looking up the global index of a
                local index e.g. l2g[0][3] will return the global index of the
                3rd node/mode in the 0th element.
        """

        self.l2g = numpy.empty(self.nElems, dtype=numpy.ndarray)

        cnt = self.nElems + 1
        for i, e in enumerate(self.elems):
            self.l2g[i] = numpy.array(
                [i] + list(range(cnt, cnt+e.n_nodes-2)) + [i+1],
                dtype=numpy.int)
            cnt += (e.n_nodes - 2)

        assert cnt == self.nModes, \
            "cnt={0}, self.nModes={1}".format(cnt, self.nModes)
