#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the InfLoopError class"""

from utils.errors.Error import Error


class InfLoopError(Error):
    """Error about infinite loop

    Args:
        nlimt: the limit of iteration number in the loop
    """

    def __init__(self, nlimt):
        self.msg = "The number of iteration during a while" +\
            "loop exceeds its limit: {0}".format(nlimt)

    def __str__(self):
        return self.msg
