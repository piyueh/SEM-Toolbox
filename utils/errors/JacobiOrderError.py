#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the MIT license.

"""Definition of the JacobiOrderError class"""

from utils.errors.Error import Error


class JacobiOrderError(Error):
    """Error regarding to the order of Jacobi polynomial"""

    def __init__(self, n):
        """__init__

        Args:
            n: the user-input order in Jacobi polynomial-related functions
        """
        self. msg = \
            "The input Jacobi order n={0} is invalid.".format(n)

    def __str__(self):
        """__str__"""
        return self.msg
