# -*- coding: utf-8 -*-

"""Definition of the InfLoopError class"""


from Error import Error


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
