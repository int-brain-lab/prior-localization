'''Copied from Jacob Svensson's website:
http://blog.nephics.com/2019/08/28/better-loadmat-for-scipy/
'''

import numpy as np
import scipy.io


def loadmat(filename):
    """Improved loadmat (replacement for scipy.io.loadmat)
    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908
    """

    def _has_struct(elem):
        """Determine if elem is an array
        and if first array item is a struct
        """
        return isinstance(elem, np.ndarray) and (
            elem.size > 0) and isinstance(
            elem[0], scipy.io.matlab.mio5_params.mat_struct)

    def _check_keys(d):
        """checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            elem = d[key]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """A recursive function which constructs from
        matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the
        elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
