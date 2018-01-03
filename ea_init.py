import copy
import numpy as np

def zperm_init(z,K):
    L = len(z['x'])
    ens = []
    for i in range(K):
        x = np.random.permutation(L)
        _z = copy.copy(z)
        for k in z.keys():
            try:
                if len(z[k]) == L:
                    _z[k] = z[k][x]
                else:
                    _z[k] = z[k]
            except TypeError:
                _z[k] = z[k]
        ens.append(copy.copy(_z))
        del(_z)
    return ens

def zmutate(z,idx):
    """ idx are indecies that should be permuted
    IMPORTAT: THIS WILL MODIFY THE INPUT!!!!!
    """

    L = len(z['x'])
    for i in idx:
        j = np.random.randint(0,L)
        for k in z.keys():
            try:
                if len(z[k]) == L:
                    _tmp    = z[k][i]
                    z[k][i] = z[k][j]
                    z[k][j] = _tmp
            except TypeError:
                pass
