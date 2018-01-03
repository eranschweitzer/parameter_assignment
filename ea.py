import numpy as np

def mutate(Psi0,K):
    Psi = []
    wmax = 0
    if Psi0 is not None:
        for psi in Psi0:
            wmax = max(wmax,psi.w)
    for i in range(K):
        r = np.random.random()
        if (Psi0 is None) or (r > wmax):
            Psi.append( EAindividual(Z=np.random.permutation(L)) )

class EAindividual(object):
    def __init__(self,Z=None):
        self.Z = Z
        self.Pi= None
        self.w = None
        self.f = None

class EAgeneration(object):
    def __init__(self,K):
        self.Psi = mutate(None,K)

    def order(self):
        """ invert objective and find the largest one """
        C = 0
        for psi in self.Psi:
            psi.w = 1./psi.f
            C = max(wmax,psi.w)
        
        """ sorting order in descending order (hence the -psi.w) """
        idx = np.argsort([-psi.w for psi in self.Psi])
        self.Psi = [self.Psi[i] for i in idx]

        """ normalize by the largest weight and form cumulative weight"""
        for i,psi in enumerate(self.Psi):
            if i > 0:
                psi.w = psi.w/C + self.Psi[i-1].w
            else:
                psi.w = psi.w/C
        """ round sum up to the nearest integer """
        C = np.ceil(self.Psi[-1])

        """ renormalize """
        for psi in self.Psi:
            psi.w = psi.w/C
