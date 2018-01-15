import numpy as np

def getvars(solvers,N,L):
    nvars = {}
    lvars = {}
    #nvars['Pgmax'] = np.empty(N)
    #nvars['Pgmin'] = np.empty(N)
    #nvars['Qgmax'] = np.empty(N)
    #nvars['Pd']    = np.empty(N)
    #nvars['Qd']    = np.empty(N)
    #nvars['Pg']    = np.empty(N)
    #nvars['Qg']    = np.empty(N)
    #nvars['theta'] = np.empty(N)
    #nvars['u']     = np.empty(N)
    #lvars['r']     = np.empty(L)
    #lvars['x']     = np.empty(L)
    #lvars['b']     = np.empty(L)
    #lvars['phi']   = np.empty(L)
    for k,v in solvers[0].getvars().items():
        if v.shape[0] == solvers[0].N:
            nvars[k] = np.empty(N)
        elif v.shape[0] == solvers[0].L:
            lvars[k] = np.empty(L)
    return vars_update(nvars,lvars,solvers)

def vars_update(nvars, lvars, solvers):
    for s in solvers:
        vars = s.getvars()
        #### node variables
        for k in nvars:
            nvars[k][s.rnmap] = vars[k] 
        #### branch variables
        for k in lvars:
            lvars[k][s.rlmap] = vars[k]
    return nvars,lvars 
