import numpy as np
import gurobipy as gb
import helpers as hlp
import logging

def mycallback(model,where):
    """ Create a callback for termination
        Termination is: 
        MIPgap OR (Time Limit AND solution) OR (Solution)
    """
    if where == gb.GRB.Callback.MIPSOL:
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        Pg           = sum(model.cbGetSolution(model._Pg.values()))
        criteria     = (Pg - model._Pd)/model._Pd
        logging.info('Current solution: solcnt: %d, sum(Pg)=%0.2f, sum(load)=%0.2f, criteria=%0.3g',solcnt, Pg, model._Pd, criteria)
        if criteria < model._lossterm: 
            logging.info('      terminating in MISOL due to minimal losses')
            model.terminate()
    if where == gb.GRB.Callback.MIPSOL:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500): 
            logging.info('      terminating in MISOL')
            model.terminate()
    elif where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIP_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):
            logging.info('      terminating in MIP')
            model.terminate()
    elif where == gb.GRB.Callback.MIPNODE:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPNODE_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):
            logging.info('      terminating in MIPNODE')
            model.terminate()
    else:
        pass

def tieassign(G, nvars, lvars, lossmin, lossterm, fmax, dmax, htheta, umin, umax, ztie, tset):

    N = G.number_of_nodes()
    L = G.number_of_edges()
    E = len(tset)
    bmap = dict(zip(tset,range(E)))
    rbmap = np.empty(E, dtype='int')
    for k,v in bmap.items():
        rbmap[v] = k
    ### get primitive admittance values ####
    # IMPORTANT!!! Yfixed is size Lx1 where the boundary entries will be some arbitrary number
    # that should be ignored
    Yfixed = hlp.Yparts( lvars['r'], lvars['x'], b=lvars['b'] )
    Ytie   = hlp.Yparts( ztie['r'],  ztie['x'],  b=ztie['b'] )
    bigM = hlp.bigM_calc(Ytie,fmax,umax,dmax)

    m = gb.Model()

    m.setParam('LogFile','/tmp/GurobiMultivar.log')
    m.setParam('LogToConsole',0)
    m.setParam('MIPGap',0.15)
    #m.setParam('SolutionLimit',1) #stop after this many solutions are found
    m.setParam('TimeLimit', 1500)
    m.setParam('MIPFocus',1)
    m.setParam('ImproveStartTime',60)
    m.setParam('Threads',60)
   
    #############
    # Variables
    #############
    Z     = m.addVars(E,E,vtype=gb.GRB.BINARY,name="Z")

    theta = m.addVars(N,lb=-gb.GRB.INFINITY, name="theta")
    u     = m.addVars(N,lb=umin, ub=umax,name="u")
    phi   = m.addVars(L,lb=0,ub=dmax*dmax/2,name='phi')
    
    Pf    = m.addVars(L,lb=-fmax, ub=fmax, name="Pf")
    Pt    = m.addVars(L,lb=-fmax, ub=fmax, name="Pt")
    Qf    = m.addVars(L,lb=-fmax, ub=fmax, name="Qf")
    Qt    = m.addVars(L,lb=-fmax, ub=fmax, name="Qt")

    Pg    = m.addVars(N,lb=-gb.GRB.INFINITY, name="Pg")
    Qg    = m.addVars(N,lb=-gb.GRB.INFINITY, name="Qg")

    m._Pg   = Pg
    m._Pd   = sum(nvars['Pd'])
    m._lossterm = lossterm
    d = dmax/htheta
    ###############
    # Constraints
    ###############
    m.addConstr( Pg.sum("*") >= m._Pd*(1+lossmin) )
    for n1,n2,l in G.edges_iter(data='id'):
        m.addConstr( theta[n1] - theta[n2] <=  dmax)
        m.addConstr( theta[n1] - theta[n2] >= -dmax)
        for t in range(htheta+1):
            m.addConstr(phi[l] >= -0.5*(t*d)**2 + (t*d)*(theta[n1] - theta[n2]))
            m.addConstr(phi[l] >= -0.5*(t*d)**2 + (t*d)*(theta[n2] - theta[n1]))
        if l in tset:
            for _l2 in tset:
                l1 = bmap[l]; l2 = bmap[_l2]
                m.addConstr(Pf[l] - Ytie['gff'][l2]*(1+u[n1]) - Ytie['gft'][l2]*(1-phi[l]+u[n2]) + Ytie['bft'][l2]*(theta[n2] - theta[n1]) + bigM*(1 - Z[l1,l2]) >= 0)
                m.addConstr(Pf[l] - Ytie['gff'][l2]*(1+u[n1]) - Ytie['gft'][l2]*(1-phi[l]+u[n2]) + Ytie['bft'][l2]*(theta[n2] - theta[n1]) - bigM*(1 - Z[l1,l2]) <= 0)
                m.addConstr(Qf[l] + Ytie['bff'][l2]*(1+u[n1]) + Ytie['bft'][l2]*(1+phi[l]+u[n2]) - Ytie['gft'][l2]*(theta[n2] - theta[n1]) + bigM*(1 - Z[l1,l2]) >= 0)
                m.addConstr(Qf[l] + Ytie['bff'][l2]*(1+u[n1]) + Ytie['bft'][l2]*(1+phi[l]+u[n2]) - Ytie['gft'][l2]*(theta[n2] - theta[n1]) - bigM*(1 - Z[l1,l2]) <= 0)
                m.addConstr(Pt[l] - Ytie['gtt'][l2]*(1+u[n2]) - Ytie['gtf'][l2]*(1-phi[l]+u[n1]) + Ytie['btf'][l2]*(theta[n1] - theta[n2]) + bigM*(1 - Z[l1,l2]) >= 0)
                m.addConstr(Pt[l] - Ytie['gtt'][l2]*(1+u[n2]) - Ytie['gtf'][l2]*(1-phi[l]+u[n1]) + Ytie['btf'][l2]*(theta[n1] - theta[n2]) - bigM*(1 - Z[l1,l2]) <= 0)
                m.addConstr(Qt[l] + Ytie['btt'][l2]*(1+u[n2]) + Ytie['btf'][l2]*(1+phi[l]+u[n1]) - Ytie['gtf'][l2]*(theta[n1] - theta[n2]) + bigM*(1 - Z[l1,l2]) >= 0)
                m.addConstr(Qt[l] + Ytie['btt'][l2]*(1+u[n2]) + Ytie['btf'][l2]*(1+phi[l]+u[n1]) - Ytie['gtf'][l2]*(theta[n1] - theta[n2]) - bigM*(1 - Z[l1,l2]) <= 0)
        else:
            m.addConstr(Pf[l] - Yfixed['gff'][l]*(1+u[n1]) - Yfixed['gft'][l]*(1-phi[l]+u[n2]) + Yfixed['bft'][l]*(theta[n2] - theta[n1]) == 0)
            m.addConstr(Qf[l] + Yfixed['bff'][l]*(1+u[n1]) + Yfixed['bft'][l]*(1+phi[l]+u[n2]) - Yfixed['gft'][l]*(theta[n2] - theta[n1]) == 0)
            m.addConstr(Pt[l] - Yfixed['gtt'][l]*(1+u[n2]) - Yfixed['gtf'][l]*(1-phi[l]+u[n1]) + Yfixed['btf'][l]*(theta[n1] - theta[n2]) == 0)
            m.addConstr(Qt[l] + Yfixed['btt'][l]*(1+u[n2]) + Yfixed['btf'][l]*(1+phi[l]+u[n1]) - Yfixed['gtf'][l]*(theta[n1] - theta[n2]) == 0)

    m.addConstrs( Pg[i] - nvars['Pd'][i] - sum(Pt[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - sum(Pf[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0 for i in range(N)) 
    m.addConstrs( Qg[i] - nvars['Qd'][i] - sum(Qt[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - sum(Qf[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0 for i in range(N)) 

    m.addConstrs( Pg[i] >=  nvars['Pgmin'][i]/100 for i in range(N) )
    m.addConstrs( Pg[i] <=  nvars['Pgmax'][i]/100 for i in range(N) )
    m.addConstrs( Qg[i] >= -nvars['Qgmax'][i]/100 for i in range(N) )
    m.addConstrs( Qg[i] <=  nvars['Qgmax'][i]/100 for i in range(N) )

    m.addConstrs( Z.sum(i,'*')  == 1 for i in range(E))
    m.addConstrs( Z.sum('*',i)  == 1 for i in range(E))

    ###############
    # Objective
    ###############
    obj = Pg.sum("*") + phi.sum('*')

    ###############
    # Solve
    ###############
    m.setObjective(obj,gb.GRB.MINIMIZE)
    m.optimize(mycallback)

    ### get variables ####
    lvars['r'][rbmap] = hlp.var2mat(ztie['r'], E, perm=Z)
    lvars['x'][rbmap] = hlp.var2mat(ztie['x'], E, perm=Z)
    lvars['b'][rbmap] = hlp.var2mat(ztie['b'], E, perm=Z)
    lvars['Pf']       = hlp.var2mat(Pf, L)
    lvars['Qf']       = hlp.var2mat(Qf, L)
    lvars['Pt']       = hlp.var2mat(Pt, L)
    lvars['Qt']       = hlp.var2mat(Qt, L)
    lvars['phi']      = hlp.var2mat(phi,L)
    nvars['theta']    = hlp.var2mat(theta, N)
    nvars['u']        = hlp.var2mat(u, N)
    nvars['Pg']       = hlp.var2mat(Pg,N)
    nvars['Qg']       = hlp.var2mat(Qg,N)
    return {**nvars, **lvars}

