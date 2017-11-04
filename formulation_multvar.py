import sys
import itertools
import gurobipy as gb
import numpy as np
from scipy import sparse
import networkx as nx
import helpers as hlp
import logging

def mycallback(model,where):
    """ Create a callback for termination
        Termination is: 
        MIPgap OR (Time Limit AND solution) OR (Solution)
    """
    if where == gb.GRB.Callback.MIPSOL:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT)
        obj          = model.cbGet(gb.GRB.Callback.MIPSOL_OBJBST)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500) or ((obj - model._pload) < 1e-2):
            logging.info('      terminating in MISOL')
            model.terminate()
    elif where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIP_SOLCNT)
        obj          = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500) or ((obj - model._pload) < 1e-2):
            logging.info('      terminating in MIP')
            model.terminate()
    elif where == gb.GRB.Callback.MIPNODE:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPNODE_SOLCNT)
        obj          = model.cbGet(gb.GRB.Callback.MIPNODE_OBJBST)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500) or ((obj - model._pload) < 1e-2):
            logging.info('      terminating in MIPNODE')
            model.terminate()
    else:
        pass

def single_system(G,fmax,dmax,htheta,umin,umax,z,S,bigM):

    N = G.number_of_nodes()
    L = G.number_of_edges()

    ### get primitive admittance values ####
    Y = hlp.Yparts(z['r'],z['x'],b=z['b'])

    m = gb.Model()
    m.setParam('LogFile','/tmp/GurobiMultivar.log')
    m.setParam('LogToConsole',0)
    m.setParam('MIPGap',0.15)
    #m.setParam('SolutionLimit',1) #stop after this many solutions are found
    m.setParam('TimeLimit', 1500)
    m.setParam('MIPFocus',1)
    m.setParam('ImproveStartTime',60)
    m.setParam('Threads',60)
   
    m._pload = sum(S['Pd']*100)
    #############
    # Variables
    #############
    Pi    = m.addVars(N,N,vtype=gb.GRB.BINARY,name="Pi")
    Z     = m.addVars(L,L,vtype=gb.GRB.BINARY,name="Z")

    theta = m.addVars(N,lb=-gb.GRB.INFINITY, name="theta")
    u     = m.addVars(N,lb=umin, ub=umax,name="u")
    phi   = m.addVars(L,lb=0,ub=dmax*dmax/2,name='phi')
    
    Pd    = m.addVars(N,lb=-gb.GRB.INFINITY, name="Pd")
    Qd    = m.addVars(N,lb=-gb.GRB.INFINITY, name="Qd")
    Pg    = m.addVars(N,lb=-gb.GRB.INFINITY, name="Pg")
    Qg    = m.addVars(N,lb=-gb.GRB.INFINITY, name="Qg")
    Qgp   = m.addVars(N,lb=0, name="Qgp")
    Qgn   = m.addVars(N,lb=0, name="Qgn")

    Pf    = m.addVars(L,lb=-fmax, ub=fmax, name="Pf")
    Pt    = m.addVars(L,lb=-fmax, ub=fmax, name="Pt")
    Qf    = m.addVars(L,lb=-fmax, ub=fmax, name="Qf")
    Qt    = m.addVars(L,lb=-fmax, ub=fmax, name="Qt")

    d = dmax/htheta
    ###############
    # Constraints
    ###############
    for n1,n2,l in G.edges_iter(data='id'):
        m.addConstr( theta[n1] - theta[n2] <=  dmax)
        m.addConstr( theta[n1] - theta[n2] >= -dmax)
        for t in range(htheta+1):
            m.addConstr(phi[l] >= -0.5*(t*d)**2 + (t*d)*(theta[n1] - theta[n2]))
            m.addConstr(phi[l] >= -0.5*(t*d)**2 + (t*d)*(theta[n2] - theta[n1]))

    for n1,n2,l in G.edges_iter(data='id'):
        for _,_,l2 in G.edges_iter(data='id'):
            m.addConstr(Pf[l] - Y['gff'][l2]*(1+u[n1]) - Y['gft'][l2]*(1-phi[l]+u[n2]) + Y['bft'][l2]*(theta[n2] - theta[n1]) + bigM*(1 - Z[l,l2]) >= 0)
            m.addConstr(Pf[l] - Y['gff'][l2]*(1+u[n1]) - Y['gft'][l2]*(1-phi[l]+u[n2]) + Y['bft'][l2]*(theta[n2] - theta[n1]) - bigM*(1 - Z[l,l2]) <= 0)
            m.addConstr(Qf[l] + Y['bff'][l2]*(1+u[n1]) + Y['bft'][l2]*(1+phi[l]+u[n2]) - Y['gft'][l2]*(theta[n2] - theta[n1]) + bigM*(1 - Z[l,l2]) >= 0)
            m.addConstr(Qf[l] + Y['bff'][l2]*(1+u[n1]) + Y['bft'][l2]*(1+phi[l]+u[n2]) - Y['gft'][l2]*(theta[n2] - theta[n1]) - bigM*(1 - Z[l,l2]) <= 0)
            m.addConstr(Pt[l] - Y['gtt'][l2]*(1+u[n2]) - Y['gtf'][l2]*(1-phi[l]+u[n1]) + Y['btf'][l2]*(theta[n1] - theta[n2]) + bigM*(1 - Z[l,l2]) >= 0)
            m.addConstr(Pt[l] - Y['gtt'][l2]*(1+u[n2]) - Y['gtf'][l2]*(1-phi[l]+u[n1]) + Y['btf'][l2]*(theta[n1] - theta[n2]) - bigM*(1 - Z[l,l2]) <= 0)
            m.addConstr(Qt[l] + Y['btt'][l2]*(1+u[n2]) + Y['btf'][l2]*(1+phi[l]+u[n1]) - Y['gtf'][l2]*(theta[n1] - theta[n2]) + bigM*(1 - Z[l,l2]) >= 0)
            m.addConstr(Qt[l] + Y['btt'][l2]*(1+u[n2]) + Y['btf'][l2]*(1+phi[l]+u[n1]) - Y['gtf'][l2]*(theta[n1] - theta[n2]) - bigM*(1 - Z[l,l2]) <= 0)

    m.addConstrs( Pd[i] == sum(Pi[i,j]*S['Pd'][j] for j in range(N))/100 for i in range(N))
    m.addConstrs( Qd[i] == sum(Pi[i,j]*S['Qd'][j] for j in range(N))/100 for i in range(N))
    
    m.addConstrs( Pg[i] <=  sum(Pi[i,j]*S['Pgmax'][j] for j in range(N))/100 for i in range(N))
    m.addConstrs( Pg[i] >=  sum(Pi[i,j]*S['Pgmin'][j] for j in range(N))/100 for i in range(N))
    m.addConstrs( Qg[i] <=  sum(Pi[i,j]*S['Qgmax'][j] for j in range(N))/100 for i in range(N))
    m.addConstrs( Qg[i] >= -sum(Pi[i,j]*S['Qgmax'][j] for j in range(N))/100 for i in range(N))
    
    m.addConstrs( Qg[i] - Qgp[i] <= 0 for i in range(N))
    m.addConstrs( Qg[i] + Qgn[i] >= 0 for i in range(N))

    m.addConstrs( Pg[i] - Pd[i] - sum(Pt[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - sum(Pf[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0 for i in range(N)) 
    m.addConstrs( Qg[i] - Qd[i] - sum(Qt[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - sum(Qf[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0 for i in range(N)) 

    m.addConstrs( Pi.sum(i,'*') == 1 for i in range(N))
    m.addConstrs( Pi.sum('*',i) == 1 for i in range(N))
    m.addConstrs( Z.sum(i,'*')  == 1 for i in range(L))
    m.addConstrs( Z.sum('*',i)  == 1 for i in range(L))


    ###############
    # Objective
    ###############
    obj = Pg.sum('*') + phi.sum('*')#+ Qgp.sum('*') + Qgn.sum('*')

    ###############
    # Solve
    ##############
    m.setObjective(obj,gb.GRB.MINIMIZE)
    m.optimize(mycallback)
    

    ### get variables ####
    vars = {}
    vars['Pgmax'] = hlp.var2mat(S['Pgmax'], N, perm=Pi)
    vars['Pgmin'] = hlp.var2mat(S['Pgmin'], N, perm=Pi)
    vars['Qgmax'] = hlp.var2mat(S['Qgmax'], N, perm=Pi)
    vars['Pd']    = hlp.var2mat(Pd, N)
    vars['Qd']    = hlp.var2mat(Qd, N)
    vars['Pg']    = hlp.var2mat(Pg, N)
    vars['Qg']    = hlp.var2mat(Qg, N)
    vars['Pf']    = hlp.var2mat(Pf, L)
    vars['Qf']    = hlp.var2mat(Qf, L)
    vars['Pt']    = hlp.var2mat(Pt, L)
    vars['Qt']    = hlp.var2mat(Qt, L)
    vars['r']     = hlp.var2mat(z['r'], L, perm=Z)
    vars['x']     = hlp.var2mat(z['x'], L, perm=Z)
    vars['b']     = hlp.var2mat(z['b'], L, perm=Z)
    vars['theta'] = hlp.var2mat(theta, N)
    vars['u']     = hlp.var2mat(u, N)
    vars['phi']   = hlp.var2mat(phi,L)
    return vars 
