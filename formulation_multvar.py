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
        Pg       = sum(model.cbGetSolution(model._Pg.values()))
        criteria = (Pg - model._pload)/model._pload
        solcnt   = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        logging.info('Current solution: solcnt: %d, sum(Pg)=%0.2f, sum(load)=%0.2f, criteria=%0.3g', solcnt, Pg, model._pload, criteria)
        if (solcnt > 0) and (criteria < model._lossterm):
            logging.info('      terminating in MISOL due to minimal losses')
            model.terminate()
    if where == gb.GRB.Callback.MIPSOL:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        obj          = model.cbGet(gb.GRB.Callback.MIPSOL_OBJBST)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500):
            logging.info('      terminating in MISOL')
            model.terminate()
    elif where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIP_SOLCNT)
        obj          = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500):
            logging.info('      terminating in MIP')
            model.terminate()
    elif where == gb.GRB.Callback.MIPNODE:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPNODE_SOLCNT)
        obj          = model.cbGet(gb.GRB.Callback.MIPNODE_OBJBST)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500):
            logging.info('      terminating in MIPNODE')
            model.terminate()
    else:
        pass

def mycallback2(model,where):
    if where == gb.GRB.Callback.MIPSOL:
        in_sum   = sum(model.cbGetSolution(model._beta[i]) for _,j in model._ebound_map['in'].items()  for i in j)
        out_sum  = sum(model.cbGetSolution(model._beta[i]) for _,j in model._ebound_map['out'].items() for i in j)
        Pg       = sum(model.cbGetSolution(model._Pg.values()))
        criteria = (Pg - model._pload + in_sum - out_sum)/model._pload
        solcnt   = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        logging.info('Current solution: solcnt: %d, solmin: %d, sum(beta_in)=%0.2f, sum(beta_out)=%0.2f, sum(Pg)=%0.2f, sum(load)=%0.2f, criteria=%0.3g',solcnt,model._solmin,in_sum, out_sum, Pg, model._pload, criteria)
        if (solcnt > model._solmin) and (criteria < model._lossterm):
            logging.info('      terminating in MISOL due to minimal losses')
            model.terminate()
    if where == gb.GRB.Callback.MIPSOL:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500): 
            logging.info('      terminating in MISOL due to time')
            model.terminate()
    elif where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIP_SOLCNT)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500):
            logging.info('      terminating in MIP due to time')
            model.terminate()
    elif where == gb.GRB.Callback.MIPNODE:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPNODE_SOLCNT)
        if ((solcnt > 1) and elapsed_time > 500) or (elapsed_time > 1500):
            logging.info('      terminating in MIPNODE due to time')
            model.terminate()
    else:
        pass

def single_system(G,lossmin,lossterm,fmax,dmax,htheta,umin,umax,z,S,bigM):

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
   
    m._pload = sum(S['Pd'])/100
    m._lossterm = lossterm
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

    m._Pg = Pg
    d = dmax/htheta
    ###############
    # Constraints
    ###############
    m.addConstr( Pg.sum("*") >= m._pload*(1+lossmin) )
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

class ZoneMILP(object):
    def __init__(self,G,lossmin,lossterm,fmax,dmax,htheta,umin,umax,z,S,bigM,ebound,ebound_map):
        
        N = G.number_of_nodes()
        L = G.number_of_edges()
        nmap = dict(zip(G.nodes(),range(N)))
        rnmap= np.empty(N,dtype='int')
        for k,v in nmap.items():
            rnmap[v] = k
        lmap = {}
        for i,(_,_,l) in enumerate(G.edges_iter(data='id')):
            lmap[l] = i
        rlmap = np.empty(L,dtype='int')
        for k,v, in lmap.items():
            rlmap[v] = k
        ### get primitive admittance values ####
        Y = hlp.Yparts(z['r'],z['x'],b=z['b'])
        
        ### save inputs
        self.N = N; self.L = L
        self.z = z; self.S = S
        self.nmap = nmap; self.rnmap = rnmap
        self.lmap = lmap; self.rlmap = rlmap
        self.ebound = ebound

        self.m = gb.Model()
        self.m.setParam('LogFile','/tmp/GurobiMultivar.log')
        self.m.setParam('LogToConsole',0)
        self.m.setParam('MIPGap',0.15)
        #m.setParam('SolutionLimit',1) #stop after this many solutions are found
        self.m.setParam('TimeLimit', 1500)
        self.m.setParam('MIPFocus',1)
        self.m.setParam('ImproveStartTime',60)
        self.m.setParam('Threads',60)
   
        self.m._pload = sum(S['Pd'])/100
        #############
        # Variables
        #############
        self.Pi    = self.m.addVars(N,N,vtype=gb.GRB.BINARY,name="Pi")
        self.Z     = self.m.addVars(L,L,vtype=gb.GRB.BINARY,name="Z")

        self.theta = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="theta")
        self.u     = self.m.addVars(N,lb=umin, ub=umax,name="u")
        self.phi   = self.m.addVars(L,lb=0,ub=dmax*dmax/2,name='phi')

        self.Pd    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Pd")
        self.Qd    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Qd")
        self.Pg    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Pg")
        self.Qg    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Qg")

        self.Pf    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Pf")
        self.Pt    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Pt")
        self.Qf    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Qf")
        self.Qt    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Qt")

        #NOTE: beta and gamma are on EXTERNAL/GLOBAL indexing!!!!
        self.beta   = self.m.addVars(ebound, lb=-fmax, ub=fmax, name='beta')
        self.gamma  = self.m.addVars(ebound, lb=-fmax, ub=fmax, name='gamma')

        self.beta_p = self.m.addVars(ebound, lb=0, ub=fmax, name='beta_n')
        self.beta_n = self.m.addVars(ebound, lb=0, ub=fmax, name='beta_m')
        self.gamma_p= self.m.addVars(ebound, lb=0, ub=fmax, name='gamma_n')
        self.gamma_n= self.m.addVars(ebound, lb=0, ub=fmax, name='gamma_m')

        self.m._Pg   = self.Pg
        self.m._beta = self.beta
        self.m._solmin = 0
        self.m._ebound_map = ebound_map
        self.m._lossterm = lossterm
        d = dmax/htheta
        self.w  = {l: 0 for l in ebound}
        self.nu = {l: 0 for l in ebound}

        ###############
        # Constraints
        ###############
        self.m.addConstr( self.Pg.sum("*") + sum(self.beta[i] for _,j in ebound_map['in'].items() for i in j) - sum(self.beta[i] for _,j in ebound_map['out'].items() for i in j) >= self.m._pload*(1+lossmin) )
        for _n1,_n2,_l in G.edges_iter(data='id'):
            n1 = nmap[_n1]; n2 = nmap[_n2];  l = lmap[_l]
            self.m.addConstr( self.theta[n1] - self.theta[n2] <=  dmax)
            self.m.addConstr( self.theta[n1] - self.theta[n2] >= -dmax)
            for t in range(htheta+1):
                self.m.addConstr(self.phi[l] >= -0.5*(t*d)**2 + (t*d)*(self.theta[n1] - self.theta[n2]))
                self.m.addConstr(self.phi[l] >= -0.5*(t*d)**2 + (t*d)*(self.theta[n2] - self.theta[n1]))

            for _,_,_l2 in G.edges_iter(data='id'):
                l2 = lmap[_l2]
                self.m.addConstr( self.Pf[l] - Y['gff'][l2]*(1+self.u[n1]) - Y['gft'][l2]*(1-self.phi[l]+self.u[n2]) + Y['bft'][l2]*(self.theta[n2] - self.theta[n1]) + bigM*(1 - self.Z[l,l2]) >= 0)
                self.m.addConstr( self.Pf[l] - Y['gff'][l2]*(1+self.u[n1]) - Y['gft'][l2]*(1-self.phi[l]+self.u[n2]) + Y['bft'][l2]*(self.theta[n2] - self.theta[n1]) - bigM*(1 - self.Z[l,l2]) <= 0)
                self.m.addConstr( self.Qf[l] + Y['bff'][l2]*(1+self.u[n1]) + Y['bft'][l2]*(1+self.phi[l]+self.u[n2]) - Y['gft'][l2]*(self.theta[n2] - self.theta[n1]) + bigM*(1 - self.Z[l,l2]) >= 0)
                self.m.addConstr( self.Qf[l] + Y['bff'][l2]*(1+self.u[n1]) + Y['bft'][l2]*(1+self.phi[l]+self.u[n2]) - Y['gft'][l2]*(self.theta[n2] - self.theta[n1]) - bigM*(1 - self.Z[l,l2]) <= 0)
                self.m.addConstr( self.Pt[l] - Y['gtt'][l2]*(1+self.u[n2]) - Y['gtf'][l2]*(1-self.phi[l]+self.u[n1]) + Y['btf'][l2]*(self.theta[n1] - self.theta[n2]) + bigM*(1 - self.Z[l,l2]) >= 0)
                self.m.addConstr( self.Pt[l] - Y['gtt'][l2]*(1+self.u[n2]) - Y['gtf'][l2]*(1-self.phi[l]+self.u[n1]) + Y['btf'][l2]*(self.theta[n1] - self.theta[n2]) - bigM*(1 - self.Z[l,l2]) <= 0)
                self.m.addConstr( self.Qt[l] + Y['btt'][l2]*(1+self.u[n2]) + Y['btf'][l2]*(1+self.phi[l]+self.u[n1]) - Y['gtf'][l2]*(self.theta[n1] - self.theta[n2]) + bigM*(1 - self.Z[l,l2]) >= 0)
                self.m.addConstr( self.Qt[l] + Y['btt'][l2]*(1+self.u[n2]) + Y['btf'][l2]*(1+self.phi[l]+self.u[n1]) - Y['gtf'][l2]*(self.theta[n1] - self.theta[n2]) - bigM*(1 - self.Z[l,l2]) <= 0)

        self.m.addConstrs( self.Pd[i] ==  sum( self.Pi[i,j]*S['Pd'][j]    for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Qd[i] ==  sum( self.Pi[i,j]*S['Qd'][j]    for j in range(N) )/100 for i in range(N))

        self.m.addConstrs( self.Pg[i] <=  sum( self.Pi[i,j]*S['Pgmax'][j] for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Pg[i] >=  sum( self.Pi[i,j]*S['Pgmin'][j] for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Qg[i] <=  sum( self.Pi[i,j]*S['Qgmax'][j] for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Qg[i] >= -sum( self.Pi[i,j]*S['Qgmax'][j] for j in range(N) )/100 for i in range(N))
       
        self.m.addConstrs( self.Pg[i] - self.Pd[i] - sum( self.Pt[lmap[l['id']]] for _,_,l in G.in_edges_iter([rnmap[i]],data='id') ) - \
                sum( self.Pf[lmap[l]] for _,_,l in G.out_edges_iter([rnmap[i]],data='id') ) + \
                sum( self.beta[l] for l in ebound_map['in'].get(rnmap[i],[]) ) - \
                sum( self.beta[l] for l in ebound_map['out'].get(rnmap[i],[]) ) == 0 for i in range(N))
        self.m.addConstrs( self.Qg[i] - self.Qd[i] - sum( self.Qt[lmap[l['id']]] for _,_,l in G.in_edges_iter([rnmap[i]],data='id') ) - \
                sum( self.Qf[lmap[l]] for _,_,l in G.out_edges_iter([rnmap[i]],data='id') ) + \
                sum( self.gamma[l] for l in ebound_map['in'].get(rnmap[i],[]) ) - \
                sum( self.gamma[l] for l in ebound_map['out'].get(rnmap[i],[]) ) == 0 for i in range(N)) 

        self.m.addConstrs( self.Pi.sum(i,'*') == 1 for i in range(N))
        self.m.addConstrs( self.Pi.sum('*',i) == 1 for i in range(N))
        self.m.addConstrs( self.Z.sum(i,'*')  == 1 for i in range(L))
        self.m.addConstrs( self.Z.sum('*',i)  == 1 for i in range(L))

        self.bp_abs = self.m.addConstrs(self.beta_p[i]  - self.beta[i]  >= 0 for i in ebound)
        self.bn_abs = self.m.addConstrs(self.beta_n[i]  + self.beta[i]  >= 0 for i in ebound)
        self.gp_abs = self.m.addConstrs(self.gamma_p[i] - self.gamma[i] >= 0 for i in ebound)
        self.gn_abs = self.m.addConstrs(self.gamma_n[i] + self.gamma[i] >= 0 for i in ebound)
        ###############
        # Objective
        ###############
        def obj():
            return self.Pg.sum('*') + self.phi.sum('*')
        self.obj = obj
        self.m.setObjective(self.obj() + self.beta_p.sum('*') + self.beta_n.sum("*") + self.gamma_p.sum("*") + self.gamma_n.sum("*"), gb.GRB.MINIMIZE)

    ######## METHODS ###########
    def objective_update(self,beta_bar, gamma_bar, rho):
        obj = self.obj()
    
        for i in self.ebound:
            # update dual variables w and nu
            self.w[i]  += rho*(self.beta[i].X  - beta_bar[i])
            self.nu[i] += rho*(self.gamma[i].X - gamma_bar[i])
    
            # update objective
            obj += self.w[i]*self.beta[i] #Lagrangian term
            obj += (rho/2)*(self.beta[i] - beta_bar[i])*(self.beta[i] - beta_bar[i]) # augmented Lagrangian term
            obj += self.nu[i]*self.gamma[i] #Lagrangian term
            obj += (rho/2)*(self.gamma[i] - gamma_bar[i])*(self.gamma[i] - gamma_bar[i]) # augmented Lagrangian term
        self.m.setObjective(obj, gb.GRB.MINIMIZE)
    
    def remove_abs_vars(self):
        """ remove the beta_abs and gamma_abs variables and constraints"""
        self.m.remove(self.bp_abs)
        self.m.remove(self.bn_abs)
        self.m.remove(self.gp_abs)
        self.m.remove(self.gn_abs)
        self.m.remove(self.beta_p)
        self.m.remove(self.beta_n)
        self.m.remove(self.gamma_p)
        self.m.remove(self.gamma_n)

    def optimize(self):
        self.m.optimize(mycallback2)

    def getvars(self):
        vars = {}
        vars['Pgmax'] = hlp.var2mat(self.S['Pgmax'], self.N, perm=self.Pi)
        vars['Pgmin'] = hlp.var2mat(self.S['Pgmin'], self.N, perm=self.Pi)
        vars['Qgmax'] = hlp.var2mat(self.S['Qgmax'], self.N, perm=self.Pi)
        vars['Pd']    = hlp.var2mat(self.Pd, self.N)
        vars['Qd']    = hlp.var2mat(self.Qd, self.N)
        vars['Pg']    = hlp.var2mat(self.Pg, self.N)
        vars['Qg']    = hlp.var2mat(self.Qg, self.N)
        vars['Pf']    = hlp.var2mat(self.Pf, self.L)
        vars['Qf']    = hlp.var2mat(self.Qf, self.L)
        vars['Pt']    = hlp.var2mat(self.Pt, self.L)
        vars['Qt']    = hlp.var2mat(self.Qt, self.L)
        vars['r']     = hlp.var2mat(self.z['r'], self.L, perm=self.Z)
        vars['x']     = hlp.var2mat(self.z['x'], self.L, perm=self.Z)
        vars['b']     = hlp.var2mat(self.z['b'], self.L, perm=self.Z)
        vars['theta'] = hlp.var2mat(self.theta, self.N)
        vars['u']     = hlp.var2mat(self.u, self.N)
        vars['phi']   = hlp.var2mat(self.phi,self.L)
        return vars 
