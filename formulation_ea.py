import sys
import itertools
import gurobipy as gb
import numpy as np
from scipy import sparse
import networkx as nx
import helpers as hlp
import logging

def mycallback2(model,where):
    if where == gb.GRB.Callback.MIPSOL:
        in_sum   = sum(model.cbGetSolution(model._beta[i]) for _,j in model._ebound_map['in'].items()  for i in j)
        out_sum  = sum(model.cbGetSolution(model._beta[i]) for _,j in model._ebound_map['out'].items() for i in j)
        Pg       = sum(model.cbGetSolution(model._Pg.values()))
        criteria = (Pg - model._pload + in_sum - out_sum)/(Pg + in_sum - out_sum)
        solcnt   = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        logging.info('Current solution: solcnt: %d, solmin: %d, sum(beta_in)=%0.2f, sum(beta_out)=%0.2f, sum(Pg)=%0.2f, sum(load)=%0.2f, criteria=%0.3g',solcnt,model._solmin,in_sum, out_sum, Pg, model._pload, criteria)
        if (solcnt > model._solmin) and (criteria < model._lossterm):
            logging.info('      terminating in MISOL due to minimal losses')
            model.terminate()
    if where == gb.GRB.Callback.MIPSOL:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):# or (elapsed_time > 1500): 
            logging.info('      terminating in MISOL due to time')
            model.terminate()
    elif where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIP_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):# or (elapsed_time > 1500):
            logging.info('      terminating in MIP due to time')
            model.terminate()
    elif where == gb.GRB.Callback.MIPNODE:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPNODE_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):# or (elapsed_time > 1500):
            logging.info('      terminating in MIPNODE due to time')
            model.terminate()
    else:
        pass

class ZoneMILP(object):
    def __init__(self,G,lossmin,lossterm,fmax,dmax,htheta,umin,umax,z,S,bigM,ebound,ebound_map):
        
        N = G.number_of_nodes()
        L = G.number_of_edges()
        ### shunt impedances numbers ###########
        if S['shunt']['include_shunts']:
            Ngsh = round(S['shunt']['Gfrac']*N)
            Nbsh = round(S['shunt']['Bfrac']*N)
        else:
            Ngsh = 0; Nbsh = 0
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
        Y = hlp.Yparts(z['r'],z['x'],b=z['b'],tau=z['tap'],phi=z['shift'])
        limitflag = np.all(z['rate'] == z['rate'][0])
        
        ### save inputs
        self.N = N; self.L = L
        self.Ngsh = Ngsh; self.Nbsh = Nbsh
        self.z = z; self.S = S
        self.nmap = nmap; self.rnmap = rnmap
        self.lmap = lmap; self.rlmap = rlmap
        self.ebound = ebound

        self.m = gb.Model()
        self.m.setParam('LogFile','/tmp/GurobiMultivar.log')
        self.m.setParam('LogToConsole',0)
        self.m.setParam('MIPGap',0.15)
        #m.setParam('SolutionLimit',1) #stop after this many solutions are found
        #self.m.setParam('TimeLimit', 1500)
        self.m.setParam('MIPFocus',1)
        self.m.setParam('ImproveStartTime',60)
        self.m.setParam('Threads',60)
   
        self.m._pload = sum(S['Pd'])/100
        #############
        # Variables
        #############
        self.Pi    = self.m.addVars(N,N,vtype=gb.GRB.BINARY,name="Pi")

        self.theta = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="theta")
        #self.u     = self.m.addVars(N,lb=umin, ub=umax,name="u")
        self.u     = self.m.addVars(N,name="u")
        self.phi   = self.m.addVars(L,lb=0,ub=dmax*dmax/2,name='phi')

        self.Pd    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Pd")
        self.Qd    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Qd")
        self.Pg    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Pg")
        self.Qg    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Qg")

        if Ngsh > 0:
            self.Psh = self.m.addVars(N,lb=S['shunt']['min'][0],ub=S['shunt']['max'][0])
            self.gsh = self.m.addVars(N,vtype=gb.GRB.BINARY)
        else:
            self.Psh = np.zeros(N)
        if Nbsh > 0:
            self.Qsh = self.m.addVars(N,lb=S['shunt']['min'][1],ub=S['shunt']['max'][1])
            self.Qshp= self.m.addVars(N,lb=0,ub=S['shunt']['max'][1])
            self.Qshn= self.m.addVars(N,lb=0,ub=S['shunt']['max'][1])
            self.bsh = self.m.addVars(N,vtype=gb.GRB.BINARY)
        else:
            self.Qsh = np.zeros(N)
        self.Pf    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Pf")
        self.Pt    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Pt")
        self.Qf    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Qf")
        self.Qt    = self.m.addVars(L,lb=-fmax, ub=fmax, name="Qt")

        # slacks
        self.s     = self.m.addVars(L,lb=0, ub=0.5*fmax) # flow limit slack
        self.sup   = self.m.addVars(N,lb=0, ub=0.5*umax) # voltage slack up
        self.sun   = self.m.addVars(N,lb=0, ub=0.5*umax) # voltage slack down

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
        # voltage limits
        self.m.addConstrs( self.u[i] >= umin - self.sun[i] for i in range(N))
        self.m.addConstrs( self.u[i] <= umax + self.sup[i] for i in range(N))

        # minimum loss constraint
        self.m.addConstr( self.Pg.sum("*") + sum(self.beta[i] for _,j in ebound_map['in'].items() for i in j) - sum(self.beta[i] for _,j in ebound_map['out'].items() for i in j) >= self.m._pload*(1/(1-lossmin)) ) 
        
        # edge constraints
        for _n1,_n2,_l in G.edges_iter(data='id'):
            n1 = nmap[_n1]; n2 = nmap[_n2];  l = lmap[_l]
            ### angle limits
            self.m.addConstr( self.theta[n1] - self.theta[n2] <=  dmax)
            self.m.addConstr( self.theta[n1] - self.theta[n2] >= -dmax)

            ##### flow limits #########
            if limitflag:
                self.m.addConstr(-sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) - self.s[l] <= self.Pf[l] )
                self.m.addConstr( sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) + self.s[l] >= self.Pf[l] )
                self.m.addConstr(-sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) - self.s[l] <= self.Pt[l] )
                self.m.addConstr( sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) + self.s[l] >= self.Pt[l] )
                self.m.addConstr(-sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) - self.s[l] <= self.Qf[l] )
                self.m.addConstr( sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) + self.s[l] >= self.Qf[l] )
                self.m.addConstr(-sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) - self.s[l] <= self.Qt[l] )
                self.m.addConstr( sum( self.Z[l,i]*z['rate'][i] for i in range(L) ) + self.s[l] >= self.Qt[l] )

            for t in range(htheta+1):
                self.m.addConstr(self.phi[l] >= -0.5*(t*d)**2 + (t*d)*(self.theta[n1] - self.theta[n2]))
                self.m.addConstr(self.phi[l] >= -0.5*(t*d)**2 + (t*d)*(self.theta[n2] - self.theta[n1]))

            #### branch flows ####
            self.m.addConstr( self.Pf[l] - Y['gff'][l]*(1+self.u[n1]) - Y['gft'][l]*(1-self.phi[l]+self.u[n2]) - Y['bft'][l]*(self.theta[n1] - self.theta[n2]) == 0)
            self.m.addConstr( self.Qf[l] + Y['bff'][l]*(1+self.u[n1]) + Y['bft'][l]*(1+self.phi[l]+self.u[n2]) - Y['gft'][l]*(self.theta[n1] - self.theta[n2]) == 0)
            self.m.addConstr( self.Pt[l] - Y['gtt'][l]*(1+self.u[n2]) - Y['gtf'][l]*(1-self.phi[l]+self.u[n1]) + Y['btf'][l]*(self.theta[n1] - self.theta[n2]) == 0)
            self.m.addConstr( self.Qt[l] + Y['btt'][l]*(1+self.u[n2]) + Y['btf'][l]*(1+self.phi[l]+self.u[n1]) + Y['gtf'][l]*(self.theta[n1] - self.theta[n2]) == 0)

        ### load 
        self.m.addConstrs( self.Pd[i] ==  sum( self.Pi[i,j]*S['Pd'][j]    for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Qd[i] ==  sum( self.Pi[i,j]*S['Qd'][j]    for j in range(N) )/100 for i in range(N))

        ### gen
        self.m.addConstrs( self.Pg[i] <=  sum( self.Pi[i,j]*S['Pgmax'][j] for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Pg[i] >=  sum( self.Pi[i,j]*S['Pgmin'][j] for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Qg[i] <=  sum( self.Pi[i,j]*S['Qgmax'][j] for j in range(N) )/100 for i in range(N))
        self.m.addConstrs( self.Qg[i] >= -sum( self.Pi[i,j]*S['Qgmax'][j] for j in range(N) )/100 for i in range(N))
       
        ### nodal balance
        self.m.addConstrs( self.Pg[i] - self.Psh[i] - self.Pd[i] - sum( self.Pt[lmap[l['id']]] for _,_,l in G.in_edges_iter([rnmap[i]],data='id') ) - \
                sum( self.Pf[lmap[l]] for _,_,l in G.out_edges_iter([rnmap[i]],data='id') ) + \
                sum( self.beta[l] for l in ebound_map['in'].get(rnmap[i],[]) ) - \
                sum( self.beta[l] for l in ebound_map['out'].get(rnmap[i],[]) ) == 0 for i in range(N))
        self.m.addConstrs( self.Qg[i] + self.Qsh[i] - self.Qd[i] - sum( self.Qt[lmap[l['id']]] for _,_,l in G.in_edges_iter([rnmap[i]],data='id') ) - \
                sum( self.Qf[lmap[l]] for _,_,l in G.out_edges_iter([rnmap[i]],data='id') ) + \
                sum( self.gamma[l] for l in ebound_map['in'].get(rnmap[i],[]) ) - \
                sum( self.gamma[l] for l in ebound_map['out'].get(rnmap[i],[]) ) == 0 for i in range(N)) 

        ###### shunts ##############
        if Ngsh > 0:
            self.m.addConstrs( self.Psh[i] >= self.gsh[i]*S['shunt']['min'][0] for i in range(N))
            self.m.addConstrs( self.Psh[i] <= self.gsh[i]*S['shunt']['max'][0] for i in range(N))
            self.m.addConstr(  self.gsh.sum('*') <= Ngsh )
        if Nbsh > 0:
            self.m.addConstrs( self.Qsh[i] >= self.bsh[i]*S['shunt']['min'][1] for i in range(N))
            self.m.addConstrs( self.Qsh[i] <= self.bsh[i]*S['shunt']['max'][1] for i in range(N))
            self.m.addConstr(  self.bsh.sum('*') <= Nbsh )
            self.m.addConstrs( self.Qsh[i] - self.Qshp[i] <= 0 for i in range(N))
            self.m.addConstrs( self.Qsh[i] + self.Qshn[i] >= 0 for i in range(N))

        self.m.addConstrs( self.Pi.sum(i,'*') == 1 for i in range(N))
        self.m.addConstrs( self.Pi.sum('*',i) == 1 for i in range(N))

        self.bp_abs = self.m.addConstrs(self.beta_p[i]  - self.beta[i]  >= 0 for i in ebound)
        self.bn_abs = self.m.addConstrs(self.beta_n[i]  + self.beta[i]  >= 0 for i in ebound)
        self.gp_abs = self.m.addConstrs(self.gamma_p[i] - self.gamma[i] >= 0 for i in ebound)
        self.gn_abs = self.m.addConstrs(self.gamma_n[i] + self.gamma[i] >= 0 for i in ebound)
        ###############
        # Objective
        ###############
        if Nbsh > 0:
            def obj():
                return self.Pg.sum('*') + self.phi.sum('*') + self.Qshp.sum("*") + self.Qshn.sum("*") \
                       + self.s.sum("*") + 100*(self.sup.sum("*") + self.sun.sum("*"))
        else:
            def obj():
                return self.Pg.sum('*') + self.phi.sum('*') \
                       + self.s.sum("*") + 100*(self.sup.sum("*") + self.sun.sum("*"))
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
        vars['s']     = hlp.var2mat(self.s,  self.L)
        vars['sup']   = hlp.var2mat(self.s,  self.N)
        vars['sun']   = hlp.var2mat(self.s,  self.N)
        vars['theta'] = hlp.var2mat(self.theta, self.N)
        vars['u']     = hlp.var2mat(self.u, self.N)
        vars['phi']   = hlp.var2mat(self.phi,self.L)
        if self.Ngsh > 0:
            vars['GS']= hlp.var2mat(self.Psh,self.N)
        if self.Nbsh > 0:
            vars['BS']= hlp.var2mat(self.Qsh,self.N)
        return vars 
