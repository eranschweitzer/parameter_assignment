import sys
import itertools
import gurobipy as gb
import numpy as np
from scipy import sparse
import networkx as nx
import helpers as hlp
#import logging
import logfun as lg
import multvar_solution_check as chk

def mycallback2(model,where):
    if where == gb.GRB.Callback.MIPSOL:
        in_sum   = sum(model.cbGetSolution(model._beta[i]) for _,j in model._ebound_map['in'].items()  for i in j)
        out_sum  = sum(model.cbGetSolution(model._beta[i]) for _,j in model._ebound_map['out'].items() for i in j)
        Pg       = sum(model.cbGetSolution(model._Pg.values()))
        criteria = (Pg - model._pload + in_sum - out_sum)/(Pg + in_sum - out_sum)
        solcnt   = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        phiconst = 0
        for _n1,_n2,_l in model._G.edges_iter(data='id'):
            n1 = model._nmap[_n1]; n2 = model._nmap[_n2];  l = model._lmap[_l];
            if 0.5*(model.cbGetSolution(model._theta[n1]) - model.cbGetSolution(model._theta[n2]))**2 - model.cbGetSolution(model._phi[l]) < -1e-5:
                #model._tmpconst.append(model.addConstr(model._phi[l] <= 0.5*(model.cbGetSolution(model._theta[n1]) - model.cbGetSolution(model._theta[n2]))**2))
                phiconst += 1
        #logging.info('Current solution: solcnt: %d, solmin: %d, sum(beta_in)=%0.2f, sum(beta_out)=%0.2f, sum(Pg)=%0.2f, sum(load)=%0.2f, criteria=%0.3g, phiconst=%d',solcnt,model._solmin,in_sum, out_sum, Pg, model._pload, criteria, phiconst)
        lg.log_callback(model, solcnt, in_sum, out_sum, Pg, criteria, phiconst, logger=model._logger)
        if (solcnt > model._solmin) and (criteria < model._lossterm):
            #logging.info('      terminating in MISOL due to minimal losses')
            lg.log_calback_terminate('MISOL', 'minimal losses', logger=model._logger)
            model.terminate()
    if where == gb.GRB.Callback.MIPSOL:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):# or (elapsed_time > 1500): 
            #logging.info('      terminating in MISOL due to time')
            lg.log_calback_terminate('MISOL', 'time', logger=model._logger)
            model.terminate()
    elif where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIP_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):# or (elapsed_time > 1500):
            #logging.info('      terminating in MIP due to time')
            lg.log_calback_terminate('MIP', 'time', logger=model._logger)
            model.terminate()
    elif where == gb.GRB.Callback.MIPNODE:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPNODE_SOLCNT) + 1
        if ((solcnt > 1) and elapsed_time > 500):# or (elapsed_time > 1500):
            #logging.info('      terminating in MIPNODE due to time')
            lg.log_calback_terminate('MIPNODE', 'time', logger=model._logger)
            model.terminate()
    else:
        pass

class ZoneMILP(object):
    def __init__(self,G,consts,params,zperm,ebound=None,ebound_map=None,nperm=False, zone=0):
        
        if ebound is None:
            ebound = []
        if ebound_map is None:
            ebound_map = {'in':{}, 'out':{}}

        N = G.number_of_nodes()
        L = G.number_of_edges()
        self.wcnt = {'mps':0, 'mst':0}

        ### pick central node and limits for theta
        central_node = hlp.pick_ang0_node(G) # IMPORTANT: this will be in GLOBAL index
        theta_max    = hlp.theta_max(G,central_node)

        ### shunt impedances numbers ###########
        if params['S']['shunt']['include_shunts']:
            Ngsh = round(params['S']['shunt']['Gfrac']*N)
            Nbsh = round(params['S']['shunt']['Bfrac']*N)
        else:
            Ngsh = 0; Nbsh = 0
        ### mapping
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
        Y = hlp.Yparts(params['z']['r'], params['z']['x'], b=params['z']['b'], tau=params['z']['tap'], phi=params['z']['shift'])
        
        ### save inputs
        self.N = N; self.L = L
        self.Ngsh = Ngsh; self.Nbsh = Nbsh
        self.nmap = nmap; self.rnmap = rnmap
        self.lmap = lmap; self.rlmap = rlmap
        self.S    = params['S']
        self.z    = params['z']
        self.ebound = ebound
        self.nperm = nperm
        self.zperm = zperm
        self.G = G
        self.zone=zone
        self.consts = consts

        self.m = gb.Model()
        #self.m.setParam('LogFile','/tmp/GurobiMultivar.log')
        self.m.setParam('LogFile', consts['gurobi_config']['LogFile'])
        self.m.setParam('LogToConsole',0)
        #m.setParam('SolutionLimit',1) #stop after this many solutions are found
        #self.m.setParam('TimeLimit', 1500)
        #self.m.setParam('MIPFocus',1)
        #self.m.setParam('Threads',60)
        #self.m.setParam('MIPGap',0.15)
        self.m.setParam('IntFeasTol', 1e-6)
        self.m.setParam('ImproveStartTime',60)
        
        for key, value in consts['gurobi_config'].items():
            if key != 'LogFile':
                self.m.setParam(key, value)
   
        self.m._pload = sum(params['S']['Pd'])/100
        #############
        # Variables
        #############
        if not nperm:
            self.Pi    = self.m.addVars(N,N,vtype=gb.GRB.BINARY,name="Pi")

        self.theta = self.m.addVars(N,lb=-theta_max, ub=theta_max, name="theta")
        #self.u     = self.m.addVars(N,lb=umin, ub=umax,name="u")
        self.u     = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="u")
        self.phi   = self.m.addVars(L,lb=0,ub=consts['dmax']*consts['dmax']/2,name='phi')

        self.Pd    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Pd")
        self.Qd    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Qd")
        self.Pg    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Pg")
        self.Qg    = self.m.addVars(N,lb=-gb.GRB.INFINITY, name="Qg")

        if Ngsh > 0:
            self.Psh = self.m.addVars(N,lb=params['S']['shunt']['min'][0],ub=params['S']['shunt']['max'][0])
            self.gsh = self.m.addVars(N,vtype=gb.GRB.BINARY)
        else:
            self.Psh = np.zeros(N)
        if Nbsh > 0:
            self.Qsh = self.m.addVars(N,lb=params['S']['shunt']['min'][1],ub=params['S']['shunt']['max'][1])
            self.Qshp= self.m.addVars(N,lb=0,ub=params['S']['shunt']['max'][1])
            self.Qshn= self.m.addVars(N,lb=0,ub=params['S']['shunt']['max'][1])
            self.bsh = self.m.addVars(N,vtype=gb.GRB.BINARY)
        else:
            self.Qsh = np.zeros(N)
        #self.Pf    = self.m.addVars(L,lb=-consts['fmax'], ub=consts['fmax'], name="Pf")
        #self.Pt    = self.m.addVars(L,lb=-consts['fmax'], ub=consts['fmax'], name="Pt")
        #self.Qf    = self.m.addVars(L,lb=-consts['fmax'], ub=consts['fmax'], name="Qf")
        #self.Qt    = self.m.addVars(L,lb=-consts['fmax'], ub=consts['fmax'], name="Qt")
        self.Pf    = self.m.addVars(L,lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="Pf")
        self.Pt    = self.m.addVars(L,lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="Pt")
        self.Qf    = self.m.addVars(L,lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="Qf")
        self.Qt    = self.m.addVars(L,lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name="Qt")

        # slacks
        self.sf    = self.m.addVars(L,lb=0, ub=0.5*consts['fmax'], name="sf") # flow limit slack
        self.su    = self.m.addVars(N,lb=0, ub=0.5*consts['umax'], name="su") # voltage slack up
        self.sd    = self.m.addVars(L,lb=0, ub=np.pi-consts['dmax'], name="sd") #angle difference slack up

        #NOTE: beta and gamma are on EXTERNAL/GLOBAL indexing!!!!
        self.beta   = self.m.addVars(ebound, lb=-consts['fmax'], ub=consts['fmax'], name='beta')
        self.gamma  = self.m.addVars(ebound, lb=-consts['fmax'], ub=consts['fmax'], name='gamma')

        self.beta_p = self.m.addVars(ebound, lb=0, ub=consts['fmax'], name='beta_n')
        self.beta_n = self.m.addVars(ebound, lb=0, ub=consts['fmax'], name='beta_m')
        self.gamma_p= self.m.addVars(ebound, lb=0, ub=consts['fmax'], name='gamma_n')
        self.gamma_n= self.m.addVars(ebound, lb=0, ub=consts['fmax'], name='gamma_m')

        self.m._Pg   = self.Pg
        self.m._beta = self.beta
        self.m._solmin = 0
        self.m._ebound_map = ebound_map
        self.m._lossterm = consts['lossterm']
        self.m._G = G
        self.m._phi = self.phi
        self.m._theta = self.theta
        self.m._nmap  = nmap
        self.m._lmap  = lmap
        self.m._tmpconst = []
        self.m._zone = zone
        dphi = 2*consts['dmax']/consts['htheta']
        self.w  = {l: 0 for l in ebound}
        self.nu = {l: 0 for l in ebound}

        ###############
        # Constraints
        ###############
        # fix central node to have theta of 0
        self.m.addConstr( self.theta[nmap[central_node]] == 0 )
        # voltage limits
        self.m.addConstrs( self.u[i]  >= consts['umin'] - self.su[i] for i in range(N))
        self.m.addConstrs( self.u[i]  <= consts['umax'] + self.su[i] for i in range(N))

        # beta limits
        for l in ebound:
            zl = zperm[l]
            self.m.addConstr( self.beta[l]     >= -params['z']['rate'][zl] )
            self.m.addConstr( self.beta[l]     <= +params['z']['rate'][zl] )
            self.m.addConstr( self.gamma[l]    >= -params['z']['rate'][zl] )
            self.m.addConstr( self.gamma[l]    <= +params['z']['rate'][zl] )
            self.bp_lim = self.m.addConstr( self.beta_p[l]   <= +params['z']['rate'][zl] )
            self.bn_lim = self.m.addConstr( self.beta_n[l]   <= +params['z']['rate'][zl] )
            self.gp_lim = self.m.addConstr( self.gamma_p[l]  <= +params['z']['rate'][zl] )
            self.gn_lim = self.m.addConstr( self.gamma_n[l]  <= +params['z']['rate'][zl] )

        # minimum loss constraint
        self.m.addConstr( self.Pg.sum("*") + sum(self.beta[i] for _,j in ebound_map['in'].items() for i in j) - sum(self.beta[i] for _,j in ebound_map['out'].items() for i in j) >= self.m._pload*(1/(1-consts['lossmin'])) ) 
        # var generation plus import PLUS export should be positive. Idea is to not let generators only absorb vars
        self.m.addConstr( self.Qg.sum("*") + sum(self.gamma[i] for _,j in ebound_map['in'].items() for i in j) + sum(self.gamma[i] for _,j in ebound_map['out'].items() for i in j) >= 0 )
        
        # edge constraints
        for _n1,_n2,_l in G.edges_iter(data='id'):
            n1 = nmap[_n1]; n2 = nmap[_n2];  l = lmap[_l]; zl = zperm[_l];
            ### angle limits
            self.m.addConstr( self.theta[n1] - self.theta[n2] <=  consts['dmax'] + self.sd[l])
            self.m.addConstr( self.theta[n1] - self.theta[n2] >= -consts['dmax'] - self.sd[l])

            ##### flow limits #########
            self.m.addConstr( self.Pf[l]  >= -params['z']['rate'][zl] - self.sf[l])
            self.m.addConstr( self.Pf[l]  <= +params['z']['rate'][zl] + self.sf[l])
            self.m.addConstr( self.Pt[l]  >= -params['z']['rate'][zl] - self.sf[l])
            self.m.addConstr( self.Pt[l]  <= +params['z']['rate'][zl] + self.sf[l])
            self.m.addConstr( self.Qf[l]  >= -params['z']['rate'][zl] - self.sf[l])
            self.m.addConstr( self.Qf[l]  <= +params['z']['rate'][zl] + self.sf[l])
            self.m.addConstr( self.Qt[l]  >= -params['z']['rate'][zl] - self.sf[l])
            self.m.addConstr( self.Qt[l]  <= +params['z']['rate'][zl] + self.sf[l])
            
            for t in range(int(consts['htheta']) + 1):
                self.m.addConstr(self.phi[l] >= -0.5*(-consts['dmax'] + t*dphi)**2 + (-consts['dmax'] + t*dphi)*(self.theta[n1] - self.theta[n2]))
                #self.m.addConstr(self.phi[l] >= -0.5*(t*d)**2 + (t*d)*(self.theta[n1] - self.theta[n2]))
                #self.m.addConstr(self.phi[l] >= -0.5*(t*d)**2 + (t*d)*(self.theta[n2] - self.theta[n1]))

            #### branch flows ####
            self.m.addConstr( self.Pf[l] - Y['gff'][zl]*(1+self.u[n1]) - Y['gft'][zl]*(1-self.phi[l]+self.u[n2]) - Y['bft'][zl]*(self.theta[n1] - self.theta[n2]) == 0)
            self.m.addConstr( self.Qf[l] + Y['bff'][zl]*(1+self.u[n1]) + Y['bft'][zl]*(1-self.phi[l]+self.u[n2]) - Y['gft'][zl]*(self.theta[n1] - self.theta[n2]) == 0)
            self.m.addConstr( self.Pt[l] - Y['gtt'][zl]*(1+self.u[n2]) - Y['gtf'][zl]*(1-self.phi[l]+self.u[n1]) + Y['btf'][zl]*(self.theta[n1] - self.theta[n2]) == 0)
            self.m.addConstr( self.Qt[l] + Y['btt'][zl]*(1+self.u[n2]) + Y['btf'][zl]*(1-self.phi[l]+self.u[n1]) + Y['gtf'][zl]*(self.theta[n1] - self.theta[n2]) == 0)

        if not nperm:
            ### load 
            self.m.addConstrs( self.Pd[i] ==  sum( self.Pi[i,j]*params['S']['Pd'][j]    for j in range(N) )/100 for i in range(N))
            self.m.addConstrs( self.Qd[i] ==  sum( self.Pi[i,j]*params['S']['Qd'][j]    for j in range(N) )/100 for i in range(N))
            ### gen
            self.m.addConstrs( self.Pg[i] <=  sum( self.Pi[i,j]*params['S']['Pgmax'][j] for j in range(N) )/100 for i in range(N))
            self.m.addConstrs( self.Pg[i] >=  sum( self.Pi[i,j]*params['S']['Pgmin'][j] for j in range(N) )/100 for i in range(N))
            self.m.addConstrs( self.Qg[i] <=  sum( self.Pi[i,j]*params['S']['Qgmax'][j] for j in range(N) )/100 for i in range(N))
            self.m.addConstrs( self.Qg[i] >= -sum( self.Pi[i,j]*params['S']['Qgmax'][j] for j in range(N) )/100 for i in range(N))
        else:
            ### load
            self.m.addConstrs( self.Pd[i] == params['S']['Pd'][rnmap[i]]/100 for i in range(N))
            self.m.addConstrs( self.Qd[i] == params['S']['Qd'][rnmap[i]]/100 for i in range(N))
            ### gen
            self.m.addConstrs( self.Pg[i] <=  params['S']['Pgmax'][rnmap[i]] /100 for i in range(N))
            self.m.addConstrs( self.Pg[i] >=  params['S']['Pgmin'][rnmap[i]] /100 for i in range(N))
            self.m.addConstrs( self.Qg[i] <=  params['S']['Qgmax'][rnmap[i]] /100 for i in range(N))
            self.m.addConstrs( self.Qg[i] >= -params['S']['Qgmax'][rnmap[i]] /100 for i in range(N))

       
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
            self.m.addConstrs( self.Psh[i] >= self.gsh[i]*consts['S']['shunt']['min'][0] for i in range(N))
            self.m.addConstrs( self.Psh[i] <= self.gsh[i]*consts['S']['shunt']['max'][0] for i in range(N))
            self.m.addConstr(  self.gsh.sum('*') <= Ngsh )
        if Nbsh > 0:
            self.m.addConstrs( self.Qsh[i] >= self.bsh[i]*consts['S']['shunt']['min'][1] for i in range(N))
            self.m.addConstrs( self.Qsh[i] <= self.bsh[i]*consts['S']['shunt']['max'][1] for i in range(N))
            self.m.addConstr(  self.bsh.sum('*') <= Nbsh )
            self.m.addConstrs( self.Qsh[i] - self.Qshp[i] <= 0 for i in range(N))
            self.m.addConstrs( self.Qsh[i] + self.Qshn[i] >= 0 for i in range(N))

        if not nperm:
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
            def obj(scale=1):
                w = {'sf': 10, 'su':100, 'sd': 100, 'beta':2}
                for k,v in w.items():
                    w[k] = max(v*scale,v)
                w['phi'] = 1.0
                return self.Pg.sum('*') + w['phi']*self.phi.sum('*') + self.Qshp.sum("*") + self.Qshn.sum("*") \
                        + w['sf']*self.sf.sum("*") + w['su']*self.su.sum("*") + w['sd']*self.sd.sum("*") \
                        + w['beta']*(self.beta_p.sum('*') + self.beta_n.sum("*") + self.gamma_p.sum("*") + self.gamma_n.sum("*")) 
        else:
            def obj(scale=1):
                w = {'sf': 10, 'su':100, 'sd': 100, 'beta':2}
                for k,v in w.items():
                    w[k] = max(v*scale,v)
                w['phi'] = 1.0
                return self.Pg.sum('*') + w['phi']*self.phi.sum('*') \
                        + w['sf']*self.sf.sum("*") + w['su']*self.su.sum("*") + w['sd']*self.sd.sum("*") \
                        + w['beta']*(self.beta_p.sum('*') + self.beta_n.sum("*") + self.gamma_p.sum("*") + self.gamma_n.sum("*")) 
        self.obj = obj
        #self.m.setObjective(self.obj() + 2*(self.beta_p.sum('*') + self.beta_n.sum("*") + self.gamma_p.sum("*") + self.gamma_n.sum("*")), gb.GRB.MINIMIZE)
        self.m.setObjective(self.obj(), gb.GRB.MINIMIZE)

    ######## METHODS ###########
    def objective_update(self,beta_bar, gamma_bar, rho):
        
        if self.consts['aug_relax']:
            self.beta_bar  = beta_bar
            self.gamma_bar = gamma_bar
            try:
                self.const_update(beta_bar, gamma_bar)
            except AttributeError:
                self.auglag_relax(beta_bar, gamma_bar)
        
        obj = self.obj()
    
        for i in self.ebound:
            # update dual variables w and nu
            self.w[i]  += rho*(self.beta[i].X  - beta_bar[i])
            self.nu[i] += rho*(self.gamma[i].X - gamma_bar[i])
    
            # update objective
            obj += self.w[i]*self.beta[i] #Lagrangian term
            obj += self.nu[i]*self.gamma[i] #Lagrangian term
            if not self.consts['aug_relax']:
                obj += (rho/2)*(self.beta[i] - beta_bar[i])*(self.beta[i] - beta_bar[i]) # augmented Lagrangian term
                obj += (rho/2)*(self.gamma[i] - gamma_bar[i])*(self.gamma[i] - gamma_bar[i]) # augmented Lagrangian term
            else:
                obj += (rho/2)*self.beta2[i]
                obj += (rho/2)*self.gamma2[i]
        self.m.setObjective(obj, gb.GRB.MINIMIZE)

    def auglag_relax(self,beta_bar, gamma_bar):
        """ initialize the relaxation constraints for the augmented lagrangian """
        self.beta2   = self.m.addVars(self.ebound, lb=0, ub=4*self.consts['fmax']*self.consts['fmax'], name='beta2')
        self.gamma2  = self.m.addVars(self.ebound, lb=0, ub=4*self.consts['fmax']*self.consts['fmax'], name='gamma2')
        self.b2 = {} ; self.g2 = {}
        for l in self.ebound:
            zl = self.zperm[l]
            delta_max = 2*self.z['rate'][zl] # maximum beta/gamma error
            #hbeta = np.round(delta_max**2/self.consts['beta2_err'])
            hbeta = hlp.polyhedral_h(delta_max, self.consts['beta2_err'] )
            d = 2*delta_max/hbeta

            self.m.addConstr( self.beta2[l]  <= delta_max*delta_max )
            self.m.addConstr( self.gamma2[l] <= delta_max*delta_max )
            
            for t in range(int(hbeta)+1):
                self.b2[l,t] = self.m.addConstr( self.beta2[l]  - 2*(-delta_max + t*d - beta_bar[l])*self.beta[l]   >= beta_bar[l]**2  - (-delta_max + t*d)**2 )
                self.g2[l,t] = self.m.addConstr( self.gamma2[l] - 2*(-delta_max + t*d - gamma_bar[l])*self.gamma[l] >= gamma_bar[l]**2 - (-delta_max + t*d)**2 )

    def const_update(self, beta_bar, gamma_bar):
        for l in self.ebound:
            zl = self.zperm[l]
            delta_max = 2*self.z['rate'][zl] # maximum beta/gamma error
            #hbeta = np.round(delta_max**2/self.consts['beta2_err'])
            hbeta = hlp.polyhedral_h(delta_max, self.consts['beta2_err'] )
            d = 2*delta_max/hbeta
            for t in range(int(hbeta)+1):
                beta_coeff  = -2*(-delta_max + t*d - beta_bar[l])
                gamma_coeff = -2*(-delta_max + t*d - gamma_bar[l])
                beta_rhs    = beta_bar[l]**2 - (-delta_max + t*d)**2
                gamma_rhs   = gamma_bar[l]**2 - (-delta_max + t*d)**2
                
                self.b2[l,t].RHS = beta_rhs
                self.g2[l,t].RHS = gamma_rhs
                self.m.chgCoeff(self.b2[l,t], self.beta[l], beta_coeff)
                self.m.chgCoeff(self.g2[l,t], self.gamma[l], gamma_coeff)

    def auglag_error(self):
        e = {'beta': {}, 'gamma': {}}
        for l in self.beta2:
            e['beta'][l]  = (self.beta[l].X  - self.beta_bar[l])**2  - self.beta2[l].X
            e['gamma'][l] = (self.gamma[l].X - self.gamma_bar[l])**2 - self.gamma2[l].X
        return e

    def phi_error(self):
        e = np.empty(self.L)
        for _n1,_n2,_l in self.G.edges_iter(data='id'):
            n1 = self.nmap[_n1]; n2 = self.nmap[_n2];  l = self.lmap[_l];
            e[l] = 0.5*(self.theta[n1].X - self.theta[n2].X)**2 - self.phi[l].X
        return e
    
    def remove_abs_vars(self):
        """ remove the beta_abs and gamma_abs variables and constraints"""
        ### constraints
        self.remove_try(self.bp_abs)
        self.remove_try(self.bn_abs)
        self.remove_try(self.gp_abs)
        self.remove_try(self.gn_abs)
        self.remove_try(self.bp_lim)
        self.remove_try(self.bn_lim)
        self.remove_try(self.gp_lim)
        self.remove_try(self.gn_lim)
        ### variables
        self.remove_try(self.beta_p)
        self.remove_try(self.beta_n)
        self.remove_try(self.gamma_p)
        self.remove_try(self.gamma_n)

        if self.Nbsh > 0:
            def obj(scale=1):
                w = {'sf': 10, 'su':100, 'sd': 100, 'beta':2}
                for k,v in w.items():
                    w[k] = max(v*scale,v)
                w['phi'] = 1.0
                return self.Pg.sum('*') + w['phi']*self.phi.sum('*') + self.Qshp.sum("*") + self.Qshn.sum("*") \
                        + w['sf']*self.sf.sum("*") + w['su']*self.su.sum("*") + w['sd']*self.sd.sum("*")
        else:
            def obj(scale=1):
                w = {'sf': 10, 'su':100, 'sd': 100, 'beta':2}
                for k,v in w.items():
                    w[k] = max(v*scale,v)
                w['phi'] = 1.0
                return self.Pg.sum('*') + w['phi']*self.phi.sum('*') \
                        + w['sf']*self.sf.sum("*") + w['su']*self.su.sum("*") + w['sd']*self.sd.sum("*") 
        self.obj = obj

    def remove_try(self, var):
        try:
            self.m.remove(var)
        except gb.GurobiError:
            for k,v in var.items():
                self.m.remove(v)

    def optimize(self, write_model=False, logger=None, **kwargs):
        self.m._logger = logger
        if write_model:
            self.write(pre=True, **kwargs)
        self.m.optimize(mycallback2)
        if write_model:
            self.write(pre=False, **kwargs)
        try:
            if self.m.solcount == 0:
                self.fix_Pi()
                self.clear_tmpconst()
                self.m.setParam('BarHomogeneous', 1)
                self.m.setParam('NumericFocus', 3)
                if write_model:
                    self.write(pre=True, **kwargs)
                self.m.optimize(mycallback2)
                if write_model:
                    self.write(pre=False, **kwargs)
                self.unfix_Pi()
                # set back to defaults
                self.m.setParam('BarHomogeneous', -1)
                self.m.setParam('NumericFocus', 0)
            else:
                self.store_Pi()
        except AttributeError:
            pass
        #self.clear_tmpconst()

    #def clear_tmpconst(self):
    #    import ipdb; ipdb.set_trace()
    #    for i in range(len(self.m._tmpconst)):
    #        self.m.remove(self.m._tmpconst.pop())

    def store_Pi(self):
            self.m._Pi = {(i,j): self.Pi[i,j].X for i,j in self.Pi.keys()}

    def fix_Pi(self):
        for i,j in self.Pi.keys():
            if self.m._Pi[i,j] > 0.5:
                self.Pi[i,j].vtype = gb.GRB.CONTINUOUS
                self.Pi[i,j].ub = 1
                self.Pi[i,j].lb = 1
            else:
                self.Pi[i,j].vtype = gb.GRB.CONTINUOUS
                self.Pi[i,j].ub = 0
                self.Pi[i,j].lb = 0
        #self.m._Pifixflag = True

    def unfix_Pi(self):
        for i,j in self.Pi.keys():
            self.Pi[i,j].vtype = gb.GRB.BINARY
            self.Pi[i,j].ub = 1
            self.Pi[i,j].lb = 0
        #self.m._Pifixflag = False

    @property
    def objective(self):
        return self.m.objVal

    def set_timelimit(self,tlim):
        self.m.setParam('TimeLimit', tlim)
    
    def getvars(self, Sonly=False, includez=False):
        vars = {}
        if not self.nperm:
            vars['Pgmax'] = hlp.var2mat(self.S['Pgmax'], self.N, perm=self.Pi)
            vars['Pgmin'] = hlp.var2mat(self.S['Pgmin'], self.N, perm=self.Pi)
            vars['Qgmax'] = hlp.var2mat(self.S['Qgmax'], self.N, perm=self.Pi)
        vars['Pd']    = hlp.var2mat(self.Pd, self.N)
        vars['Qd']    = hlp.var2mat(self.Qd, self.N)
        vars['Pg']    = hlp.var2mat(self.Pg, self.N)
        vars['Qg']    = hlp.var2mat(self.Qg, self.N)
        if Sonly:
            return vars
        vars['Pf']    = hlp.var2mat(self.Pf, self.L)
        vars['Qf']    = hlp.var2mat(self.Qf, self.L)
        vars['Pt']    = hlp.var2mat(self.Pt, self.L)
        vars['Qt']    = hlp.var2mat(self.Qt, self.L)
        vars['sf']    = hlp.var2mat(self.sf,  self.L)
        vars['su']    = hlp.var2mat(self.su,  self.N)
        vars['sd']    = hlp.var2mat(self.sd,  self.L)
        vars['theta'] = hlp.var2mat(self.theta, self.N)
        vars['u']     = hlp.var2mat(self.u, self.N)
        vars['phi']   = hlp.var2mat(self.phi,self.L)
        if self.Ngsh > 0:
            vars['GS']= hlp.var2mat(self.Psh,self.N)
        if self.Nbsh > 0:
            vars['BS']= hlp.var2mat(self.Qsh,self.N)
        if includez:
            ### add branch variables
            for k,v in self.z.items():
                try:
                    vars[k] = v[self.zperm][self.rlmap]
                except TypeError:
                    pass
        return vars 

    def get_beta(self):
        return {k: v.X for k,v in self.beta.items()}

    def get_gamma(self):
        return {k: v.X for k,v in self.gamma.items()}

    def write(self, fname='mymodel', pre=True, **kwargs):
        # save model
        s = fname + '_zone_' + str(self.zone) 
        if pre:
            self.m.write(s + '_cnt_' + str(self.wcnt['mps']) + '.mps')
            self.wcnt['mps'] += 1
        # save MIP 
        else:
            try:
                self.m.write(s + '_cnt_' + str(self.wcnt['mst']) + '.mst')
                self.wcnt['mst'] += 1
            except:
                pass

    def sol_check(self):
        vars = self.getvars(includez=True)
        ebound_map = self.m._ebound_map
        vars['beta'] = {i:self.beta[i].X for i in self.beta}
        vars['gamma'] = {i:self.gamma[i].X for i in self.gamma}
        try:
            vars['beta_p'] = {i:self.beta_p[i].X for i in self.beta_p}
            vars['beta_n'] = {i:self.beta_n[i].X for i in self.beta_n}
            vars['gamma_p'] = {i:self.gamma_p[i].X for i in self.gamma_p}
            vars['gamma_n'] = {i:self.gamma_n[i].X for i in self.gamma_n}
        except gb.GurobiError:
            pass
        try:
            vars['beta2']     = {i:self.beta2[i].X     for i in self.beta2}
            vars['gamma2']    = {i:self.gamma2[i].X    for i in self.beta2}
            vars['beta_bar']  = {i:self.beta_bar[i]    for i in self.beta2}
            vars['gamma_bar'] = {i:self.gamma_bar[i]   for i in self.beta2}
        except AttributeError:
            pass
        maps = {'nmap':self.nmap, 'lmap': self.lmap, 'rnmap': self.rnmap, 'rlmap': self.rlmap}
        chk.rescheck(vars,G=self.G, maps=maps, ebound_map=ebound_map)
