import gurobipy as gb
import numpy as np
from scipy import sparse
import networkx as nx
import logging
import ipdb

FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.INFO,datefmt='%H:%M:%S')

def mycallback(model,where):
    """ Create a callback for termination
        Termination is: 
        MIPgap OR (Time Limit AND solution) OR (Solution)
    """
    if where == gb.GRB.Callback.MIPSOL:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPSOL_SOLCNT)
        if (solcnt > 1) and elapsed_time > 500:
            logging.info('      terminating in MISOL')
            model.terminate()
        #elif elapsed_time > 1000 and not model._Zfixflag:
        #    logging.info('      terminating withought solution')
        #    model.terminate()
    elif where == gb.GRB.Callback.MIP:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIP_SOLCNT)
        if (solcnt > 1) and elapsed_time > 500:
            logging.info('      terminating in MIP')
            model.terminate()
        #elif (elapsed_time > 1000) and not model._Zfixflag:
        #    logging.info('      terminating withought solution')
        #    model.terminate()
    elif where == gb.GRB.Callback.MIPNODE:
        elapsed_time = model.cbGet(gb.GRB.Callback.RUNTIME)
        solcnt       = model.cbGet(gb.GRB.Callback.MIPNODE_SOLCNT)
        if (solcnt > 1) and elapsed_time > 500:
            logging.info('      terminating in MIPNODE')
            model.terminate()
        #elif elapsed_time > 1000 and not model._Zfixflag:
        #    logging.info('      terminating withought solution')
        #    model.terminate()
    else:
        pass

class ZoneMILP(object):
    def __init__(self,zone,invars):
        self.zone = zone
        self.invars = invars
        ### data #####
        balance_epsilon = invars['balance_epsilon']
        G = invars['G']
        p = invars['p']
        b = invars['b']
        f_max = invars['f_max']
        delta_max = invars['delta_max']
        M = invars['M']

        node_num = G.number_of_nodes()
        branch_num = G.number_of_edges()
        p_map = dict(zip(range(node_num),   p.keys()))
        b_map = dict(zip(range(branch_num), b.keys()))
        node_mapping = dict(zip(G.nodes(),range(G.number_of_nodes())))
        inv_node_map = {v: k for k,v in node_mapping.items()}
        edge_mapping = {}
        for j,(u,v,l) in enumerate(G.edges_iter(data='id')):
                edge_mapping[l] = j
        inv_edge_map = {v: k for k,v in edge_mapping.items()}
        #node_mapping = invars['n_map']
        #edge_mapping = invars['e_map']
        ebound_map,ebound_list = invars['ebound']
        boundary = [node_mapping[i] for i in invars['boundary']]
        G = nx.relabel_nodes(G,node_mapping,copy=True)
        #mismatch = np.abs(np.sum(p))
        mismatch = sum(p.values())

        ######## Create Model ########
        self.m = gb.Model()
        self.m.setParam('LogFile','/tmp/GurobiZone.log')
        self.m.setParam('LogToConsole',0)
        self.m.setParam('MIPGap',0.15)
        #self.m.setParam('SolutionLimit',5) #stop after this many solutions are found
        self.m.setParam('TimeLimit', 300)
        self.m.setParam('MIPFocus',1)
        self.m.setParam('ImproveStartTime',60)
        self.m.setParam('Threads',60)

        ################
        # Variables
        ################
        #print('Creating Pi variable....',end="",flush=True)
        self.Pi =    self.m.addVars(range(node_num),range(node_num),vtype=gb.GRB.BINARY,name="Pi")
        self.m._Pifixflag = False
        #print('Complete.')
        #print('Creating Z variable....',end="",flush=True)
        self.Z =     self.m.addVars(range(branch_num),range(branch_num),vtype=gb.GRB.BINARY,name="Z")
        self.m._Zfixflag = False
        #print('Complete.')
        #print('Creating theta variable...',end="",flush=True)
        self.theta = self.m.addVars(range(node_num),lb=-3.14,ub=3.14,name="theta")
        #print('Complete.')
        #print('Creating s variable...',end="",flush=True)
        self.s =     self.m.addVars(range(branch_num),name="s")
        #print('Complete.')
        #print('creating f variable...',end="",flush=True)
        self.f =     self.m.addVars(range(branch_num),lb=-f_max,ub=f_max,name="f")
        #print('Complete')

        self.beta       = self.m.addVars(ebound_list, ub=f_max, lb=-f_max, name="beta")
        self.beta_plus  = self.m.addVars(ebound_list,  ub=f_max, name="beta_plus")
        self.beta_minus = self.m.addVars(ebound_list,  ub=f_max, name="beta_minus")
        #self.beta_plus  = self.m.addVars(ebound_list, range(2), ub=f_max, name="beta_plus")
        #self.beta_minus = self.m.addVars(ebound_list, range(2), ub=f_max, name="beta_minus")
   
        #################
        # Objective
        ################
        #print('Setting objective...',end="",flush=True)
        def obj():
            return self.s.sum('*') + self.beta_plus.sum('*') + self.beta_minus.sum('*')
        #obj = self.s.sum('*')
        ##obj += max(1,mismatch)*self.beta_plus.sum('*')
        ##obj += max(1,mismatch)*self.beta_minus.sum('*')
        #obj += self.beta_plus.sum('*')
        #obj += self.beta_minus.sum('*')
        self.obj = obj
        self.m.setObjective(self.obj(),gb.GRB.MINIMIZE) 
        #print('Complete.')

        ##################
        # Constraints
        #################
        #print('Delta Constraints....',end="",flush=True)
        for u,v,l in G.edges_iter(data='id'): 
            self.m.addConstr( ( self.theta[u] - self.theta[v] <=  delta_max),'delta_max[%s]' %(edge_mapping[l]) )
            self.m.addConstr( ( self.theta[u] - self.theta[v] >= -delta_max),'delta_min[%s]' %(edge_mapping[l]) )
        #print('Complete.')
        #print('Flow Constraints...',end="",flush=True)
        for u,v,l in G.edges_iter(data='id'):
            for _,_,l_tilde in G.edges_iter(data='id'):
                self.m.addConstr(
                   (self.f[edge_mapping[l]] + b[b_map[edge_mapping[l_tilde]]]*(self.theta[u] - self.theta[v]) + M*(1-self.Z[edge_mapping[l],edge_mapping[l_tilde]]) >= 0 ),
                   "flow_1[%s,%s]" %(edge_mapping[l],edge_mapping[l_tilde]))
                self.m.addConstr(
                   (self.f[edge_mapping[l]] + b[b_map[edge_mapping[l_tilde]]]*(self.theta[u] - self.theta[v]) - M*(1-self.Z[edge_mapping[l],edge_mapping[l_tilde]]) <= 0 ),
                   "flow_2[%s,%s]" %(edge_mapping[l],edge_mapping[l_tilde]))
        #print('Complete.')
        #print('Balance Constraints...',end="",flush=True)
        for n in range(node_num):
            if n in boundary:
                self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) + \
                            sum(self.beta[l] for l in ebound_map['in'][inv_node_map[n]]) - \
                            sum(self.beta[l] for l in ebound_map['out'][inv_node_map[n]]) <= balance_epsilon),"balance1[%s]" %(n)
                        )
                self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) + \
                            sum(self.beta[l] for l in ebound_map['in'][inv_node_map[n]]) - \
                            sum(self.beta[l] for l in ebound_map['out'][inv_node_map[n]]) >= balance_epsilon),"balance2[%s]" %(n)
                        )

            else:
                self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) <= balance_epsilon),"balance1[%s]" %(n)
                        )
                self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                                sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                                sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) >= -balance_epsilon),"balance2[%s]" %(n)
                        )
        #print('Complete.')
        #print('Permutation Constraints...',end="",flush=True)
        self.m.addConstrs( (self.Pi.sum(n,'*') == 1 for n in range(node_num)),"Pi_rows")
        self.m.addConstrs( (self.Pi.sum('*',n) == 1 for n in range(node_num)),"Pi_cols")
        self.m.addConstrs( (self.Z.sum(l,'*')  == 1 for l in range(branch_num)),"Z_rows")
        self.m.addConstrs( (self.Z.sum('*',l)  == 1 for l in range(branch_num)),"Z_cols")
        #print('Complete.')
        #print('Slack Constraints...',end="",flush=True)
        for u,v,l in G.edges_iter(data='id'):
            self.m.addConstr( (self.s[edge_mapping[l]] + (self.theta[u] - self.theta[v]) >= 0 ),"slack_1[%s]" %(edge_mapping[l]))
            self.m.addConstr( (self.s[edge_mapping[l]] - (self.theta[u] - self.theta[v]) >= 0 ),"slack_2[%s]" %(edge_mapping[l]))
        self.m.addConstrs( (self.beta_plus[l]  >=  self.beta[l] for l in self.beta.keys()), "bp_slack")
        self.m.addConstrs( (self.beta_minus[l] >= -self.beta[l] for l in self.beta.keys()), "bm_slack")
        #print('Complete.')

        #### don't allow zeros load on degree 1 nodes #####
        #### if zone then 0 on degree one is allowed if it is a boundary node
        #print('Degree one constraints...',end="",flush=True)
        for i,deg in G.degree_iter():
            if deg == 1:
                if i not in boundary:
                    for j in np.where(np.array([p[p_map[n]] for n in range(node_num)]) == 0)[0]:
                        self.m.addConstr(self.Pi[i,j] == 0,name="deg_one[%s,%s]" %(i,j))
        #print('Complete.')

        ## Initialze weights
        #self.w = {'bp':{key: 0 for key in self.beta_plus.keys()}, 'bm':{key: 0 for key in self.beta_minus.keys()}}
        self.w = {e: 0 for e in ebound_list}

        ### save a few more variables for later
        self.node_num      = node_num 
        self.branch_num    = branch_num 
        self.node_mapping  = node_mapping 
        self.inv_node_map  = inv_node_map
        self.edge_mapping  = edge_mapping 
        self.inv_edge_map  = inv_edge_map
        self.p_map         = p_map
        self.b_map         = b_map
        self.boundary      = boundary 
        self.ebound_list   = ebound_list
        self.mismatch      = mismatch 
        self.pl_delta      = f_max/4   #delta variable for piecewise linear approximation to quadratic penalty

    def ph_objective_update(self,beta_bar,rho):
        node_mapping = self.node_mapping
        obj = self.obj()
        #obj = self.s.sum('*')
        #obj += max(1,self.mismatch)*self.beta_plus.sum('*')
        #obj += max(1,self.mismatch)*self.beta_minus.sum('*')
        #for i in self.invars['boundary']: #loops over the GLOBAL node ids in boundary
        for i in self.ebound_list:
            #self.w[node_mapping[i]] += rho*(self.beta_val[i] - beta_bar[i])
            self.w[i] += rho*(self.beta_val[i] - beta_bar[i])
            #self.w['bp'][node_mapping[i]] += rho*(self.beta_plus[node_mapping[i]].X -  beta_bar['bp'][i])
            #self.w['bm'][node_mapping[i]] += rho*(self.beta_minus[node_mapping[i]].X - beta_bar['bm'][i])
            
            obj += self.w[i]*self.beta[i] 
            obj += (rho/2)*(self.beta[i] - beta_bar[i])*(self.beta[i] - beta_bar[i])
            #obj += (rho/2)*(3*self.pl_delta*(self.beta_minus[i,1] + self.beta_plus[i,1]) +\
            #        self.pl_delta*(self.beta_minus[i,0] + self.beta_plus[i,0]) - 2*self.beta[i]*beta_bar[i])
        self.m.setObjective(obj)

    def lr_objective_update(self,nu,nu_map):
        obj = self.obj()
        for i in self.ebound_list:
            obj += nu_map[i][self.zone]*nu[i]*self.beta[i]
        self.m.setObjective(obj)

    def optimize(self):
        if self.m.solcount > 0:
            #self.m._vars = [i.X for i in self.m.getVars()]
            self.m._Z  = {(i,j): self.Z[i,j].X for i,j in self.Z.keys()}
            self.m._Pi = {(i,j): self.Pi[i,j].X for i,j in self.Pi.keys()}
        #self.m.optimize()
        self.m.optimize(mycallback)
        self._beta = None
        self._p    = None
        self._b    = None
        self._theta= None
        if self.m.solcount > 0:
            #self.m._vars = [i.X for i in self.m.getVars()]
            if not self.m._Zfixflag:
                self.m._Z = {(i,j): self.Z[i,j].X for i,j in self.Z.keys()}
            if not self.m._Pifixflag:
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
        self.m._Pifixflag = True

    def unfix_Pi(self):
        for i,j in self.Pi.keys():
            self.Pi[i,j].vtype = gb.GRB.BINARY
            self.Pi[i,j].ub = 1
            self.Pi[i,j].lb = 0
        self.m._Pifixflag = False

    def fix_Z(self):
        for i,j in self.Z.keys():
            if self.m._Z[i,j] > 0.5:
                self.Z[i,j].vtype = gb.GRB.CONTINUOUS
                self.Z[i,j].ub = 1
                self.Z[i,j].lb = 1
            else:
                self.Z[i,j].vtype = gb.GRB.CONTINUOUS
                self.Z[i,j].ub = 0
                self.Z[i,j].lb = 0
        self.m._Zfixflag = True

    def unfix_Z(self):
        for i,j in self.Z.keys():
            self.Z[i,j].vtype = gb.GRB.BINARY
            self.Z[i,j].ub = 1
            self.Z[i,j].lb = 0
        self.m._Zfixflag = False

    def fix_start(self):
        """ fix binary variables, therefore guaranteeing a feasible solution"""
        for i,v in enumerate(self.m.getVars()):
            v.start = self.m._vars[i]
        #for i,j in self.Z.keys():
        #    if self.b_map[j] == self.b_out[self.inv_edge_map[i]]:
        #        self.Z[i,j].start = 1
        #    else:
        #        self.Z[i,j].start = 0
        #for i,j in self.Pi.keys():
        #    if self.p_map[j] == self.p_out[self.inv_node_map[i]]:
        #        self.Pi[i,j].start = 1
        #    else:
        #        self.Pi[i,j].start = 0

        
    @property
    def beta_val(self):
        if self._beta is None: 
            self._beta = {}
            #for i in self.invars['boundary']:
            for i in self.ebound_list:
                self._beta[i] = self.beta[i].X
                #self._beta[i] = self.beta[self.node_mapping[i]].X
        return self._beta

    @property
    def p_out(self):
        """ map {node in complete graph: key in injection input dictionary} """
        if self._p is None:
            self._p = {}
            for i,j in self.Pi.keys():
                if self.m._Pi[i,j] > 0.5:
                   self._p[self.inv_node_map[i]] =  self.p_map[j]
        return self._p

    @property
    def b_out(self):
        """ map {branch id in complete graph: key in branch susceptance input dictionary} """
        if self._b is None:
            self._b = {}
            for i,j in self.Z.keys():
                if self.m._Z[i,j] > 0.5:
                    self._b[self.inv_edge_map[i]] = self.b_map[j]
        return self._b

    @property
    def theta_out(self):
        """ return dictionary of angles with GLOBAL graph node ids """
        if self._theta is None:
            self._theta = {}
            for i in self.theta.keys():
                self._theta[self.inv_node_map[i]] = self.theta[i].X
        return self._theta

def full_MILP(invars,zone=False,logfile=None):
    """
        Needed inputs:
        node_num: the number of nodes in the system
        branch_num: the number of branches in the sytem
        p: vector of power injections (not ordered) (in per unit)
        b: vector of branch susceptances (not ordered) (in per unit)
        incident_lines: list of lines incident on each node (dictionary)
        fn,tn: map from branch to from node or to node respectively
        f_max: maximum allowable flow on a line anywhere
        delta_max: maximum allowable angle difference
        M: large number of the disjunctive constraints
        deg_one: list of nodes that have degree one
    """
    ######### Re-introduce variables #######
    balance_epsilon = invars['balance_epsilon']
    G = invars['G']
    p = invars['p']
    b = invars['b']
    f_max = invars['f_max']
    delta_max = invars['delta_max']
    M = invars['M']
    node_num = G.number_of_nodes()
    branch_num = G.number_of_edges()
    if zone:
        node_mapping = invars['n_map']
        edge_mapping = invars['e_map']
        boundary = [node_mapping[i] for i in invars['boundary']]
        G = nx.relabel_nodes(G,node_mapping,copy=True)
        mismatch = np.abs(np.sum(p))
    else:
        edge_mapping = dict(zip(range(branch_num),range(branch_num)))
        boundary = []

    ######## Create Model ########
    m = gb.Model()
    if logfile is not None:
        m.setParam('LogFile',logfile)
    m.setParam('LogToConsole',0)
    m.setParam('MIPGap',1e-1)
    m.setParam('SolutionLimit',5) #stop after this many solutions are found
    m.setParam('MIPFocus',3)
    m.setParam('Threads',60)

    ################
    # Variables
    ################
    print('Creating Pi variable....',end="",flush=True)
    Pi = m.addVars(range(node_num),range(node_num),vtype=gb.GRB.BINARY,name="Pi")
    print('Complete.')
    print('Creating Z variable....',end="",flush=True)
    Z = m.addVars(range(branch_num),range(branch_num),vtype=gb.GRB.BINARY,name="Z")
    print('Complete.')
    print('Creating theta variable...',end="",flush=True)
    theta = m.addVars(range(node_num),lb=-3.14,ub=3.14,name="theta")
    print('Complete.')
    print('Creating s variable...',end="",flush=True)
    s = m.addVars(range(branch_num),name="s")
    print('Complete.')
    print('creating f variable...',end="",flush=True)
    f = m.addVars(range(branch_num),lb=-f_max,ub=f_max,name="f")
    print('Complete')

    if zone:
        beta_plus = m.addVars(boundary,ub=f_max,name="beta_plus")
        beta_minus = m.addVars(boundary,ub=f_max,name="beta_minus")
   
    #################
    # Objective
    ################
    print('Setting objective...',end="",flush=True)
    obj = s.sum('*')
    if zone:
        obj += max(1,mismatch)*beta_plus.sum('*')
        obj += max(1,mismatch)*beta_minus.sum('*')
    m.setObjective(obj,gb.GRB.MINIMIZE) 
    print('Complete.')

    ##################
    # Constraints
    #################
    print('Delta Constraints....',end="",flush=True)
    for u,v,l in G.edges_iter(data='id'): 
        m.addConstr( ( theta[u] - theta[v] <= delta_max),'delta_max[%s]' %(edge_mapping[l]) )
        m.addConstr( ( theta[u] - theta[v] >= -delta_max),'delta_min[%s]' %(edge_mapping[l]) )
    print('Complete.')
    print('Flow Constraints...',end="",flush=True)
    for u,v,l in G.edges_iter(data='id'):
        for _,_,l_tilde in G.edges_iter(data='id'):
            m.addConstr(
               (f[edge_mapping[l]] + b[edge_mapping[l_tilde]]*(theta[u] - theta[v]) + M*(1-Z[edge_mapping[l],edge_mapping[l_tilde]]) >= 0 ),
               "flow_1[%s,%s]" %(edge_mapping[l],edge_mapping[l_tilde]))
            m.addConstr(
               (f[edge_mapping[l]] + b[edge_mapping[l_tilde]]*(theta[u] - theta[v]) - M*(1-Z[edge_mapping[l],edge_mapping[l_tilde]]) <= 0 ),
               "flow_2[%s,%s]" %(edge_mapping[l],edge_mapping[l_tilde]))
    print('Complete.')
    print('Balance Constraints...',end="",flush=True)
    for n in range(node_num):
        if n in boundary:
            m.addConstr(
                    ( sum(p[n_tilde]*Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                        sum(f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                        sum(f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) + \
                        beta_plus[n] - beta_minus[n] <= balance_epsilon),"balance1[%s]" %(n)
                    )
            m.addConstr(
                    ( sum(p[n_tilde]*Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) + \
                            beta_plus[n] - beta_minus[n] >= -balance_epsilon),"balance2[%s]" %(n)
                    )

        else:
            m.addConstr(
                    ( sum(p[n_tilde]*Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                        sum(f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                        sum(f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) <= balance_epsilon),"balance1[%s]" %(n)
                    )
            m.addConstr(
                    ( sum(p[n_tilde]*Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) >= -balance_epsilon),"balance2[%s]" %(n)
                    )
    print('Complete.')
    print('Permutation Constraints...',end="",flush=True)
    m.addConstrs( (Pi.sum(n,'*') == 1 for n in range(node_num)),"Pi_rows")
    m.addConstrs( (Pi.sum('*',n) == 1 for n in range(node_num)),"Pi_cols")
    m.addConstrs( (Z.sum(l,'*') == 1 for l in range(branch_num)),"Z_rows")
    m.addConstrs( (Z.sum('*',l) == 1 for l in range(branch_num)),"Z_cols")
    print('Complete.')
    print('Slack Constraints...',end="",flush=True)
    for u,v,l in G.edges_iter(data='id'):
        m.addConstr( (s[edge_mapping[l]] + (theta[u] - theta[v]) >= 0 ),"slack_1[%s]" %(edge_mapping[l]))
        m.addConstr( (s[edge_mapping[l]] - (theta[u] - theta[v]) >= 0 ),"slack_2[%s]" %(edge_mapping[l]))
    print('Complete.')

    #### don't allow zeros load on degree 1 nodes #####
    #### if zone then 0 on degree one is allowed if it is a boundary node
    print('Degree one constraints...',end="",flush=True)
    for i,deg in G.degree_iter():
        if deg == 1:
            if i not in boundary:
                for j in np.where(p == 0)[0]:
                    m.addConstr(Pi[i,j] == 0,name="deg_one[%s,%s]" %(i,j))
    print('Complete.')

    m.optimize()
    return m

def only_p(invars,set_start=False,logfile=None):
    """
        Needed inputs:
        node_num: the number of nodes in the system
        branch_num: the number of branches in the sytem
        p: vector of power injections (not ordered) (in per unit)
        b: vector of branch susceptances (not ordered) (in per unit)
        incident_lines: list of lines incident on each node (dictionary)
        fn,tn: map from branch to from node or to node respectively
        f_max: maximum allowable flow on a line anywhere
        delta_max: maximum allowable angle difference
        M: large number of the disjunctive constraints
        deg_one: list of nodes that have degree one
    """
    ######### Re-introduce variables #######
    balance_epsilon = invars['balance_epsilon']
    node_num = invars['node_num']
    branch_num = invars['branch_num']
    p = invars['p']
    b = invars['b']
    from_lines = invars['from_lines']
    to_lines = invars['to_lines']
    fn = invars['fn']
    tn = invars['tn']
    f_max = invars['f_max']
    delta_max = invars['delta_max']
    M = invars['M']
    deg_one = invars['deg_one']

    ######## Create Model ########
    m = gb.Model()
    if logfile is not None:
        m.setParam('LogFile',logfile)
    m.setParam('LogToConsole',0)
    m.setParam('MIPGap',1e-1)
    m.setParam('SolutionLimit',3) #stop after this many solutions are found
    m.setParam('MIPFocus',3)
    m.setParam('Threads',60)

    ################
    # Variables
    ################
    print('Creating Pi variable....',end="",flush=True)
    Pi = m.addVars(range(node_num),range(node_num),vtype=gb.GRB.BINARY,name="Pi")
    print('Complete.')
    print('Creating theta variable...',end="",flush=True)
    theta = m.addVars(range(node_num),lb=-3.14,ub=3.14,name="theta")
    print('Complete.')
    print('Creating s variable...',end="",flush=True)
    s = m.addVars(range(branch_num),name="s")
    print('Complete.')
    print('creating f variable...',end="",flush=True)
    f = m.addVars(range(branch_num),lb=-f_max,ub=f_max,name="f")
    print('Complete')
    
    if set_start:
        for i,j in Pi:
            if i == j:
                Pi[i,j].start = 1
            else:
                Pi[i,j].start = 0
        for l in range(branch_num):
            f[l].start = invars['flow'][l]
            s[l].start = invars['slack'][l]
        for i in range(node_num):
            theta[i].start = invars['theta'][i]

    #################
    # Objective
    ################
    print('Setting objective...',end="",flush=True)
    m.setObjective(s.sum('*'),gb.GRB.MINIMIZE) 
    print('Complete.')

    ##################
    # Constraints
    #################
    print('Delta Constraints....',end="",flush=True)
    m.addConstrs( ( theta[fn[l]] - theta[tn[l]] <= delta_max for l in range(branch_num)),'delta_max')
    m.addConstrs( ( theta[fn[l]] - theta[tn[l]] >= -delta_max for l in range(branch_num)),'delta_min')
    print('Complete.')
    print('Flow Constraints...',end="",flush=True)
    m.addConstrs(
            (f[l] + b[l]*(theta[fn[l]] - theta[tn[l]]) == 0 for l in range(branch_num) ),"flow"
            )
    print('Complete.')
    print('Balance Constraints...',end="",flush=True)
    m.addConstrs(
            ( sum(p[n_tilde]*Pi[n,n_tilde] for n_tilde in range(node_num)) + sum(f[l] for l in to_lines[n]) - sum(f[l] for l in from_lines[n]) <= balance_epsilon for n in range(node_num)),"balance1"
            )
    m.addConstrs(
            ( sum(p[n_tilde]*Pi[n,n_tilde] for n_tilde in range(node_num)) + sum(f[l] for l in to_lines[n]) - sum(f[l] for l in from_lines[n]) >= -balance_epsilon for n in range(node_num)),"balance2"
            )
    print('Complete.')
    print('Permutation Constraints...',end="",flush=True)
    m.addConstrs( (Pi.sum(n,'*') == 1 for n in range(node_num)),"Pi_rows")
    m.addConstrs( (Pi.sum('*',n) == 1 for n in range(node_num)),"Pi_cols")
    print('Complete.')
    print('Slack Constraints...',end="",flush=True)
    m.addConstrs( (s[l] + (theta[fn[l]] - theta[tn[l]]) >= 0 for l in range(branch_num)),"slack_1")
    m.addConstrs( (s[l] - (theta[fn[l]] - theta[tn[l]]) >= 0 for l in range(branch_num)),"slack_2")
    print('Complete.')

    #### don't allow zeros load on degree 1 nodes #####
    print('Degree one constraints...',end="",flush=True)
    for i in deg_one:
        for j in np.where(p == 0)[0]:
            m.addConstr(Pi[i,j] == 0,name="deg_one[%s,%s]" %(i,j))
    print('Complete.')

    m.optimize()
    return m

def only_b(invars,set_start=False,logfile=None):
    """
        Needed inputs:
        node_num: the number of nodes in the system
        branch_num: the number of branches in the sytem
        p: vector of power injections (not ordered) (in per unit)
        b: vector of branch susceptances (not ordered) (in per unit)
        incident_lines: list of lines incident on each node (dictionary)
        fn,tn: map from branch to from node or to node respectively
        f_max: maximum allowable flow on a line anywhere
        delta_max: maximum allowable angle difference
        M: large number of the disjunctive constraints
        deg_one: list of nodes that have degree one
    """
    ######### Re-introduce variables #######
    balance_epsilon = invars['balance_epsilon']
    node_num = invars['node_num']
    branch_num = invars['branch_num']
    p = invars['p']
    b = invars['b']
    from_lines = invars['from_lines']
    to_lines = invars['to_lines']
    fn = invars['fn']
    tn = invars['tn']
    f_max = invars['f_max']
    delta_max = invars['delta_max']
    M = invars['M']
    deg_one = invars['deg_one']

    ######## Create Model ########
    m = gb.Model()
    if logfile is not None:
        m.setParam('LogFile',logfile)
    m.setParam('LogToConsole',0)
    m.setParam('MIPGap',1e-1)
    m.setParam('SolutionLimit',3) #stop after this many solutions are found
    m.setParam('MIPFocus',3)
    m.setParam('Threads',60)

    ################
    # Variables
    ################
    print('Creating Z variable....',end="",flush=True)
    Z = m.addVars(range(branch_num),range(branch_num),vtype=gb.GRB.BINARY,name="Z")
    print('Complete.')
    print('Creating theta variable...',end="",flush=True)
    theta = m.addVars(range(node_num),lb=-3.14,ub=3.14,name="theta")
    print('Complete.')
    print('Creating s variable...',end="",flush=True)
    s = m.addVars(range(branch_num),name="s")
    print('Complete.')
    print('creating f variable...',end="",flush=True)
    f = m.addVars(range(branch_num),lb=-f_max,ub=f_max,name="f")
    print('Complete')
    
    if set_start:
        for i,j in Z:
            if i == j:
                Z[i,j].start = 1
            else:
                Z[i,j].start = 0
        for l in range(branch_num):
            f[l].start = invars['flow'][l]
            s[l].start = invars['slack'][l]
        for i in range(node_num):
            theta[i].start = invars['theta'][i]
            
    #################
    # Objective
    ################
    print('Setting objective...',end="",flush=True)
    m.setObjective(s.sum('*'),gb.GRB.MINIMIZE) 
    print('Complete.')

    ##################
    # Constraints
    #################
    print('Delta Constraints....',end="",flush=True)
    m.addConstrs( ( theta[fn[l]] - theta[tn[l]] <= delta_max for l in range(branch_num)),'delta_max')
    m.addConstrs( ( theta[fn[l]] - theta[tn[l]] >= -delta_max for l in range(branch_num)),'delta_min')
    print('Complete.')
    print('Flow Constraints...',end="",flush=True)
    m.addConstrs(
            (f[l] + b[l_tilde]*(theta[fn[l]] - theta[tn[l]]) + M*(1-Z[l,l_tilde]) >= 0 for l in range(branch_num) for l_tilde in range(branch_num)),"flow_1"
            )
    m.addConstrs(
            (f[l] + b[l_tilde]*(theta[fn[l]] - theta[tn[l]]) - M*(1-Z[l,l_tilde]) <= 0 for l in range(branch_num) for l_tilde in range(branch_num)),"flow_2"
            )
    print('Complete.')
    print('Balance Constraints...',end="",flush=True)
    m.addConstrs(
            ( p[n] + sum(f[l] for l in to_lines[n]) - sum(f[l] for l in from_lines[n]) <= balance_epsilon for n in range(node_num)),"balance1"
            )
    m.addConstrs(
            ( p[n] + sum(f[l] for l in to_lines[n]) - sum(f[l] for l in from_lines[n]) >= -balance_epsilon for n in range(node_num)),"balance2"
            )
    print('Complete.')
    print('Permutation Constraints...',end="",flush=True)
    m.addConstrs( (Z.sum(l,'*') == 1 for l in range(branch_num)),"Z_rows")
    m.addConstrs( (Z.sum('*',l) == 1 for l in range(branch_num)),"Z_cols")
    print('Complete.')
    print('Slack Constraints...',end="",flush=True)
    m.addConstrs( (s[l] + (theta[fn[l]] - theta[tn[l]]) >= 0 for l in range(branch_num)),"slack_1")
    m.addConstrs( (s[l] - (theta[fn[l]] - theta[tn[l]]) >= 0 for l in range(branch_num)),"slack_2")
    print('Complete.')

    m.optimize()
    return m
