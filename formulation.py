import sys
import itertools
import gurobipy as gb
import numpy as np
from scipy import sparse
import networkx as nx
import logging

FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

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
        elif (elapsed_time > 1000) and (not model._parflag):
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
        elif (elapsed_time > 1000) and (not model._parflag):
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
        elif (elapsed_time > 1000) and (not model._parflag):
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
        beta_max = invars['beta_max']
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
        #self.m.setParam('TimeLimit', 300)
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
        self.m._parflag  = False
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

        self.beta       = self.m.addVars(ebound_list, ub=beta_max, lb=-beta_max, name="beta")
        self.beta_plus  = self.m.addVars(ebound_list,  ub=beta_max, name="beta_plus")
        self.beta_minus = self.m.addVars(ebound_list,  ub=beta_max, name="beta_minus")
        #self.beta_plus  = self.m.addVars(ebound_list, range(2), ub=f_max, name="beta_plus")
        #self.beta_minus = self.m.addVars(ebound_list, range(2), ub=f_max, name="beta_minus")
   
        #################
        # Objective
        ################
        #print('Setting objective...',end="",flush=True)
        def obj(creg):
            return creg*self.s.sum('*') + self.beta_plus.sum('*') + self.beta_minus.sum('*')
        #obj = self.s.sum('*')
        ##obj += max(1,mismatch)*self.beta_plus.sum('*')
        ##obj += max(1,mismatch)*self.beta_minus.sum('*')
        #obj += self.beta_plus.sum('*')
        #obj += self.beta_minus.sum('*')
        self.obj = obj
        self.m.setObjective(self.obj(invars['creg']),gb.GRB.MINIMIZE) 
        #print('Complete.')
        
        ##################
        # Constraints
        #################
        #print('Delta Constraints....',end="",flush=True)
        for u,v,l in G.edges_iter(data='id'): 
            self.m.addConstr( ( self.theta[u] - self.theta[v] <=  delta_max), name='delta_max[%s]' %(edge_mapping[l]) )
            self.m.addConstr( ( self.theta[u] - self.theta[v] >= -delta_max), name='delta_min[%s]' %(edge_mapping[l]) )
            if l in invars['pmap']:
                # parallel branches should have similar susceptances
                self.m.addConstr( sum(self.Z[edge_mapping[l],i]*b[b_map[i]] for i in range(branch_num)) - \
                                  sum(self.Z[edge_mapping[invars['pmap'][l]],i]*b[b_map[i]] for i in range(branch_num))>= -invars['pepsilon'], name='pepsilon_min[%s]' %(edge_mapping[l]))
                self.m.addConstr( sum(self.Z[edge_mapping[l],i]*b[b_map[i]] for i in range(branch_num)) - \
                                  sum(self.Z[edge_mapping[invars['pmap'][l]],i]*b[b_map[i]] for i in range(branch_num))<=  invars['pepsilon'], name='pepsilon_max[%s]' %(edge_mapping[l]))
        #print('Complete.')
        #print('Flow Constraints...',end="",flush=True)
        for u,v,l in G.edges_iter(data='id'):
            for _,_,l_tilde in G.edges_iter(data='id'):
                self.m.addConstr(
                   (self.f[edge_mapping[l]] + b[b_map[edge_mapping[l_tilde]]]*(self.theta[u] - self.theta[v]) + M*(1-self.Z[edge_mapping[l],edge_mapping[l_tilde]]) >= 0 ),
                   name="flow_1[%s,%s]" %(edge_mapping[l],edge_mapping[l_tilde]))
                self.m.addConstr(
                   (self.f[edge_mapping[l]] + b[b_map[edge_mapping[l_tilde]]]*(self.theta[u] - self.theta[v]) - M*(1-self.Z[edge_mapping[l],edge_mapping[l_tilde]]) <= 0 ),
                   name="flow_2[%s,%s]" %(edge_mapping[l],edge_mapping[l_tilde]))
        #print('Complete.')
        #print('Balance Constraints...',end="",flush=True)
        self.bound_const1 = {}
        self.bound_const2 = {}
        for n in range(node_num):
            if n in boundary:
                self.bound_const1[n] = self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) + \
                            sum(self.beta[l] for l in ebound_map['in'][inv_node_map[n]]) - \
                            sum(self.beta[l] for l in ebound_map['out'][inv_node_map[n]]) <= balance_epsilon), name="balance1[%s]" %(n)
                        )
                self.bound_const2[n] = self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) + \
                            sum(self.beta[l] for l in ebound_map['in'][inv_node_map[n]]) - \
                            sum(self.beta[l] for l in ebound_map['out'][inv_node_map[n]]) >= balance_epsilon), name="balance2[%s]" %(n)
                        )

            else:
                self.bound_const1[n] = self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                            sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                            sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) <= balance_epsilon), name="balance1[%s]" %(n)
                        )
                self.bound_const2[n] = self.m.addConstr(
                        ( sum(p[p_map[n_tilde]]*self.Pi[n,n_tilde] for n_tilde in range(node_num)) + \
                                sum(self.f[edge_mapping[l['id']]] for _,_,l in G.in_edges_iter([n],data='id')) - \
                                sum(self.f[edge_mapping[l]] for _,_,l in G.out_edges_iter([n],data='id')) >= -balance_epsilon), name="balance2[%s]" %(n)
                        )
        #print('Complete.')
        #print('Permutation Constraints...',end="",flush=True)
        self.m.addConstrs( (self.Pi.sum(n,'*') == 1 for n in range(node_num)),   name="Pi_rows")
        self.m.addConstrs( (self.Pi.sum('*',n) == 1 for n in range(node_num)),   name="Pi_cols")
        self.m.addConstrs( (self.Z.sum(l,'*')  == 1 for l in range(branch_num)), name="Z_rows")
        self.m.addConstrs( (self.Z.sum('*',l)  == 1 for l in range(branch_num)), name="Z_cols")
        #print('Complete.')
        #print('Slack Constraints...',end="",flush=True)
        for u,v,l in G.edges_iter(data='id'):
            self.m.addConstr( (self.s[edge_mapping[l]] + (self.theta[u] - self.theta[v]) >= 0 ), name="slack_1[%s]" %(edge_mapping[l]))
            self.m.addConstr( (self.s[edge_mapping[l]] - (self.theta[u] - self.theta[v]) >= 0 ), name="slack_2[%s]" %(edge_mapping[l]))
        self.bp_slack = self.m.addConstrs( (self.beta_plus[l]  >=  self.beta[l] for l in self.beta.keys()), name="bp_slack")
        self.bm_slack = self.m.addConstrs( (self.beta_minus[l] >= -self.beta[l] for l in self.beta.keys()), name="bm_slack")
        #print('Complete.')
        
        inlist  = np.concatenate( [np.array(val) for val in ebound_map['in'].values()])
        outlist = np.concatenate( [np.array(val) for val in ebound_map['out'].values()])
        for l in ebound_list:
            if l in invars['pmap']:
                if ((l in inlist) and (invars['pmap'][l] in inlist)) or ((l in outlist) and (invars['pmap'][l] in outlist)):
                    # same orientatin of boundary parallel edges
                    self.m.addconstr( self.beta[l] - self.beta[invars['pmap'][l]] == 0)
                else:
                    # opposite orientation
                    self.m.addconstr( self.beta[l] + self.beta[invars['pmap'][l]] == 0)

        #### don't allow zeros load on degree 1 nodes #####
        #### if zone then 0 on degree one is allowed if it is a boundary node
        #print('Degree one constraints...',end="",flush=True)
#        for i,deg in G.degree_iter():
#            if deg == 1:
#                if i not in boundary:
#                    for j in np.where(np.array([p[p_map[n]] for n in range(node_num)]) == 0)[0]:
#                        self.m.addConstr(self.Pi[i,j] == 0, name="deg_one[%s,%s]" %(i,j))
        #print('Complete.')

        ## Initialze weights
        #self.w = {'bp':{key: 0 for key in self.beta_plus.keys()}, 'bm':{key: 0 for key in self.beta_minus.keys()}}
        self.w = {e: 0 for e in ebound_list}

        ### save a few more variables for later
        self.G             = G
        self.node_num      = node_num 
        self.branch_num    = branch_num 
        self.node_mapping  = node_mapping 
        self.inv_node_map  = inv_node_map
        self.edge_mapping  = edge_mapping 
        self.inv_edge_map  = inv_edge_map
        self.p_map         = p_map
        self.p             = p
        self.b_map         = b_map
        self.b             = b
        self.boundary      = boundary 
        self.ebound_list   = ebound_list
        self.ebound_map    = ebound_map
        self.mismatch      = mismatch 
        self.f_max         = f_max
        self.delta_max     = delta_max
        self.Pg            = invars['Pg']
        self.Pd            = invars['Pd']
        self.creg          = invars['creg']

        #self.fix_parallel_b()

    def ph_objective_update(self,beta_bar,rho):
        node_mapping = self.node_mapping
        obj = self.obj(self.creg)
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
        obj = self.obj(self.creg)
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
    
    def fix_parallel_b(self):

        logging.info('Fixing parallel b values')
        self.m._parflag = True
        inv_b_map = {v:k for k,v in self.b_map.items()}
        sorted_b = sorted(self.b, key=self.b.get)   # keys of b for values of b in sorted order
        bdiff    = np.abs(np.diff(sorted(self.b.values()))) #absolute value of difference between sorted values
        sort_idx = np.argsort(bdiff)
        logging.debug('sorted bdiff = [%0.2g, %0.2g, %0.2g, ...]', bdiff[sort_idx[0]], bdiff[sort_idx[1]], bdiff[sort_idx[2]])
        map = {}
        cnt = 0
        for _,_,l in self.G.edges_iter(data='id'):
            if l in self.invars['pmap']:
                row1 = self.edge_mapping[l]
                row2 = self.edge_mapping[self.invars['pmap'][l]]
                logging.debug( 'edge %d/%d (ext/int) is parallel to %d/%d (ext/int)',l,row1,self.invars['pmap'][l],row2)
                if row1 in map:
                    logging.info('parallel case 1')
                    col1 = None
                    dir = 1
                    while True:
                        col2_ptr = map[row1] + dir
                        if col2_ptr > len(sorted_b):
                            dir = -1
                            continue
                        if inv_b_map[sorted_b[col2_ptr]] not in map.values():
                            break
                        else:
                            dir += np.sign(dir)
                    map[row2] = col2_ptr
                    col2 = inv_b_map[sorted_b[col2_ptr]]
                elif row2 in map:
                    logging.debug('parallel case 2')
                    col2 = None
                    dir = 1
                    while True:
                        col1_ptr = map[row2] + dir
                        if col1_ptr > len(sorted_b):
                            dir = -1
                            continue
                        if inv_b_map[sorted_b[col1_ptr]] not in map.values():
                            break
                        else:
                            dir += np.sign(dir)
                    map[row1] = col1_ptr
                    col1 = inv_b_map[sorted_b[col1_ptr]]
                else:
                    logging.debug('parallel case 3')
                    while True: 
                        col1_ptr = sort_idx[cnt]
                        col2_ptr = col1_ptr + 1
                        cnt += 1
                        if (col1_ptr not in map.values()) and (col2_ptr not in map.values()):
                            break
                    col1 = inv_b_map[sorted_b[col1_ptr]]
                    col2 = inv_b_map[sorted_b[col2_ptr]]
                    map[row1] = col1_ptr
                    map[row2] = col2_ptr
                for row,col in zip([row1,row2],[col1,col2]):
                    if col is not None:
                        for i in range(self.branch_num):
                            if i == col:
                                self.Z[row,i].ub = 1
                                self.Z[row,i].lb = 1
                            else:
                                self.Z[row,i].ub = 0
                                self.Z[row,i].lb = 0
                        for i in range(self.branch_num):
                            if i == row:
                                self.Z[i,col].ub = 1
                                self.Z[i,col].lb = 1
                            else:
                                self.Z[i,col].ub = 0
                                self.Z[i,col].lb = 0
        for k,v in map.items():
            logging.debug('edge %d (interal) mapped to entry %d of b (internal), b=%0.3f', k,inv_b_map[sorted_b[v]],self.b[sorted_b[v]])


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
        self.m._parflag = True

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

    def fix_beta(self,beta_bar):
        """ fix beta values, remove the beta slack variables and related constraints"""
        for l in self.beta.keys():
            ## fix beta value
            self.beta[l].lb = beta_bar[l]
            self.beta[l].ub = beta_bar[l]
            ## remove constraints
            #self.m.remove(self.m.getConstrByName("bp_slack[%s]" %(l)))
            #self.m.remove(self.m.getConstrByName("bm_slack[%s]" %(l)))
            self.m.remove(self.bp_slack[l])
            self.m.remove(self.bm_slack[l])
            ## remove beta_plus and minus
            self.m.remove(self.beta_plus[l])
            self.m.remove(self.beta_minus[l])

    def add_balance_slack(self):
        """ add slack variables to balance constraints """
        #constr_list =  [self.m.getConstrByName("balance1[%s]" %(i)) for i in range(self.node_num)]
        #constr_list += [self.m.getConstrByName("balance2[%s]" %(i)) for i in range(self.node_num)]
        constr_list =  [self.bound_const1[i] for i in range(self.node_num)]
        constr_list += [self.bound_const2[i] for i in range(self.node_num)]
        self.slack1 = {}; self.slack2= {};
        for n in range(self.node_num):
            self.slack1[n] = self.m.addVar(column=gb.Column(coeffs=[ 1 for i in range(2)], constrs=[self.bound_const1[n], self.bound_const2[n]]), name="slack1[%s]" %(n)) 
            self.slack2[n] = self.m.addVar(column=gb.Column(coeffs=[-1 for i in range(2)], constrs=[self.bound_const1[n], self.bound_const2[n]]), name="slack2[%s]" %(n)) 


    def balance_slack_objective(self,penalty):
        obj = self.s.sum('*')
        for n in range(self.node_num):
            #obj += penalty*(self.slack1[n]*self.slack1[n] + self.slack2[n]*self.slack2[n])
            obj += penalty*(self.slack1[n] + self.slack2[n])
        self.m.setObjective(obj, gb.GRB.MINIMIZE)

    def fixed_beta(self,beta_bar,gen_params,load_params):
        m = gb.Model()
        m.setParam('LogFile','/tmp/GurobiZone.log')
        m.setParam('LogToConsole',0)
        m.setParam('MIPGap',0.15)
        m.setParam('Threads',60)
        
        Pg_non_zero = {i for i in range(self.node_num) if self.Pg[self.p_out[self.inv_node_map[i]]] != 0}
        Pd_non_zero = {i for i in range(self.node_num) if self.Pd[self.p_out[self.inv_node_map[i]]] != 0}

        #### Variables#######
        alpha = m.addVars(range(self.node_num),lb=0,name="alpha")
        theta = m.addVars(range(self.node_num),lb=-3.14,ub=3.14,name="theta")
        s =     m.addVars(range(self.branch_num),name="s")
        f =     m.addVars(range(self.branch_num),lb=-self.f_max,ub=self.f_max,name="f")

        ### Constraints ####
        for u,v,l in self.G.edges_iter(data='id'): 
            m.addConstr( (theta[u] - theta[v] <=  self.delta_max), name='delta_max[%s]' %(self.edge_mapping[l]) )
            m.addConstr( (theta[u] - theta[v] >= -self.delta_max), name='delta_min[%s]' %(self.edge_mapping[l]) )
            m.addConstr( (f[self.edge_mapping[l]] + self.b[self.b_out[l]]*(theta[u] - theta[v]) == 0), name='flow[%s]' %(self.edge_mapping[l]))
            m.addConstr( (s[self.edge_mapping[l]] + (theta[u] - theta[v]) >= 0 ), name="slack_1[%s]" %(self.edge_mapping[l]))
            m.addConstr( (s[self.edge_mapping[l]] - (theta[u] - theta[v]) >= 0 ), name="slack_2[%s]" %(self.edge_mapping[l]))

        for n in range(self.node_num):
            if n in self.boundary:
                m.addConstr(
                        (  alpha[n]*self.p[self.p_out[self.inv_node_map[n]]] + \
                            sum(f[self.edge_mapping[l['id']]] for _,_,l in self.G.in_edges_iter([n],data='id')) - \
                            sum(f[self.edge_mapping[l]] for _,_,l in self.G.out_edges_iter([n],data='id')) + \
                            sum(beta_bar[l] for l in self.ebound_map['in' ][self.inv_node_map[n]]) - \
                            sum(beta_bar[l] for l in self.ebound_map['out'][self.inv_node_map[n]]) == 0), name="balance[%s]" %(n)
                        )
            else:
                m.addConstr(
                        (  alpha[n]*self.p[self.p_out[self.inv_node_map[n]]] + \
                            sum(f[self.edge_mapping[l['id']]] for _,_,l in self.G.in_edges_iter([n],data='id')) - \
                            sum(f[self.edge_mapping[l]] for _,_,l in self.G.out_edges_iter([n],data='id')) == 0), name="balance[%s]" %(n)
                        )

        m.addConstrs((alpha[i]*self.Pg[self.p_out[self.inv_node_map[i]]] >= gen_params['vmin']  for i in Pg_non_zero), name="Pgmin")
        m.addConstrs((alpha[i]*self.Pg[self.p_out[self.inv_node_map[i]]] <= gen_params['vmax']  for i in Pg_non_zero), name="Pgmax")
        m.addConstrs((alpha[i]*self.Pd[self.p_out[self.inv_node_map[i]]] >= load_params['vmin'] for i in Pd_non_zero), name="Pdmin")
        m.addConstrs((alpha[i]*self.Pd[self.p_out[self.inv_node_map[i]]] <= load_params['vmax'] for i in Pd_non_zero), name="Pdmax")

        ### objective ###
        pmax = abs(max(self.p.values(), key=lambda x: abs(x)))
        obj = gb.QuadExpr()
        for i in alpha:
            obj += (abs(self.p[self.p_out[self.inv_node_map[i]]])/pmax)*(alpha[i]- 1)*(alpha[i] - 1)
        
        m.setObjective(obj,gb.GRB.MINIMIZE)
        #m.write('zone%d_fixed_beta.lp' %(self.zone))
        m.optimize()
        if m.status == 2:
            # return dictionary keyed by node number and with values the multiple for the injection vector
            alpha_out = {self.inv_node_map[i]: alpha[i].X if alpha[i].X > 0 else 1.0 for i in alpha}
            logging.info("Model solved successfully: objective: %0.3f, mean(alpha) = %0.3f, max(alpha) = %0.3f, min(alpha) = %0.3f", 
                    m.objVal,np.mean(list(alpha_out.values())), max(alpha_out.values()), min(alpha_out.values()))
            self.alpha_out = alpha_out
        else:
            logging.info("Model solved unsuccessfuly with status %d", m.status)
            sys.exit(0)

    @property
    def total_slack(self):
        out =  sum(self.slack1[i].X for i in range(self.node_num))
        out += sum(self.slack2[i].X for i in range(self.node_num))
        return out

    @property
    def slack_stat(self):
        out = [self.slack1[i].X if self.slack1[i].X != 0 else self.slack2[i].X for i in range(self.node_num)]
        return {'mean': np.mean(out), 'std': np.std(out)}

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


    
def fix_beta_and_rescale(G,Pg_out,Pd_out,b_out,params):
    """ solve the full model fixing injections and impedances.
    At this point the model is linear so the full system can be solved
    Load and generation are rescaled as little as possible to 
    satisfy all constraints
    
    params must include the delta_max, f_max, Pgmax, Pgmin, Pdmax, and Pdmin
    """
    m = gb.Model()
    
    
    # Define sets similar to the load balancing problem:
    # * where are the generators
    # * Where are the loads
    # * When both, when is generation greater than load, and other way around
    
    pd_ls_pg = np.where(((Pg_out-Pd_out) > 0) & (Pd_out > 0) & (Pg_out > 0))[0]
    pd_gr_pg = np.where(((Pg_out-Pd_out) < 0) & (Pd_out > 0) & (Pg_out > 0))[0]
    Pg_map = np.where(Pg_out>0)[0]
    Pd_map = np.where(Pd_out>0)[0]
    
    
    # Define constants
    #Pgmax = max(Pg_out)
    #Pdmax = max(Pd_out)
    #delta_max       = 60.0*np.pi/180.0
    #f_max           = 10
    
    
    # Define variables $\alpha_g$ and $\alpha_d$ which scale the load, in addition to the standard DC powerflow variables
    
    alpha_g = m.addVars(Pg_map,lb=0)
    alpha_d = m.addVars(Pd_map,lb=0)
    theta   = m.addVars(G.number_of_nodes(), lb=-3.14 , ub=3.14 , name="theta")
    f       = m.addVars(G.number_of_edges(), lb=-params['f_max'], ub=params['f_max'], name="f")
    
    
    # Add constraints similar to generation/load balancing:
    # * The load and generation must remain equal
    # * Generation must remain within bounds
    # * Load must remain within bounds
    # * Where load and generation are present their relationship (which one is bigger) cannot change
    
    m.addConstr(sum(alpha_g[i]*Pg_out[i] for i in Pg_map) - sum(alpha_d[i]*Pd_out[i] for i in Pd_map) == 0);
    m.addConstrs(alpha_g[i]*Pg_out[i] >= params['Pgmin']  for i in Pg_map);
    m.addConstrs(alpha_g[i]*Pg_out[i] <= params['Pgmax']  for i in Pg_map);
    m.addConstrs(alpha_d[i]*Pd_out[i] >= params['Pdmin'] for i in Pd_map);
    m.addConstrs(alpha_d[i]*Pd_out[i] <= params['Pdmax'] for i in Pd_map);
    m.addConstrs(alpha_d[i]*Pd_out[i] <= alpha_g[i]*Pg_out[i]   for i in pd_ls_pg);
    m.addConstrs(alpha_d[i]*Pd_out[i] >= alpha_g[i]*Pg_out[i]   for i in pd_gr_pg);
    m.addConstr(sum(alpha_g[i]*Pg_out[i] for i in Pg_map) == sum(Pg_out)); # constraint forcing the total load/gen to remain the same
    
    
    # Add $\Delta\theta$ constraints and $B\theta$ description of powerflow constraints.
    # Also adds variable $s$ constraints that allow the access the absolute value of the angle difference.
    
    for u,v,l in G.edges_iter(data='id'): 
        m.addConstr( (theta[u] - theta[v] <=  params['delta_max']), name='delta_max[%s]' %(l) );
        m.addConstr( (theta[u] - theta[v] >= -params['delta_max']), name='delta_min[%s]' %(l) );
        m.addConstr( (f[l] + b_out[l]*(theta[u] - theta[v]) == 0), name='flow[%s]' %(l) );
    
    
    # Add nodal balance constraint
    
    for i in G.nodes_iter():
        if (i in alpha_g) and (i in alpha_d):
            m.addConstr(
                    (  (alpha_g[i]*Pg_out[i] - alpha_d[i]*Pd_out[i])/100. + \
                        sum(f[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - \
                        sum(f[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0), name="balance[%s]" %(i)
                    );
        elif i in alpha_g:
            m.addConstr(
                    (  (alpha_g[i]*Pg_out[i])/100. + \
                        sum(f[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - \
                        sum(f[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0), name="balance[%s]" %(i)
                    );
        elif i in alpha_d:
            m.addConstr(
                    (  -(alpha_d[i]*Pd_out[i])/100. + \
                        sum(f[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - \
                        sum(f[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0), name="balance[%s]" %(i)
                    );
        else:
            m.addConstr(
                     (sum(f[l['id']] for _,_,l in G.in_edges_iter([i],data='id')) - \
                      sum(f[l] for _,_,l in G.out_edges_iter([i],data='id')) == 0), name="balance[%s]" %(i)
                     );
    
    
    # Add quadratic objective keeping $\alpha_g$ and $\alpha_d$ as close to 1 as possible. 
    obj = gb.QuadExpr()
    for i in alpha_g:
    #     obj += (Pg0[i]/Pgmax)*(alpha_g[i]- 1)*(alpha_g[i] - 1)
        obj += (alpha_g[i]- 1)*(alpha_g[i] - 1)
    for i in alpha_d:
    #     obj += (Pd0[i]/Pdmax)*(alpha_d[i]- 1)*(alpha_d[i] - 1)
        obj += (alpha_d[i]- 1)*(alpha_d[i] - 1)
    
    m.setObjective(obj,gb.GRB.MINIMIZE);
    
    
    # Solve
    m.optimize()
    
    # Form new load and generation vectors
    Pgnew = np.zeros(Pg_out.shape)
    for i in range(Pgnew.shape[0]):
        try:
            Pgnew[i] = Pg_out[i]*alpha_g[i].X
        except KeyError:
            Pgnew[i] = Pg_out[i]
    Pdnew = np.zeros(Pd_out.shape)
    for i in range(Pdnew.shape[0]):
        try:
            Pdnew[i] = Pd_out[i]*alpha_d[i].X
        except KeyError:
            Pdnew[i] = Pd_out[i]
   
    return Pgnew, Pdnew

#def intertie_suceptance_assign(invars):
#    m = gb.Model()
#    m.setParam('LogFile','/tmp/GurobiZone.log')
#    m.setParam('LogToConsole',0)
#    m.setParam('MIPGap',0.15)
#    #m.setParam('SolutionLimit',5) #stop after this many solutions are found
#    #m.setParam('TimeLimit', 500)
#    m.setParam('MIPFocus',1)
#    m.setParam('Threads',60)
#
#    ### data #####
#    balance_epsilon = invars['balance_epsilon']
#    G = invars['G']
#    p = invars['p']
#    b = invars['b']
#    f_max = invars['f_max']
#    delta_max = invars['delta_max']
#    M = invars['M']
#    edge_boundary = invars['edge_boundary']
#    node_num = G.number_of_nodes()
#    branch_num = G.number_of_edges()
#    
#    ###### Variables ##########
#    Z     = m.addVars(edge_boundary,edge_boundary,vtype=gb.GRB.BINARY,name="Z")
#    theta = m.addVars(range(node_num),lb=-3.14,ub=3.14,name="theta")
#    s     = m.addVars(range(branch_num),name="s")
#    #f     = m.addVars(range(branch_num),lb=-f_max,ub=f_max,name="f")
#    f     = m.addVars(range(branch_num), 
#            lb=[-f_max[0] if l not in edge_boundary else -f_max[1] for l in range(branch_num)],
#            ub=[ f_max[0] if l not in edge_boundary else  f_max[1] for l in range(branch_num)],
#            name = "f")
#
#    ###### line flow constraints ######## 
#    for u,v,l in G.edges_iter(data='id'):
#        m.addConstr( ( theta[u] - theta[v] <=  delta_max),'delta_max[%s]' %(l) )
#        m.addConstr( ( theta[u] - theta[v] >= -delta_max),'delta_min[%s]' %(l) )
#        if l in edge_boundary:
#            for l_tilde in edge_boundary:
#                m.addConstr(f[l] + b[l_tilde]*(theta[u] - theta[v]) + M*(1-Z[l,l_tilde]) >= 0 )
#                m.addConstr(f[l] + b[l_tilde]*(theta[u] - theta[v]) - M*(1-Z[l,l_tilde]) <= 0 )
#        else:
#            m.addConstr(f[l] + b[l]*(theta[u] - theta[v]) == 0)
#
#    ###### Node Balance constraints #####
#    m.addConstrs(
#            ( p[n] + sum(f[l['id']] for _,_,l in G.in_edges_iter([n],data='id')) - \
#                sum(f[l] for _,_,l in G.out_edges_iter([n],data='id')) <= balance_epsilon for n in range(node_num)),"balance1"
#            )
#    m.addConstrs(
#            ( p[n] + sum(f[l['id']] for _,_,l in G.in_edges_iter([n],data='id')) - \
#                sum(f[l] for _,_,l in G.out_edges_iter([n],data='id')) >= -balance_epsilon for n in range(node_num)),"balance1"
#            )
#    ######## permutation matrix constraints #############
#    m.addConstrs( (Z.sum(l,'*')  == 1 for l in edge_boundary),"Z_rows")
#    m.addConstrs( (Z.sum('*',l)  == 1 for l in edge_boundary),"Z_cols")
#    for u,v,l in G.edges_iter(data='id'):
#        m.addConstr( (s[l] + f[l] >= 0 ),"slack_1[%s]" %(l))
#        m.addConstr( (s[l] - f[l] >= 0 ),"slack_2[%s]" %(l))
#
#    ####### Objective ############
#    m.setObjective(s.sum('*'),gb.GRB.MINIMIZE) 
#
#    m.optimize()
#
#    if m.solcount > 0:
#        print('max f: %0.3g' %(max([f[i].X for i in f]) ) )
#        print('max delta: %0.3g' %(max([abs(theta[u].X - theta[v].X) for u,v in G.edges_iter()]) ) )
#        bnew = b.copy()
#        for l,l_tilde in itertools.product(edge_boundary,edge_boundary):
#            if Z[l,l_tilde].X > 0.5:
#                bnew[l] = b[l_tilde]
#        return bnew
#    else:
#        logging.info('Solver exited with status %d and no solution', m.status)
#        sys.exit(0)
