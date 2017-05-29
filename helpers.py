import random
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from scipy import stats, optimize, interpolate

def var2mat(var,M):
    """M is size as a tuple
    returns a dense numpy array of size M """
    out = np.zeros(M)
    for key in var:
        out[key] = var[key].X
    return out

def model_status(m):
    status_codes = {1:'Loaded', 2:'Optimal',3:'Infeasible',4:'Inf_OR_UNBD',5:'Unbounded',6:'Cutoff',
            7:'Iteration_limit',8:'Node_limit',9:'Time_limit',10:'Solution_limit',
            11:'Interrupted',12:'Numeric',13:'Suboptimal',14:'Inprogress',15:'User_obj_limit'}
    status = status_codes[m.getAttr("Status")]
    return status


def get_permutation(m,var_name=None,dim=None):
    p = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            if m.getVarByName(var_name + '[%d,%d]' %(i,j)).X == 1:
                p[i] = j
    return p.astype(int)

def get_imbalance(m,boundary,node_mapping,imbalance=None):
    if imbalance is None:
        imbalance = {}
    for i in boundary:
        # either beta_plus or beta_minus must be zero.
        beta_plus = m.getVarByName("beta_plus[%d]" %(node_mapping[i])).X
        beta_minus = m.getVarByName("beta_minus[%d]" %(node_mapping[i])).X
        imbalance[i] = {'bp':beta_plus, 'bm':beta_minus}
    return imbalance

def get_var_list(m,var_name,length):
    return [m.getVarByName(var_name +'[%d]' %(i)).X for i in range(length)]

def load_data(fname):
    ######### load data ##############
    bus_data = pd.read_csv(fname + '_bus.csv',header=0);
    gen_data = pd.read_csv(fname + '_gen.csv',header=0);
    branch_data = pd.read_csv(fname + '_branch.csv',header=0);
    
    ####### Convert to zero indexing ############
    bus_data['BUS_I'] -= 1
    gen_data['GEN_BUS'] -= 1
    branch_data['F_BUS'] -= 1
    branch_data['T_BUS'] -= 1
    return (bus_data,gen_data,branch_data)

def degree_one(f,t):
    a = np.sort(np.concatenate([f,t]))
    out = []
    for i,v in enumerate(a):
        if (i > 0) and (i < len(a)):
            if (a[i-1] != v) and (a[i+1] != v):
                out.append(v)
        elif i == 0:
            if a[i + 1] != v:
                out.append(v)
        else:
            if a[i-1] != v:
                out.append(v)
    return out

def incident_lines_map(fn,tn,node_num):
    from_lines = {key: [] for key in range(node_num)}
    to_lines = {key: [] for key in range(node_num)}
    for l in fn:
        from_lines[fn[l]] += [l]
        to_lines[tn[l]] += [l]
    return from_lines,to_lines


def power_injections(gen_data,bus_data,equalize=True):
    """ create the power injection vector defined as Pg - Pd
    IT IS ASSUMED THAT THE BUSES ARE ORDERD CONSECUTIVELY!!!
    """
    if np.any(np.diff(bus_data['BUS_I']) != 1):
        print('Buses are not properly ordered')
        assert False

    Pg = np.zeros(bus_data.shape[0])
    for bus,v in zip(gen_data['GEN_BUS'],gen_data['PG']):
        Pg[bus] += v
    if equalize:
        # ensure that generation = loss (i.e. remove loss)
        gen_buses = np.where(Pg)[0]
        loss_flag = False
        while not loss_flag:
            loss_flag = True
            losses = sum(Pg) - bus_data['PD'].sum()
            loss_per_g = losses/gen_buses.shape[0]
            for i in gen_buses:
                if Pg[i] > loss_per_g:
                    Pg[i] -= loss_per_g
                else:
                    loss_flag = False

    return Pg,bus_data['PD'].values

def load_sample(N,vmax=np.inf,vmin=-np.inf,dist='lognorm',params=None):
    if dist == 'lognorm':
        out = np.zeros(N)
        for i in range(out.shape[0]):
            samp = None
            while (samp is None) or (samp < vmin) or (samp > vmax):
                samp = np.exp(params[0] + params[1]*stats.norm.rvs())
            out[i] = samp
    elif (dist == 'kde') or (dist == 'pchip'):
        out = params.resample(size=N).squeeze()
        while np.any(out < vmin) or np.any(out > vmax):
            ids = np.where((out < vmin) | (out > vmax))[0]
            out = np.delete(out,ids)
            tmp = params.resample(size=ids.shape[0]).squeeze()
            if ids.shape[0] == 1:
                tmp = np.array([tmp])
            out = np.concatenate((out,tmp))
    else:
        print('Only log-normal, kde, and pchip distributions supported currently')
    
    if N == 1:
        return np.array([out])
    else:
        return out

def gen_sample(N,vmax=np.inf,vmin=-np.inf,dist='exp',params=None):
    if dist == 'exp':
        out = np.zeros(N)
        for i in range(out.shape[0]):
            samp = None
            while (samp is None) or (samp < vmin) or (samp > vmax):
                samp = stats.expon.rvs(loc=0,scale=params)
            out[i] = samp
    elif (dist == 'kde') or (dist == 'pchip'):
        out = params.resample(size=N).squeeze()
        while np.any(out < vmin) or np.any(out > vmax):
            ids = np.where((out < vmin) | (out > vmax))[0]
            out = np.delete(out,ids)
            tmp = params.resample(size=ids.shape[0]).squeeze()
            if ids.shape[0] == 1:
                tmp = np.array([tmp])
            out = np.concatenate((out,tmp))
    else:
        print('Only exponential, kde, and pchip distributions supported currently')
    
    if N == 1:
        return np.array([out])
    else:
        return out

def injection_sample(N,frac=None,gen_params=None,load_params=None):
    """ parameter inputs need to include a maximum and minimum vmax,vmin,
    a distribution name and the corresponding parameters for it"""
    #int_frac=0.08,inj_frac=0.13,gen_only_frac=0.5
    Nint = int(np.round(N*frac['intermediate']))
    NPg  = int(np.round(N*frac['Pg']))
    Ngen_only = int(np.round(float(NPg)*frac['gen_only']))
    Npd_ls_pg = int(np.round(float(NPg)*frac['Pd<Pg']))
    Npd_gr_pg = NPg - (Ngen_only + Npd_ls_pg)
    Nload_only= N - (Nint + NPg)
    #Ninj = int(np.round(N*inj_frac))
    #Ngen_only = int(np.round(float(Ninj)*gen_only_frac))
    #Ngen_load = Ninj - Ngen_only
    #Nload = N - (Nint + Ninj)

    load = load_sample(Nload_only,vmax=load_params['vmax'],vmin=load_params['vmin'],\
            dist=load_params['dist'],params=load_params['params'])
    
    gen_only = gen_sample(Ngen_only,vmax=gen_params['vmax'],vmin=gen_params['vmin'],\
            dist=gen_params['dist'],params=gen_params['params'])
    
    gen_load_pos = {}
    gen_load_pos['g'] = np.zeros(Npd_ls_pg)
    gen_load_pos['d'] = np.zeros(Npd_ls_pg)
    for i in range(Npd_ls_pg):
        gtmp = gen_sample(1,vmax=gen_params['vmax'],vmin=gen_params['vmin'],\
            dist=gen_params['dist'],params=gen_params['params'])
        
        dtmp = None
        while (dtmp is None) or (dtmp >= gtmp):
            dtmp = load_sample(1,vmax=load_params['vmax'],vmin=load_params['vmin'],\
                    dist=load_params['dist'],params=load_params['params'])
        gen_load_pos['g'][i] = gtmp
        gen_load_pos['d'][i] = dtmp

    gen_load_neg = {}
    gen_load_neg['g'] = np.zeros(Npd_gr_pg)
    gen_load_neg['d'] = np.zeros(Npd_gr_pg)
    for i in range(Npd_gr_pg):
        dtmp = load_sample(1,vmax=load_params['vmax'],vmin=load_params['vmin'],\
                    dist=load_params['dist'],params=load_params['params'])
        
        gtmp = None
        while (gtmp is None) or (gtmp >= dtmp):
            gtmp = gen_sample(1,vmax=gen_params['vmax'],vmin=gen_params['vmin'],\
                dist=gen_params['dist'],params=gen_params['params'])
        gen_load_neg['g'][i] = gtmp
        gen_load_neg['d'][i] = dtmp

    #Pg = np.concatenate([np.zeros(Nint),np.zeros(Nload),gen_load['g'],gen_only])
    #Pd = np.concatenate([np.zeros(Nint),load,gen_load['d'],np.zeros(Ngen_only)])
    Pg0 = np.concatenate([np.zeros(Nint), np.zeros(Nload_only), gen_load_neg['g'], gen_load_pos['g'], gen_only])
    Pd0 = np.concatenate([np.zeros(Nint), load, gen_load_neg['d'], gen_load_pos['d'], np.zeros(Ngen_only)])
    Pg, Pd = injection_equalize_optimization(Pg0, Pd0, gen_params, load_params)
    return Pg, Pd, Pg0, Pd0

def injection_equalize_optimization(Pg0,Pd0,gen_params,load_params):
    #Pg_non_zero = sum(Pg0>0)
    #Pd_non_zero = sum(Pd0>0)
    #Pg_dict = dict(zip(range(Pg_non_zero),np.where(Pg0>0)[0]))
    #Pd_dict = dict(zip(range(Pd_non_zero),np.where(Pd0>0)[0]))
    pd_ls_pg = np.where(((Pg0-Pd0) > 0) & (Pd0 > 0) & (Pg0 > 0))[0]
    pd_gr_pg = np.where(((Pg0-Pd0) < 0) & (Pd0 > 0) & (Pg0 > 0))[0]
    Pg_map = np.where(Pg0>0)[0]
    Pd_map = np.where(Pd0>0)[0]
    Pgmax = max(Pg0)
    Pdmax = max(Pd0)
    import gurobipy as gb
    m = gb.Model()
    
    #alpha_g = m.addVars(range(Pg_non_zero),lb=0)
    #alpha_d = m.addVars(range(Pd_non_zero),lb=0)
    #
    #m.addConstr(sum(alpha_g[i]*Pg0[Pg_dict[i]] for i in range(Pg_non_zero)) - 
    #              sum(alpha_d[i]*Pd0[Pd_dict[i]] for i in range(Pd_non_zero)) == 0)
    #m.addConstrs(alpha_g[i]*Pg0[Pg_dict[i]] >= gen_params['vmin'] for i in range(Pg_non_zero))
    #m.addConstrs(alpha_g[i]*Pg0[Pg_dict[i]] <= gen_params['vmax'] for i in range(Pg_non_zero))
    #m.addConstrs(alpha_d[i]*Pd0[Pd_dict[i]] >= load_params['vmin'] for i in range(Pd_non_zero))
    #m.addConstrs(alpha_d[i]*Pd0[Pd_dict[i]] <= load_params['vmax'] for i in range(Pd_non_zero))
    #
    #obj = gb.QuadExpr()
    #for i in alpha_g:
    #    obj += (Pg0[Pg_dict[i]]/Pgmax)*(alpha_g[i]- 1)*(alpha_g[i] - 1)
    #for i in alpha_d:
    #    obj += (Pd0[Pd_dict[i]]/Pdmax)*(alpha_d[i]- 1)*(alpha_d[i] - 1)
    #
    #m.setObjective(obj,gb.GRB.MINIMIZE)
    alpha_g = m.addVars(Pg_map,lb=0)
    alpha_d = m.addVars(Pd_map,lb=0)
    
    m.addConstr(sum(alpha_g[i]*Pg0[i] for i in Pg_map) - sum(alpha_d[i]*Pd0[i] for i in Pd_map) == 0)
    m.addConstrs(alpha_g[i]*Pg0[i] >= gen_params['vmin']  for i in Pg_map)
    m.addConstrs(alpha_g[i]*Pg0[i] <= gen_params['vmax']  for i in Pg_map)
    m.addConstrs(alpha_d[i]*Pd0[i] >= load_params['vmin'] for i in Pd_map)
    m.addConstrs(alpha_d[i]*Pd0[i] <= load_params['vmax'] for i in Pd_map)
    m.addConstrs(alpha_d[i]*Pd0[i] <= alpha_g[i]*Pg0[i]   for i in pd_ls_pg)
    m.addConstrs(alpha_d[i]*Pd0[i] >= alpha_g[i]*Pg0[i]   for i in pd_gr_pg)
    
    obj = gb.QuadExpr()
    for i in alpha_g:
        obj += (Pg0[i]/Pgmax)*(alpha_g[i]- 1)*(alpha_g[i] - 1)
    for i in alpha_d:
        obj += (Pd0[i]/Pdmax)*(alpha_d[i]- 1)*(alpha_d[i] - 1)
    
    m.setObjective(obj,gb.GRB.MINIMIZE)
    m.optimize()

    #ag_dict = {v:alpha_g[k].X for k,v in Pg_dict.items()}
    #ad_dict = {v:alpha_d[k].X for k,v in Pd_dict.items()}
    Pgnew = np.zeros(Pg0.shape)
    for i in range(Pgnew.shape[0]):
        try:
            #Pgnew[i] = Pg0[i]*ag_dict[i]
            Pgnew[i] = Pg0[i]*alpha_g[i].X
        except KeyError:
            Pgnew[i] = Pg0[i]
    Pdnew = np.zeros(Pd0.shape)
    for i in range(Pdnew.shape[0]):
        try:
            #Pdnew[i] = Pd0[i]*ad_dict[i]
            Pdnew[i] = Pd0[i]*alpha_d[i].X
        except KeyError:
            Pdnew[i] = Pd0[i]
    return Pgnew, Pdnew

def injection_equalize(Pg,Pd,gen_params,load_params):
    """ adjust generation and load to get sum of 0 total injections """
    gen_bus = np.where(Pg > 0)[0]
    load_bus= np.where(Pd > 0)[0]
    epsilon_out = []
    flag = False
    while not flag:
        flag = True
        epsilon = np.sum(Pg - Pd)
        epsilon_out.append(epsilon)
        epsilon_per_bus = np.abs(epsilon)/(gen_bus.shape[0] + load_bus.shape[0])
        for i in (set(gen_bus) | set(load_bus)):
            if i in load_bus:
                Pdnew = Pd[i] + np.sign(epsilon)*epsilon_per_bus
                if (Pdnew < load_params['vmin']) or (Pdnew > load_params['vmax']):
                    Pdnew = Pd[i]
                    flag = False
            else:
                Pdnew = 0
            if i in gen_bus:
                Pgnew = Pg[i] - np.sign(epsilon)*epsilon_per_bus
                if (Pgnew < gen_params['vmin']) or (Pgnew > gen_params['vmax']):
                    Pgnew = Pg[i]
                    flag = False
            else:
                Pgnew = 0
            if np.sign(Pgnew - Pdnew) == np.sign(Pg[i] - Pd[i]):
                Pg[i] = Pgnew
                Pd[i] = Pdnew
            else:
                flag = False
        epsilon_out.append(np.sum(Pg-Pd))
    return Pg, Pd, epsilon_out   

def injection_equalize_old(Pg,Pd,vmax,vmin):
    """ make sure total injections sums to 0"""
    if np.sum(Pg - Pd) > 0:
        gen_buses = np.where(Pg)[0] # indices of buses with generators
        ## too much generation
        flag = False
        while not flag:
            flag = True
            epsilon = np.sum(Pg - Pd)
            epsilon_per_g = epsilon/gen_buses.shape[0]
            for i in gen_buses:
                if Pg[i] - epsilon_per_g > vmin:
                    Pg[i] -= epsilon_per_g
                else:
                    # loops until error is small enough that every generator participates
                    flag = False
    elif np.sum(Pg - Pd) < 0:
        ## too little generation
        flag = False
        while not flag:
            flag = True
            gen_buses = np.where((Pg > 0) & (Pg < vmax))[0] # indices of buses with generators
            epsilon = np.sum(Pd - Pg)
            epsilon_per_g = epsilon/gen_buses.shape[0]
            for i in gen_buses:
                if Pg[i] + epsilon_per_g < vmax:
                    Pg[i] += epsilon_per_g
                else:
                    # loops until error is small enough that every generator participates
                    flag = False
    return Pg

def get_b_from_dist(M,dist='gamma',params=None,vmin=-np.inf,vmax=np.inf):
    if dist == 'gamma':
        rv = stats.gamma(*params)
        #x = stats.gamma.rvs(*params, size=M)
    elif dist == 'exp':
        rv = stats.expon(*params)
        #x = stats.expon.rvs(*params,size=M)
    elif (dist == 'kde') or (dist == 'pchip'):
        x = params.resample(size=M).squeeze()
        while np.any(x < vmin) or np.any(x > vmax):
            ids = np.where((x < vmin) | (x > vmax))[0]
            x = np.delete(x,ids)
            tmp = params.resample(size=ids.shape[0]).squeeze()
            if ids.shape[0] == 1:
                tmp = np.array([tmp])
            x = np.concatenate((x,tmp))
    else:
        print('Only gamma, exp, and kde  distributions supported currently')
    if dist != 'kde':
        x = np.zeros(M)
        for i in range(M):
            xtmp = rv.rvs(size=1)[0]
            while (xtmp < vmin) or (xtmp > vmax):
                xtmp = rv.rvs(size=1)[0]
            x[i] = xtmp
    return -1./x

class PchipDist(object):
    def __init__(self,data,bins=None):
        if bins is None:
            bins = 'auto'
        h = np.histogram(data, bins=bins, density=True)
        cdf = np.cumsum(h[0]*np.diff(h[1]))
        cdf[-1] = 1.
        self.P = interpolate.PchipInterpolator(h[1],np.concatenate(([0.],cdf)))
        self.mean = np.mean(data)
        self.min = h[1][0]
        self.max = h[1][-1]

    def __call__(self,x):
        return self.P(x)

    def pdf(self,x):
        return self.P.derivative()(x)

    def resample(self,size=1):
        rvs = np.zeros(size)
        for i in range(size):
            Q = np.random.rand()
            rvs[i] = optimize.brentq(lambda x: self.P(x) - Q, self.min, self.max)
        if size == 1:
            rvs = rvs[0]
        return rvs

def zone_power_sample(N, p_in, Mboundary, beta_max):
    max_unbalance = Mboundary*beta_max
    count = 0
    while True:
        ph = {k: p_in[k] for k in random.sample(list(p_in),N)}
        if abs(sum([ph[i] for i in ph])) <= max_unbalance:
            break
        count += 1
        if count > 250:
            logging.info('Cannot create sufficiently balanced zone')
            sys.exit(0)
    return ph
    
