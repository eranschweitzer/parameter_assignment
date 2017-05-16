import pandas as pd
import numpy as np
import networkx as nx
import pickle
from scipy import stats

def var2mat(var,M):
    """M is size as a tuple
    returns a dense numpy array of size M """
    out = np.zeros(M)
    for key in var:
        out[key] = var[key].X

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


def power_injections(gen_data,bus_data):
    """ create the power injection vector defined as Pg - Pd
    IT IS ASSUMED THAT THE BUSES ARE ORDERD CONSECUTIVELY!!!
    """
    if np.any(np.diff(bus_data['BUS_I']) != 1):
        print('Buses are not properly ordered')
        assert False

    Pg = np.zeros(bus_data.shape[0])
    for bus,v in zip(gen_data['GEN_BUS'],gen_data['PG']):
        Pg[bus] += v
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
    else:
        print('Only log-normal distribution supported currently')
    
    if N == 1:
        return out[0]
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
    else:
        print('Only exponential distribution supported currently')
    
    if N == 1:
        return out[0]
    else:
        return out

def injection_sample(N,int_frac=0.08,inj_frac=0.13,gen_only_frac=0.5,gen_params=None,load_params=None):
    """ parameter inputs need to include a maximum and minimum vmax,vmin,
    a distribution name and the corresponding parameters for it"""

    Nint = int(np.round(N*int_frac))
    Ninj = int(np.round(N*inj_frac))
    Ngen_only = int(np.round(float(Ninj)*gen_only_frac))
    Ngen_load = Ninj - Ngen_only
    Nload = N - (Nint + Ninj)

    load = load_sample(Nload,vmax=load_params['vmax'],vmin=load_params['vmin'],\
            dist=load_params['dist'],params=load_params['params'])
    
    gen_only = gen_sample(Ngen_only,vmax=gen_params['vmax'],vmin=gen_params['vmin'],\
            dist=gen_params['dist'],params=gen_params['params'])
    
    gen_load = {}
    gen_load['g'] = np.zeros(Ngen_load)
    gen_load['d'] = np.zeros(Ngen_load)
    for i in range(Ngen_load):
        gtmp = gen_sample(1,vmax=gen_params['vmax'],vmin=gen_params['vmin'],\
            dist=gen_params['dist'],params=gen_params['params'])
        
        dtmp = None
        while (dtmp is None) or (dtmp > gtmp):
            dtmp = load_sample(1,vmax=load_params['vmax'],vmin=load_params['vmin'],\
                    dist=load_params['dist'],params=load_params['params'])
        gen_load['g'][i] = gtmp
        gen_load['d'][i] = dtmp

    Pg = np.concatenate([np.zeros(Nint),np.zeros(Nload),gen_load['g'],gen_only])
    Pd = np.concatenate([np.zeros(Nint),load,gen_load['d'],np.zeros(Ngen_only)])

    Pg = injection_equalize(Pg,Pd,gen_params['vmax'],gen_params['vmin'])
    return Pg,Pd

def injection_equalize(Pg,Pd,vmax,vmin):
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
    x = np.zeros(M)
    if dist == 'gamma':
        rv = stats.gamma(*params)
        #x = stats.gamma.rvs(*params, size=M)
    if dist == 'exp':
        rv = stats.expon(*params)
        #x = stats.expon.rvs(*params,size=M)
    else:
        print('Only gamma distribution supported currently')
    for i in range(M):
        xtmp = rv.rvs(size=1)[0]
        while (xtmp < vmin) or (xtmp > vmax):
            xtmp = rv.rvs(size=1)[0]
        x[i] = xtmp
    return -1./x
