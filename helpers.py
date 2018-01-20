import random
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from scipy import stats, optimize, interpolate

def var2mat(var,M,perm=None):
    """M is size as a tuple
    returns a dense numpy array of size M """
    out = np.zeros(M)
    if perm is None:
        for key in var:
            out[key] = var[key].X
    else:
        for i,j in perm.keys():
            if perm[i,j].X > 0.5:
                out[i] = var[j]
    return out

def progress(i, T, res=0.1):
    if np.floor((i-1)/T/res) != np.floor(i/T/res):
        return np.floor(i/T/res)*res

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

def injection_sample(N,frac=None,gen_params=None,load_params=None, sysLoad=None):
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
    Pg, Pd = injection_equalize_optimization(Pg0, Pd0, gen_params, load_params, sysLoad=sysLoad)
    return Pg, Pd, Pg0, Pd0

def injection_equalize_optimization(Pg0,Pd0,gen_params,load_params,sysLoad=None):

    pd_ls_pg = np.where(((Pg0-Pd0) > 0) & (Pd0 > 0) & (Pg0 > 0))[0]
    pd_gr_pg = np.where(((Pg0-Pd0) < 0) & (Pd0 > 0) & (Pg0 > 0))[0]
    Pg_map = np.where(Pg0>0)[0]
    Pd_map = np.where(Pd0>0)[0]
    Pgmax = max(Pg0)
    Pdmax = max(Pd0)
    import gurobipy as gb
    m = gb.Model()

    alpha_g = m.addVars(Pg_map,lb=0)
    alpha_d = m.addVars(Pd_map,lb=0)
    
    m.addConstr(sum(alpha_g[i]*Pg0[i] for i in Pg_map) - sum(alpha_d[i]*Pd0[i] for i in Pd_map) == 0)
    m.addConstrs(alpha_g[i]*Pg0[i] >= gen_params['vmin']  for i in Pg_map)
    m.addConstrs(alpha_g[i]*Pg0[i] <= gen_params['vmax']  for i in Pg_map)
    m.addConstrs(alpha_d[i]*Pd0[i] >= load_params['vmin'] for i in Pd_map)
    m.addConstrs(alpha_d[i]*Pd0[i] <= load_params['vmax'] for i in Pd_map)
    m.addConstrs(alpha_d[i]*Pd0[i] <= alpha_g[i]*Pg0[i]   for i in pd_ls_pg)
    m.addConstrs(alpha_d[i]*Pd0[i] >= alpha_g[i]*Pg0[i]   for i in pd_gr_pg)
    if sysLoad is not None:
        m.addConstr(sum(alpha_g[i]*Pg0[i] for i in Pg_map) == sysLoad)
    
    obj = gb.QuadExpr()
    for i in alpha_g:
        obj += (Pg0[i]/Pgmax)*(alpha_g[i]- 1)*(alpha_g[i] - 1)
    for i in alpha_d:
        obj += (Pd0[i]/Pdmax)*(alpha_d[i]- 1)*(alpha_d[i] - 1)
    
    m.setObjective(obj,gb.GRB.MINIMIZE)
    m.optimize()

    Pgnew = np.zeros(Pg0.shape)
    for i in range(Pgnew.shape[0]):
        try:
            Pgnew[i] = Pg0[i]*alpha_g[i].X
        except KeyError:
            Pgnew[i] = Pg0[i]
    Pdnew = np.zeros(Pd0.shape)
    for i in range(Pdnew.shape[0]):
        try:
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

def powervectors_from_iteration_dump(input_name,dump_name):
    """ restore powervectors from an iteration dump file """

    Pg,Pd,x,Pg0,Pd0 = pickle.load(open(input_name, 'rb'))
    beta,beta_bar,beta_diff,wdump,nu_map,pdump,bdump = pickle.load(open(dump_name,'rb'))
    p_out = {}
    for i in pdump.keys():
        p_out.update(pdump[i])
    Pg_out = np.array([Pg[p_out[i]] for i in range(len(p_out))])
    Pd_out = np.array([Pd[p_out[i]] for i in range(len(p_out))])
    return Pg_out, Pd_out
   
def parallel_map(Gin):
    G = nx.MultiGraph(Gin) #convert to non-directed
    pmap = {}
    for u,v in G.edges_iter():
        if len(G.edge[u][v]) > 1:
            ids = sorted([val['id'] for k,val in G.edge[u][v].items()])
            for e in range(len(ids)-1):
                pmap[ids[e]] = ids[e+1]
    return pmap

def multivar_kde_sample(kde,actual_vars=False,cols=None):
    if actual_vars:
        if cols is None:
            I = np.random.randint(0,kde.n)
        else:
            I = cols[np.random.randint(0,len(cols))]
        return kde.dataset[:,I]
    else:
        return kde.resample(1)

def multivar_power_sample(N,resd,resg,resf):
    """ numbers """
    Nintermediate = int(round(resf['intermediate']*N))
    Ngen          = int(round(resf['gen']*N))
    Ngenonly      = int(round(resf['gen_only']*N))
    Nloadonly     = N - (Ngen + Nintermediate)
    NQdPd        = int(round(resf['Qd_Pd']*N))
    NQgPg        = int(round(resf['Qg_Pg']*Ngen))

    while True:
        x = {k:np.empty(N) for k in ['Pgmax','Pgmin','Qgmax','Pd','Qd']}
        x['actual_vars_d'] = resd['actual_vars']
        x['actual_vars_g'] = resg['actual_vars']
        genlist  = ['Pgmax','Pgmin','Qgmax']
        loadlist = ['Pd','Qd']
        gensampdict = {resg['order'][j]:i for i,j in enumerate(resg['inkde'])}

        QdPdcnt = 0
        QgPgcnt = 0
        ptr = 0
        """ intermediate buses """
        for i in range(Nintermediate):
            for k in (genlist + loadlist):
                x[k][ptr] = 0
            ptr += 1

        """ load only buses """
        for i in range(Nloadonly):
            #s = resd['kde'].resample(1)
            s = multivar_kde_sample(resd['kde'],actual_vars=resd['actual_vars'])
            while np.any(s > 1.1*resd['max']) | np.any(s < resd['min']) | ((QdPdcnt >= NQdPd) and (s[1] > s[0])):
                #s = resd['kde'].resample(1)
                s = multivar_kde_sample(resd['kde'],actual_vars=resd['actual_vars'])
            x['Pd'][ptr] = s[0]
            x['Qd'][ptr] = s[1]
            if s[1] > s[0]:
                QdPdcnt += 1
            for k in genlist:
                x[k][ptr] = 0
            ptr += 1

        """ gen and load buses """
        for i in range(Ngen-Ngenonly):
            #load part
            #s = resd['kde'].resample(1)
            s = multivar_kde_sample(resd['kde'],actual_vars=resd['actual_vars'])
            while np.any(s > 1.1*resd['max']) | np.any(s < resd['min']) | ((QdPdcnt >= NQdPd) and (s[1] > s[0])):
                #s = resd['kde'].resample(1)
                s = multivar_kde_sample(resd['kde'],actual_vars=resd['actual_vars'])
            x['Pd'][ptr] = s[0]
            x['Qd'][ptr] = s[1]
            if s[1] > s[0]:
                QdPdcnt += 1
            # gen part
            #s = resg['kde'].resample(1)
            s = multivar_kde_sample(resg['kde'],actual_vars=resg['actual_vars'])
            while np.any(s > 1.1*resg['max']) | np.any(s < 0.9*resg['min']) | ((QgPgcnt >= NQgPg) and (s[gensampdict['Qgmax']] > s[gensampdict['Pgmax']])):
                #s = resg['kde'].resample(1)
                s = multivar_kde_sample(resg['kde'],actual_vars=resg['actual_vars'])
            if s[gensampdict['Qgmax']] > s[gensampdict['Pgmax']]:
                QgPgcnt += 1
            for k,v in gensampdict.items():
                x[k][ptr] = s[v]
            #for j,k in enumerate(resg['inkde']):
            #    x[resg['order'][k]][ptr] = s[j]
            for k,v in resg['vdefault'].items():
                x[k][ptr] = v
            ptr += 1

        """ gen only buses """
        for i in range(Ngenonly):
            #s = resg['kde'].resample(1)
            s = multivar_kde_sample(resg['kde'],actual_vars=resg['actual_vars'])
            while np.any(s > 1.1*resg['max']) | np.any(s < 0.9*resg['min']) | ((QgPgcnt >= NQgPg) and (s[gensampdict['Qgmax']] > s[gensampdict['Pgmax']])):
                #s = resg['kde'].resample(1)
                s = multivar_kde_sample(resg['kde'],actual_vars=resg['actual_vars'])
            if s[gensampdict['Qgmax']] > s[gensampdict['Pgmax']]:
                QgPgcnt += 1
            for k,v in gensampdict.items():
                x[k][ptr] = s[v]
            for k,v in resg['vdefault'].items():
                x[k][ptr] = v

            for k in loadlist:
                x[k][ptr] = 0
            ptr += 1

        """ rescale generator maximum values """
        if not resg['actual_vars']:
            #pavg = sum(x['Pgmax'])/Ngen
            #qavg = sum(x['Qgmax'])/Ngen
            
            pmax = 1.1*resg['max'][gensampdict['Pgmax']]
            pmin = 0.9*resg['min'][gensampdict['Pgmax']]
            qmax = 1.1*resg['max'][gensampdict['Qgmax']]
            qmin = 0.9*resg['min'][gensampdict['Qgmax']]

            flag,x['Pgmax'],x['Qgmax'],x['Pgmin'] = maxval_rescale(x['Pgmax'],x['Qgmax'],x['Pgmin'],resf['PgAvg'],resf['QgAvg'],pmax,pmin,qmax,qmin,Ngen)
 
            #x['Pgmax'] = x['Pgmax']*(resf['PgAvg']/pavg)
            #x['Qgmax'] = x['Qgmax']*(resf['QgAvg']/qavg)
            if flag:
                x['shunt'] = resd['shunt']
                break
        else:
            x['shunt'] = resd['shunt']
            break
    return x

def maxval_rescale(P,Q,Pmin,pavg,qavg,pmax,pmin,qmax,qmin,G):

    import gurobipy as gb
    m = gb.Model()
    m.setParam('LogToConsole',0)
    
    Gmap = np.where(P>0)[0]
    PgQ  = np.where((P>0) & (P > Q))[0]
    PlQ  = np.where((P>0) & (P < Q))[0]
    Pmax = max(P)
    Qmax = max(Q)

    wp = m.addVars(Gmap,lb=0)
    #wq = m.addVars(Gmap,lb=0)
    
    m.addConstr(sum(wp[i]*P[i] for i in Gmap) == pavg*G)
    #m.addConstr(sum(wq[i]*Q[i] for i in Gmap) == qavg*G)

    m.addConstrs(wp[i]*P[i] <= pmax for i in Gmap)
    m.addConstrs(wp[i]*P[i] >= pmin for i in Gmap)
    #m.addConstrs(wq[i]*Q[i] <= qmax for i in Gmap)
    #m.addConstrs(wq[i]*Q[i] >= qmin for i in Gmap)

    #m.addConstrs(wp[i]*P[i] >= wq[i]*Q[i] for i in PgQ)
    #m.addConstrs(wp[i]*P[i] <= wq[i]*Q[i] for i in PlQ)

    obj = gb.QuadExpr()
    for i in wp:
        obj += (P[i]/Pmax)**2*(wp[i]- 1)*(wp[i] - 1)
    #for i in wq:
    #    obj += (Q[i]/Qmax)*(wq[i]- 1)*(wq[i] - 1)

    m.setObjective(obj,gb.GRB.MINIMIZE)
    m.optimize()
    flag = m.status == 2
    if flag:
        for i in Gmap:
            P[i] = wp[i].X*P[i]
            Q[i] = wp[i].X*Q[i]
            Pmin[i] = wp[i].X*Pmin[i]
    return flag,P,Q,Pmin

def multivar_z_sample(M,resz):
    while True:
        x = {k:np.empty(M) for i,k in resz['order'].items()}
        x['actual_vars'] = resz['actual_vars']
        zsampdict = {resz['order'][j]:i for i,j in enumerate(resz['inkde'])}
        NRgX = int(round(resz['RgX']*M))
        NBgX = int(round(resz['BgX']*M))
        if resz['actual_vars']:
            B0 = 0 # no special zero b sampling if actual samples used
        else:
            B0   = int(round(resz['b0']*M))

        RgXcnt = 0
        BgXcnt = 0
        """ Zero Susceptance Branches """
        if resz['actual_vars']:
            cols = np.where(resz['kde'].dataset[zsampdict['b'],:] == 0)[0]
        else:
            cols = None
        for i in range(B0):
            #s = resz['kde'].resample(1)
            s = multivar_kde_sample(resz['kde'],actual_vars=resz['actual_vars'],cols=cols)
            while np.any(s > resz['max']) | np.any(s < resz['min']) | ((RgXcnt >= NRgX) and (s[zsampdict['r']] > s[zsampdict['x']])):
                #s = resz['kde'].resample(1)
                s = multivar_kde_sample(resz['kde'],actual_vars=resz['actual_vars'])
            if s[zsampdict['r']] > s[zsampdict['x']]:
                RgXcnt += 1
            for k,v in zsampdict.items():
                x[k][i] = s[v]
            for k,v in resz['vdefault'].items():
                x[k][i] = v
            x['b'][i] = 0
            

        """ Rest of Branches """
        for i in range(B0,M):
            #s = resz['kde'].resample(1)
            s = multivar_kde_sample(resz['kde'],actual_vars=resz['actual_vars'])
            while np.any(s > resz['max']) | np.any(s < resz['min']) | ((RgXcnt >= NRgX) and (s[zsampdict['r']] > s[zsampdict['x']])) | ((BgXcnt >= NBgX) and (s[zsampdict['b']] > s[zsampdict['x']])):
                #s = resz['kde'].resample(1)
                s = multivar_kde_sample(resz['kde'],actual_vars=resz['actual_vars'])
            if s[zsampdict['r']] > s[zsampdict['x']]:
                RgXcnt += 1
            if s[zsampdict['b']] > s[zsampdict['x']]:
                BgXcnt += 1
            for k,v in zsampdict.items():
                x[k][i] = s[v]
            for k,v in resz['vdefault'].items():
                x[k][i] = v
        
        if not resz['actual_vars']:
            if 'b' in zsampdict:
                bmax = resz['max'][zsampdict['b']]
                bmin = resz['min'][zsampdict['b']]
            else:
                bmax = 0; bmin = 0
            flag, x['r'],x['x'],x['b'] = z_rescale(x['r'],x['x'],x['b'],resz['xmean'],resz['bmean'],resz['max'][zsampdict['x']], resz['min'][zsampdict['x']],bmax,bmin,M)
            if flag:
                break
        else:
            break
    return x

def z_rescale(r,x,b,xmean,bmean,xmax,xmin,bmax,bmin,M):
    import gurobipy as gb
    m = gb.Model()
    m.setParam('LogToConsole',0)
    
    Xmax = max(x)
    Bmin = min(b[b>0])
    Bmax = max(b)
    xmap = np.where(x !=0 )[0]
    bmap = np.where(b !=0 )[0]
    BlX  = np.where( b < x)[0]
    wx = m.addVars(xmap,lb=0)
    wb = m.addVars(xmap,lb=0)

    m.addConstr( sum(wx[i]*x[i] for i in xmap) == xmean*M)
    m.addConstr( sum(wb[i]*b[i] for i in xmap) == bmean*M)
    m.addConstrs(wx[i]*x[i] <= xmax for i in xmap)
    m.addConstrs(wx[i]*x[i] >= xmin for i in xmap)
    m.addConstrs(wb[i]*b[i] <= bmax for i in xmap)
    m.addConstrs(wb[i]*b[i] >= bmin for i in xmap)
    m.addConstrs(wb[i]*b[i] >= Bmin for i in bmap)
    m.addConstrs(wb[i] <= wx[i] for i in BlX)

    obj = gb.QuadExpr()
    for i in xmap:
        obj += (Xmax/x[i])*(wx[i] - 1)*(wx[i] - 1)
        obj += (Bmax/max(b[i],Bmin))*(wb[i] - 1)*(wb[i] - 1)

    m.setObjective(obj,gb.GRB.MINIMIZE)
    m.optimize()
    flag = m.status == 2
    if flag:
        for i in range(M):
            r[i] = wx[i].X*r[i]
            x[i] = wx[i].X*x[i]
            b[i] = wb[i].X*b[i]
    return flag,r,x,b

def Yparts(r,x,b=None,tau=None,phi=None):
    """ Form vectors corresponding to the primitive admittance matricesi
    yff = (ys + j*b/2)/tau^2
    yft = -ys*exp(j*phi)/tau = -(gs + j*bs)*[cos(phi) + j*sin(phi)]/tau
    ytf = -ys*exp(-jphi)/tau = -(gs + j*bs)*[cos(phi) - j*sin(phi)]/tau
    ytt = ys + j*b/2
    """

    if b is None:
        b = np.zeros(len(x))
    if tau is None:
        tau = np.ones(len(x))
    if phi is None:
        phi = np.zeros(len(x))

    gs =  r/(r**2 + x**2)
    bs = -x/(r**2 + x**2)
    
    gff = gs/(tau**2); 
    gft = (-gs*np.cos(phi) + bs*np.sin(phi))/tau
    gtf = (-gs*np.cos(phi) - bs*np.sin(phi))/tau
    gtt = gs

    bff = (bs + b/2)/(tau**2)
    bft = (-gs*np.sin(phi) - bs*np.cos(phi))/tau
    btf = ( gs*np.sin(phi) - bs*np.cos(phi))/tau
    btt = bs + b/2

    return {'gff': gff, 'gft': gft, 'gtf':gtf, 'gtt':gtt, 'bff': bff, 'bft': bft, 'btf':btf, 'btt':btt}
    
def bigM_calc(Y,fmax,umax,umin,dmax,margin=1.1):
    #bigM = {}
    #bigM['pf']  = margin*(fmax + max(Y['gff'] + Y['gft'])*(1+umax) + max(np.abs(Y['bft']))*dmax - min(Y['gft'])*0.5*dmax**2)
    #bigM['qf']  = margin*(fmax + max(Y['bff'] + Y['bft'])*(1+umax) + max(np.abs(Y['gft']))*dmax - min(Y['bft'])*0.5*dmax**2)
    #bigM['pt']  = margin*(fmax + max(Y['gtt'] + Y['gtf'])*(1+umax) + max(np.abs(Y['btf']))*dmax - min(Y['gtf'])*0.5*dmax**2)
    #bigM['qt']  = margin*(fmax + max(Y['btt'] + Y['btf'])*(1+umax) + max(np.abs(Y['gtf']))*dmax - min(Y['btf'])*0.5*dmax**2)

    xmax = {}; xmin = {}
    xmax['pf'] = max(Y['gff'] + Y['gft'])
    xmin['pf'] = min(Y['gff'] + Y['gft'])
    xmax['pt'] = max(Y['gtt'] + Y['gtf'])
    xmin['pt'] = min(Y['gtt'] + Y['gtf'])
    xmax['qf'] = max(Y['bff'] + Y['bft'])
    xmin['qf'] = min(Y['bff'] + Y['bft'])
    xmax['qt'] = max(Y['btt'] + Y['btf'])
    xmin['qt'] = min(Y['btt'] + Y['btf'])

    Mpf1 = fmax + max(xmax['pf']*(1+umax),xmax['pf']*(1+umin)) - min(min(Y['gft'])*0.5*dmax*dmax,0) + max(np.abs(Y['bft']))*dmax
    Mpf2 = fmax - min(xmin['pf']*(1+umax),xmin['pf']*(1+umin)) + max(max(Y['gft'])*0.5*dmax*dmax,0) + max(np.abs(Y['bft']))*dmax
    Mpt1 = fmax + max(xmax['pt']*(1+umax),xmax['pt']*(1+umin)) - min(min(Y['gtf'])*0.5*dmax*dmax,0) + max(np.abs(Y['btf']))*dmax
    Mpt2 = fmax - min(xmin['pt']*(1+umax),xmin['pt']*(1+umin)) + max(max(Y['gtf'])*0.5*dmax*dmax,0) + max(np.abs(Y['btf']))*dmax

    Mqf1 = fmax - min(xmin['qf']*(1+umax),xmin['qf']*(1+umin)) + max(max(Y['bft'])*0.5*dmax*dmax,0) + max(np.abs(Y['gft']))*dmax
    Mqf2 = fmax + max(xmax['qf']*(1+umax),xmax['qf']*(1+umin)) - min(min(Y['bft'])*0.5*dmax*dmax,0) + max(np.abs(Y['gft']))*dmax
    Mqt1 = fmax - min(xmin['qt']*(1+umax),xmin['qt']*(1+umin)) + max(max(Y['btf'])*0.5*dmax*dmax,0) + max(np.abs(Y['gtf']))*dmax
    Mqt2 = fmax + max(xmax['qt']*(1+umax),xmax['qt']*(1+umin)) - min(min(Y['btf'])*0.5*dmax*dmax,0) + max(np.abs(Y['gtf']))*dmax

    bigM = {}
    bigM['pf']  = margin*max(Mpf1,Mpf2)
    bigM['pt']  = margin*max(Mpt1,Mpt2)
    bigM['qf']  = margin*max(Mqf1,Mqf2)
    bigM['qt']  = margin*max(Mqt1,Mqt2)
    return bigM

def def_consts(**kwargs):
    c    = {}
    c['fmax'] = 9            # default per unit maximum real power flow on line
    c['dmax'] = 40*np.pi/180 # angle difference limit over a branch
    #c['htheta'] = 7          # number of segments for approximating (theta_f - theta_t)^2/2
    c['phi_err'] = 1e-3      # desired maximum error for polyedral relaxation of (theta_f-theta_t)^2/2
    c['umin'] = np.log(0.9)  # minimum ln(voltage)
    c['umax'] = np.log(1.05) # maximum ln(voltage)
    c['lossmin'] = 0.01      # minimum losses required (fraction = (Pg - Pd)/Pg)
    c['lossterm']= 0.05      # terminate optimization when if losses are at this level or below
    c['thresholds'] = {'gap':       5.,
                  'mean_diff': 0.05,
                  'max_diff':  0.1,
                  'itermax':   5}
    c['rho'] = 1
    if kwargs is not None:
        for k,v in kwargs.items():
            c[k] = v
    return c

def update_consts(c,fin):
    truelist = [True,'True','true','t','1']
    for k, v in c.items():
        if type(v) is dict:
            update_consts(v,fin)
        else:
            if k in fin:
                if type(v) is bool:
                    c[k] = fin[k] in truelist
                elif v is None:
                    c[k] = none_test(fin[k])
                else:
                    c[k] = type(v)(fin[k])

def none_test(x):
    if x in ['None', 'none']:
        return None
    else:
        return x

def pick_ang0_node(G):
    """ select a node that will be given angle 0 
    This is done by selecting the most central node using betweeness centrality
    """
    v = nx.betweenness_centrality(nx.Graph(G)) 
    return max(v, key=v.get)

def edge_spread_from_v(G,v):
    """ return the eccentricity of node v which is the maximum distance from v to all other nodes in G """
    return nx.eccentricity(nx.Graph(G), v)

def theta_max(G,v,dmax=np.pi/2, margin=2*np.pi):
    return edge_spread_from_v(G,v)*dmax + margin

def polyhedral_h(delta_max,epsilon):
    """ return parameter h for creating polyhedral constraints to quadratic terms """
    return np.ceil( delta_max/np.sqrt(epsilon) ) 

def savepath_replace(savename, newpath):
    parts = savename.split('/')
    if newpath[-1] != '/':
        newpath += '/'
    return newpath + parts[-1]

def testing(*args):
    import fit_inputs as ftin 
    fname = args[0]
    N     = int(args[1])
    M     = int(args[2])
    bus_data, gen_data, branch_data = load_data(fname)
    resz,fmax = ftin.multivariate_z(branch_data,bw_method=0.01)
    z    = multivar_z_sample(M,resz)
    dfz  = pd.DataFrame(z)
    import ipdb; ipdb.set_trace()
    res = ftin.multivariate_power(bus_data,gen_data)
    x = multivar_power_sample(N,*res)
    df = pd.DataFrame({k:v for k,v in x.items() if k!='shunt'})
    import ipdb; ipdb.set_trace()
    

if __name__ == '__main__':
   import sys
   testing(*sys.argv[1:])
