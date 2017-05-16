import pandas as pd
import numpy as np
import networkx as nx
import pickle
from scipy import stats

def model_status(m):
    status_codes = {1:'Loaded', 2:'Optimal',3:'Infeasible',4:'Inf_OR_UNBD',5:'Unbounded',6:'Cutoff',
            7:'Iteration_limit',8:'Node_limit',9:'Time_limit',10:'Solution_limit',
            11:'Interrupted',12:'Numeric',13:'Suboptimal',14:'Inprogress',15:'User_obj_limit'}
    status = status_codes[m.getAttr("Status")]
    return status

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

def injection_sample(N,int_frac=0.08,inj_frac=0.13,gen_params=None,load_params=None):
    """ parameter inputs need to include a maximum and minimum vmax,vmin,
    a distribution name and the corresponding parameters for it"""

    Nint = int(np.round(N*int_frac))
    Ninj = int(np.round(N*inj_frac))
    Ngen_only = int(np.round(float(Ninj)/2.))
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

def incident_lines_map(fn,tn,node_num):
    from_lines = {key: [] for key in range(node_num)}
    to_lines = {key: [] for key in range(node_num)}
    for l in fn:
        from_lines[fn[l]] += [l]
        to_lines[tn[l]] += [l]
    return from_lines,to_lines
    
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

def get_b_from_dist(M,dist='gamma',params=None):
    if dist == 'gamma':
        x = stats.gamma.rvs(*params, size=M)
    else:
        print('Only gamma distribution supported currently')
    return -1./x

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

def main(savename,fdata,mode='real',decomp='zone'):
    """
        modes:
            real: only shuffle injections and impedance
            bsynth: real injections, synthetic impedance
            pbsyth: synthetic injections and impednce, still real topology
            synth: everyting synthetic
        decomp:
            None: full MILP problem
            iter: alternate solving for Pi and Z
    """
    ###### Topological data ###########
    if mode == 'syth': 
        top = pd.read_csv(fdata)
        # change to zero indexing
        top['f'] -= 1
        top['t'] -= 1
        f_node = top['f'].values
        t_node = top['t'].values
        node_num = max(top['f'].max(),top['t'].max()) + 1 #maximum node number (plus one because of zero indexing) 
        branch_num = top.shape[0]
        fn = dict(zip(top.index,top['f']))
        tn = dict(zip(top.index,top['t']))
        deg_one = degree_one(top['f'].values,top['t'].values)
    else:
        bus_data,gen_data,branch_data = load_data(fdata)
        f_node = branch_data['F_BUS'].values
        t_node = branch_data['T_BUS'].values
        node_num = bus_data.shape[0]
        branch_num = branch_data.shape[0]
        fn = dict(zip(branch_data.index,branch_data['F_BUS']))
        tn = dict(zip(branch_data.index,branch_data['T_BUS']))
        deg_one = degree_one(branch_data['F_BUS'].values,branch_data['T_BUS'].values)

    from_lines,to_lines = incident_lines_map(fn,tn,node_num) 
    G = nx.MultiDiGraph()
    G.add_edges_from(zip(f_node,t_node,[{'id':i} for i in range(f_node.shape[0])]))

    ###### power injections #########
    if mode in ['real','bsyhnth']:
        Pg,Pd = power_injections(gen_data,bus_data)
        p = (Pg-Pd)/100 # change to per unit
        p_in = np.random.permutation(p)
    else:
        gen_params = {'vmax':800,'vmin':2,'dist':'exp','params':230.4}
        load_params = {'vmax':275,'vmin':4,'dist':'lognorm','params':(3.4315,0.8363)}
        Pg,Pd = injection_sample(node_num,int_frac=0.08,inj_frac=0.13,gen_params=gen_params,load_params=load_params)
        p_in = (Pg - Pd)/100

    ######## susceptances #########
    if mode == 'real':
        b = -1/branch_data['BR_X'] 
    else:
        b = get_b_from_dist(branch_num,dist='gamma',params=(1.88734, 0, 0.05856)) 

    b_in = np.random.permutation(b)
    
    ####### constant inputs #########
    balance_epsilon = 1e-4
    delta_max = 60.0*np.pi/180.0
    f_max = 10
    M = f_max + delta_max*max(np.abs(b)) + 1 #plus one is out of precaution
    

    ####### optimization ##########
    import formulation as fm
    
    if decomp == 'None':
        #m = fm.full_MILP({'node_num':node_num,'branch_num':branch_num,'p':p_in,'b':b_in,\
        #        'delta_max':delta_max,'f_max':f_max,'M':M,'fn':fn,'tn':tn,\
        #        'from_lines':from_lines,'to_lines':to_lines,'balance_epsilon':balance_epsilon,\
        #        'deg_one':deg_one})
        m = fm.full_MILP({'G':G,'p':p_in,'b':b_in,'M':M,'delta_max':delta_max,'f_max':f_max,'balance_epsilon':balance_epsilon})

        print('%s' %(model_status(m)))
    
        power_perm = get_permutation(m,var_name='Pi',dim=G.number_of_nodes())
        susceptance_perm = get_permutation(m,var_name='Z',dim=G.number_of_edges())

    #elif decomp == 'zone':
    #    import zone_splitting as zp
    #    Nmax = 50
    #    p_in = np.random.permutation(p_in).tolist()
    #    b_in = np.random.permutation(b_in).tolist()
    #    p_out = np.zeros(G.number_of_nodes())
    #    b_out = np.zeros(G.number_of_edges())
    #    zones,boundaries = zp.get_zones(G,Nmax)
    #    imbalance = {}
    #    mdl = {}
    #    node_mapping = {}
    #    edge_mapping = {}
    #    ph = {}
    #    bh = {}
    #    zone_cnt = 0
    #    for H,boundary in zip(zones,boundaries):
    #        print('Solving Zone %d: nodes=%d, edges=%d' %(zone_cnt,H.number_of_nodes(),H.number_of_edges()))
    #        ph[zone_cnt] = [p_in.pop() for i in range(H.number_of_nodes())]
    #        bh[zone_cnt] = [b_in.pop() for i in range(H.number_of_edges())]
    #        node_mapping[zone_cnt] = dict(zip(H.nodes(),range(H.number_of_nodes())))
    #        edge_mapping[zone_cnt] = {}
    #        for i,(u,v,l) in enumerate(H.edges_iter(data='id')):
    #                edge_mapping[zone_cnt][l] = i
    #        mdl[zone_cnt] = fm.full_MILP({'G':H,'boundary':boundary,'p':ph[zone_cnt],'b':bh[zone_cnt],'n_map':node_mapping[zone_cnt],'e_map':edge_mapping[zone_cnt],\
    #                'M':M,'delta_max':delta_max,'f_max':f_max,'balance_epsilon':balance_epsilon},\
    #                zone=True,logfile='Zone%s.log' %(zone_cnt))
    #        print('%s' %(model_status(mdl[zone_cnt])))
    #
    #        #power_perm =       get_permutation(mdl[zone_cnt], var_name='Pi', dim=H.number_of_nodes())
    #        #susceptance_perm = get_permutation(mdl[zone_cnt], var_name='Z',  dim=H.number_of_edges())
    #        imbalance =        get_imbalance(mdl[zone_cnt], boundary, node_mapping[zone_cnt], imbalance=imbalance)
    #        #for global_id, subgraph_id in node_mapping[zone_cnt].items():
    #        #    p_out[global_id] = ph[power_perm[subgraph_id]]
    #        #for global_id, subgraph_id in edge_mapping[zone_cnt].items():
    #        #    b_out[global_id] = bh[susceptance_perm[subgraph_id]]
    #        
    #        zone_cnt += 1

    #    for H in zones:
    #        for u,v in nx.edge_boundary(G, H.nodes()):
    #            check1 = np.abs(imbalance[u]['bp'] - imbalance[v]['bm'])
    #            check2 = np.abs(imbalance[u]['bm'] - imbalance[v]['bp'])
    #            if check1 > check2:
    #                print('(%d,%d) case 1: beta_u_plus=%0.3f, beta_v_minus=%0.3f' %(u,v,imbalance[u]['bp'],imbalance[v]['bm']) )
    #                # fix flow direction as bus u importing and bus v exporting
    #                # that means beta_minus_u = beta_plus_v = 0
    #                ztmp = np.where([v in boundary for boundary in boundaries])[0][0]
    #                beta_v_plus = mdl[ztmp].getVarByName('beta_plus[%d]' %(node_mapping[ztmp][v]))
    #                mdl[ztmp].addConstr(beta_v_plus == 0)

    #                ztmp = np.where([u in boundary for boundary in boundaries])[0][0]
    #                beta_u_minus = mdl[ztmp].getVarByName('beta_minus[%d]' %(node_mapping[ztmp][u]))
    #                mdl[ztmp].addConstr(beta_u_minus == 0)
    #            elif (check1) < 0 and ((imbalance[u]['bp'] != 0) or (imbalance[v]['bm'] != 0)):
    #                print('(%d,%d) case 2: beta_u_plus=%0.3f, beta_v_minus=%0.3f' %(u,v,imbalance[u]['bp'],imbalance[v]['bm']) )
    #                fix_count += 1
    #                # fix flow direction as bus u exporting and bus v import
    #                # that means beta_plus_u = beta_minus_v = 0
    #                ztmp = np.where([v in boundary for boundary in boundaries])[0][0]
    #                beta_v_minus = mdl[ztmp].getVarByName('beta_minus[%d]' %(node_mapping[ztmp][v]))
    #                mdl[ztmp].addConstr(beta_v_minus == 0)

    #                ztmp = np.where([u in boundary for boundary in boundaries])[0][0]
    #                beta_u_plus = mdl[ztmp].getVarByName('beta_plus[%d]' %(node_mapping[ztmp][u]))
    #                mdl[ztmp].addConstr(beta_u_plus == 0)
    #    print('imbalance fixes applied = %d' %(fix_count))
    #    if fix_count == 0:
    #        break
    #    else:
    #        for i in mdl:
    #            print('re-running zone %d' %i)
    #            mdl[i].optimize()
    #            print('%s' %(model_status(mdl[i])))
    #            imbalance = get_imbalance(mdl[i],boundaries[i],node_mapping[i],imbalance=imbalance)
    #    #while True:
    #    #    fix_count = 0
    #    #    for H in zones:
    #    #        for u,v in nx.edge_boundary(G, H.nodes()):
    #    #            check1 = np.abs(imbalance[u]['bp'] - imbalance[v]['bm']) - np.abs(imbalance[u]['bm'] - imbalance[v]['bp'])
    #    #            if (check1) > 0 and ((imbalance[u]['bm'] != 0) or (imbalance[v]['bp'] != 0)):
    #    #                print('(%d,%d) case 1: beta_u_minus=%0.3f, beta_v_plus=%0.3f' %(u,v,imbalance[u]['bm'],imbalance[v]['bp']) )
    #    #                fix_count += 1
    #    #                # fix flow direction as bus u importing and bus v exporting
    #    #                # that means beta_minus_u = beta_plus_v = 0
    #    #                ztmp = np.where([v in boundary for boundary in boundaries])[0][0]
    #    #                beta_v_plus = mdl[ztmp].getVarByName('beta_plus[%d]' %(node_mapping[ztmp][v]))
    #    #                mdl[ztmp].addConstr(beta_v_plus == 0)

    #    #                ztmp = np.where([u in boundary for boundary in boundaries])[0][0]
    #    #                beta_u_minus = mdl[ztmp].getVarByName('beta_minus[%d]' %(node_mapping[ztmp][u]))
    #    #                mdl[ztmp].addConstr(beta_u_minus == 0)
    #    #            elif (check1) < 0 and ((imbalance[u]['bp'] != 0) or (imbalance[v]['bm'] != 0)):
    #    #                print('(%d,%d) case 2: beta_u_plus=%0.3f, beta_v_minus=%0.3f' %(u,v,imbalance[u]['bp'],imbalance[v]['bm']) )
    #    #                fix_count += 1
    #    #                # fix flow direction as bus u exporting and bus v import
    #    #                # that means beta_plus_u = beta_minus_v = 0
    #    #                ztmp = np.where([v in boundary for boundary in boundaries])[0][0]
    #    #                beta_v_minus = mdl[ztmp].getVarByName('beta_minus[%d]' %(node_mapping[ztmp][v]))
    #    #                mdl[ztmp].addConstr(beta_v_minus == 0)

    #    #                ztmp = np.where([u in boundary for boundary in boundaries])[0][0]
    #    #                beta_u_plus = mdl[ztmp].getVarByName('beta_plus[%d]' %(node_mapping[ztmp][u]))
    #    #                mdl[ztmp].addConstr(beta_u_plus == 0)
    #    #    print('imbalance fixes applied = %d' %(fix_count))
    #    #    if fix_count == 0:
    #    #        break
    #    #    else:
    #    #        for i in mdl:
    #    #            print('re-running zone %d' %i)
    #    #            mdl[i].optimize()
    #    #            print('%s' %(model_status(mdl[i])))
    #    #            imbalance = get_imbalance(mdl[i],boundaries[i],node_mapping[i],imbalance=imbalance)

    #    edge_imbalance = {}
    #    import ipdb; ipdb.set_trace()
    #    for H in zones:
    #        for u,v in nx.edge_boundary(G, H.nodes()):
    #            edge_imbalance[u,v] = np.abs(imbalance[u] - imbalance[v])
    #    
    #    while len(edge_imbalance) > 0:
    #        u,v = max(edge_imbalance, key=edge_imbalance.get)
    #        if len(G.edge[u][v]) == 1:
    #            l = G.edge[u][v][0]['id']
    #            if b_out[l] == 0:
    #                b_out[l] = b_in.pop(np.argmin(b_in)) 
    #            else:
    #                raise Exception('Problem with intertie edges')
    #            edge_imbalance.pop((u,v))
    #        else:
    #            raise Exception('Parallel intertie lines')

    elif decomp == 'iter':
        iter_max = 5
        iter_epsilon = 0.5
        obj = {'Pi': [], 'Z': []}
        flag = False
        while not flag:
            print('ROUND %d' %(len(obj['Pi'])))
            print('Pi Optimization')
            print('----------------')
            if len(obj['Pi']) == 0:
                m = fm.only_p({'node_num':node_num,'branch_num':branch_num,'p':p_in,'b':b_in,\
                        'delta_max':delta_max,'f_max':f_max,'M':M,'fn':fn,'tn':tn,\
                        'from_lines':from_lines,'to_lines':to_lines,'balance_epsilon':balance_epsilon,\
                        'deg_one':deg_one})
            else:
                flow = get_var_list(m,'f',branch_num)
                slack = get_var_list(m,'s',branch_num)
                theta = get_var_list(m,'theta',node_num)
                m = fm.only_p({'node_num':node_num,'branch_num':branch_num,'p':p_in,'b':b_in,\
                        'delta_max':delta_max,'f_max':f_max,'M':M,'fn':fn,'tn':tn,\
                        'from_lines':from_lines,'to_lines':to_lines,'balance_epsilon':balance_epsilon,\
                        'deg_one':deg_one,'flow':flow,'slack':slack,'theta':theta},set_start=True)
            print('----------------')
            print('Pi Optimization Complete.')
            print('%s' %(model_status(m)))
            try:
                print('\tObjective Value: %0.3f' %(m.ObjVal))
                obj['Pi'].append(m.ObjVal)
                power_perm = get_permutation(m,var_name='Pi',dim=node_num)
                p_in = p_in[power_perm] #fix power permutation
            except:
                import ipdb; ipdb.set_trace()
            
            print('Z optimization...')
            print('----------------')
            flow = get_var_list(m,'f',branch_num)
            slack = get_var_list(m,'s',branch_num)
            theta = get_var_list(m,'theta',node_num)
            m = fm.only_b({'node_num':node_num,'branch_num':branch_num,'p':p_in,'b':b_in,\
                    'delta_max':delta_max,'f_max':f_max,'M':M,'fn':fn,'tn':tn,\
                    'from_lines':from_lines,'to_lines':to_lines,'balance_epsilon':balance_epsilon,\
                    'deg_one':deg_one,'flow':flow,'slack':slack,'theta':theta},set_start=True)
            print('----------------')
            print('Z Optimization Complete.')
            print('%s' %(model_status(m)))
            try:
                print('\tObjective Value: %0.3f' %(m.ObjVal))
                obj['Z'].append(m.ObjVal)
                susceptance_perm = get_permutation(m,var_name='Z',dim=branch_num)
                b_in = b_in[susceptance_perm] #fix suceptance permutation
            except:
                import ipdb; ipdb.set_trace()
            if (np.abs(obj['Pi'][-1] - obj['Z'][-1]) < iter_epsilon) or (len(obj['Pi']) > iter_max):
                print('Terminating: objective epsilon=%0.3f,\t iteration:%d' %( np.abs(obj['Pi'][-1] - obj['Z'][-1]),len(obj['Pi']) -1) )
                flag = True
        print('Final Objective Value: %0.3f' %(m.ObjVal))

    if mode == 'synth':
        pickle.dump({'Pg':Pg,'Pd':Pd,'power_perm':power_perm,'susceptance_perm':susceptance_perm,\
            'p_in':p_in,'b_in':b_in},open(savename,'wb')) 
    elif decomp == 'zone':
        pickle.dump({'Pg': Pg, 'Pd': Pd, 'p_out': p_out, 'b_out': b_out},open(savename,'wb'))
    else:
        pickle.dump({'Pg':Pg,'Pd':Pd,'power_perm':power_perm,'susceptance_perm':susceptance_perm,\
            'p_in':p_in,'b_in':b_in},open(savename,'wb')) 

if __name__=='__main__':
    import sys
    main(*sys.argv[1:])
