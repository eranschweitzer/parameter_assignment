import pandas as pd
import numpy as np
import networkx as nx
import random
import logging
import pprint
import pickle
from scipy import stats
import helpers
from helpers import get_permutation, model_status, load_data, power_injections 
from helpers import injection_sample, get_b_from_dist

def main(savename,fdata,method='ph',mode='real'):
    """
        modes:
            real: only shuffle injections and impedance
            bsynth: real injections, synthetic impedance
            pbsyth: synthetic injections and impednce, still real topology
            synth: everyting synthetic
    """
    FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
    logging.basicConfig(format=FORMAT,level=logging.INFO,datefmt='%H:%M:%S')
    ###### Topological data ###########
    if mode == 'synth': 
        top = pd.read_csv(fdata)
        # change to zero indexing
        top['f'] -= 1
        top['t'] -= 1
        f_node = top['f'].values
        t_node = top['t'].values
    else:
        bus_data,gen_data,branch_data = load_data(fdata)
        f_node = branch_data['F_BUS'].values
        t_node = branch_data['T_BUS'].values

    G = nx.MultiDiGraph()
    G.add_edges_from(zip(f_node,t_node,[{'id':i} for i in range(f_node.shape[0])]))

    ###### power injections #########
    if mode in ['real','bsyhnth']:
        Pg,Pd = power_injections(gen_data,bus_data)
        p = (Pg-Pd)/100 # change to per unit
        #p_in = np.random.permutation(p)
        p_in = dict(zip(range(G.number_of_nodes()),p))
    else:
        #gen_params = {'vmax':800,'vmin':2,'dist':'exp','params':230.4}
        #load_params = {'vmax':275,'vmin':4,'dist':'lognorm','params':(3.4315,0.8363)}
        gen_params  = {'vmax':2.5e3,'vmin':0.1,'dist':'exp',    'params':77.86}
        load_params = {'vmax':365,  'vmin':0.1,'dist':'lognorm','params':(2.247,0.8737)}
        Pg,Pd = injection_sample(G.number_of_nodes(),int_frac=0.23,inj_frac=0.053,gen_only_frac=0.03,gen_params=gen_params,load_params=load_params)
        p = (Pg - Pd)/100
        p_in = dict(zip(range(G.number_of_nodes()),p))

    ######## susceptances #########
    if mode == 'real':
        b = -1/branch_data['BR_X'] 
        b_in = dict(zip(range(G.number_of_edges()),b))
    else:
        #b = get_b_from_dist(branch_num,dist='gamma',params=(1.88734, 0, 0.05856)) 
        b = get_b_from_dist(G.number_of_edges(),dist='exp',params=(0,0.041),vmin=1e-4,vmax=0.4632) 
        b_in = dict(zip(range(G.number_of_edges()),b))

    ####### constant inputs #########
    balance_epsilon = 1e-4
    delta_max = 60.0*np.pi/180.0
    f_max = 10
    M = f_max + delta_max*max(np.abs(b)) + 0.5 #plus half is out of precaution
    

    ####### optimization ##########
    import formulation as fm
    import zone_splitting as zp

    Nmax = 400; Nmin = 50;
    rho0 = 1
    alpha0 = 0.05
    gapmax = 5
    if method =='lr':
        itermax = 15
    elif method == 'ph':
        itermax = 5
    mean_beta_cut = 0.05
    max_beta_cut = 0.1
    nu = {}
    nu_map = {}
    solvers = []
    #p_in = np.random.permutation(p_in).tolist()
    #b_in = np.random.permutation(b_in).tolist()
    #p_out = np.zeros(G.number_of_nodes())
    #b_out = np.zeros(G.number_of_edges())
    logging.info('Splitting graph into zones')
    zones, boundaries, eboundary_map = zp.get_zones(G,Nmax,Nmin)
    logging.info('%d Zones created',len(zones))
    for test in zones:
        if not nx.is_connected(nx.Graph(test)):
            import ipdb; ipdb.set_trace()
    boundary_edges,n2n = zp.boundary_edges(G,zones) 
    #pickle.dump((zones,boundaries,boundary_edges,n2n),open('zone_dump.pkl','wb'))
    pickle.dump((zones, boundaries, eboundary_map,boundary_edges,n2n),open('zone_dump.pkl','wb'))
    #imbalance = {}
    #mdl = {}
    #node_mapping = {}
    #edge_mapping = {}
    #ph = {}
    #bh = {}
    zone_cnt = len(zones)
    for i,(H,boundary,ebound) in enumerate(zip(zones,boundaries,eboundary_map)):
        logging.info('Initializing Zone %d: nodes=%d, edges=%d', i, H.number_of_nodes(), H.number_of_edges())
        ph = {k: p_in[k] for k in random.sample(list(p_in),H.number_of_nodes())}
        bh = {k: b_in[k] for k in random.sample(list(b_in),H.number_of_edges())}
        for k in ph:
            p_in.pop(k)
        for k in bh:
            b_in.pop(k)
        #ph[i] = [p_in.pop() for k in ph[i]]
        #bh[i] = [b_in.pop() for k in bh[i]]
        #node_mapping = dict(zip(H.nodes(),range(H.number_of_nodes())))
        #edge_mapping = {}
        #for j,(u,v,l) in enumerate(H.edges_iter(data='id')):
        #        edge_mapping[l] = j

        invars = {'G':H,'boundary':boundary, 'ebound':ebound,'p':ph,'b':bh,\
                'M':M,'delta_max':delta_max,'f_max':f_max,'balance_epsilon':balance_epsilon}
        solvers.append(fm.ZoneMILP(i,invars))
        
    logging.info('Remaining items in p_in: %d',len(p_in))
    logging.info('Remaining items in b_in: %d, number of boundary edges: %d', len(b_in), len(boundary_edges))

    iter = 0
    alpha_dim_iter = 1
    while True:
        beta = {}
        beta_bar = {}
        beta_diff = {}
        if iter > 0:
            alpha = alpha0/np.sqrt(iter)
            rho   = rho0/np.sqrt(iter)
        else:
            alpha = alpha0
            rho   = rho0
        #    if max_beta_diff < alpha:
        #        alpha_dim_iter += 1
        #        alpha = alpha/np.sqrt(alpha_dim_iter)
        if method == 'lr':
            logging.info("Iteration %d starting: alpha = %0.3f", iter, alpha)
        elif method == 'ph':
            logging.info("Iteration %d starting: rho = %0.3f", iter, rho)
        for solver in solvers:
            logging.info("   Solving zone %d",solver.zone)
            solver.optimize()
            if solver.m.solcount == 0:
                solver.m.setParam('TimeLimit','default')
                logging.info("     fixing Z binaries")
                solver.fix_Z()
                logging.info("     fixing P binaries")
                solver.fix_Pi()

                logging.info("     resolving")
                solver.optimize()

                logging.info("     unfixing Z binaries")
                solver.unfix_Z()
                logging.info("     unfixing P binaries")
                solver.unfix_Pi()
                solver.m.setParam('TimeLimit',300)
            logging.info("      Solved with status %d, objective=%0.3f",solver.m.status,solver.m.objVal)
            beta[solver.zone] = solver.beta_val
            #beta.update(solver.beta_val)
            for k,v in solver.beta_val.items():
                if k not in beta_bar:
                    beta_bar[k] = v/2.
                else:
                    beta_bar[k] += v/2.
        for k in beta_bar.keys():
            z = np.where([k in beta[i].keys() for i in range(zone_cnt)])[0]
            zone_i = min(z); zone_j = max(z)
            if iter == 0:
                nu[k]     = alpha*(beta[zone_i][k] - beta[zone_j][k])
                nu_map[k] = {zone_i:1, zone_j:-1}
            else:
                nu[k]     += alpha*(beta[zone_i][k] - beta[zone_j][k])
            beta_diff[k] = np.abs(beta[z[0]][k] - beta[z[1]][k])
        gap = 0
        for z in beta:
            for k,v in beta[z].items():
                gap += np.abs(v - beta_bar[k])
        #str_tmp = ", ".join(["%d: %0.2f" %(k,v) for k,v in sorted(beta.items())])
        #logging.debug("beta    : {" + str_tmp + "}")

        #str_tmp = ", ".join(["%d: %0.2f" %(k,v) for k,v in sorted(beta_bar.items())])
        #logging.debug("beta_bar: {" + str_tmp  + "}")
        #for u,v in boundary_edges:
        #    logging.debug("(%d, %d): (%0.2f, %0.2f)",u,v,beta[u],beta[v])
        mean_beta_diff = sum(beta_diff.values())/len(beta_diff)
        max_beta_diff  = max(beta_diff.values())
        logging.info("   GAP: %0.3f, MEAN beta_diff: %0.3f, MAX beta_diff: %0.3f", gap, mean_beta_diff, max_beta_diff)

        wdump = {}
        pdump = {}
        bdump = {}
        for solver in solvers:
            if method == 'ph':
                solver.ph_objective_update(beta_bar,rho)
                wdump[solver.zone] = {k: v for k,v in sorted(solver.w.items())}
                pdump[solver.zone] = solver.p_out
                bdump[solver.zone] = solver.b_out
            elif method == 'lr':
                solver.lr_objective_update(nu,nu_map)
            #str_tmp = ", ".join(["%d: %0.2f" %(solver.inv_node_map[k],v) for k,v in sorted(solver.w.items())])
            #logging.debug("w_%d: {" + str_tmp  + "}", solver.zone)
        if method == 'ph':
            pickle.dump((beta,beta_bar,beta_diff,wdump,nu_map,pdump,bdump),open('iteration_%d_dump_%s.pkl' %(iter,method),'wb'))
        elif method == 'lr':
            pickle.dump((beta,beta_bar,beta_diff,nu,nu_map,pdump,bdump),open('iteration_%d_dump_%s.pkl' %(iter,method),'wb'))

        if (gap <= gapmax):
            logging.info("Stopping iteration: Gap tolerance reached")
            break
        elif iter == itermax:
            logging.info("Stopping iteration: Maximum iteration reached")
            break
        elif mean_beta_diff <= mean_beta_cut:
            logging.info("Stopping iteration: Mean beta cutoff reached")
            break
        elif max_beta_diff <= max_beta_cut:
            logging.info("Stopping iteration: Max beta cutoff reached")
            break

        iter += 1

    p_out = {}
    b_out = {}
    theta_out = {}
    for solver in solvers:
        p_out.update(solver.p_out)
        b_out.update(solver.b_out)
        theta_out.update(solver.theta_out)
    
    #boundary_angle_diff = {}
    #for u,v in boundary_edges:
    #    boundary_angle_diff[u,v] = np.abs(theta_out[u] - theta_out[v])
    
    #edge_order = sorted(boundary_angle_diff,key=boundary_angle_diff.get,reverse=True)
    edge_order = sorted(beta_bar,key=lambda x: abs(beta_bar[x]),reverse=True)
    b_order    = sorted(b_in,key=b_in.get)
    for l in edge_order:
        b_out[l] = b_order.pop(0)
    #for u,v in edge_order:
    #    for i,d in G.edge[u][v].items():
    #        b_out[d['id']] = b_order.pop(0)
    
    if len(p_out) != G.number_of_nodes():
        import ipdb; ipdb.set_trace()
    if len(b_out) != G.number_of_edges():
        import ipdb; ipdb.set_trace()
    Pg_out = np.array([Pg[p_out[i]] for i in range(G.number_of_nodes())])
    Pd_out = np.array([Pd[p_out[i]] for i in range(G.number_of_nodes())])
    b_out  = np.array([b[b_out[i]]  for i in range(G.number_of_edges())])
    pickle.dump({'Pg': Pg_out, 'Pd': Pd_out, 'b': b_out},open(savename,'wb'))

if __name__=='__main__':
    import sys
    main(*sys.argv[1:])
