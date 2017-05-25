import sys
from datetime import datetime
import time
import pandas as pd
import numpy as np
import networkx as nx
import random
import logging
import pprint
import pickle
from scipy import stats
import helpers as hlp
from helpers import get_permutation, model_status, load_data, power_injections 
from helpers import injection_sample, get_b_from_dist 
import formulation as fm

def timestamp():
    return datetime.now().strftime('%d-%m-%Y_%H%M')

def main():
    """
        modes:
            real: only shuffle injections and impedance
            bsynth: real injections, synthetic impedance
            pbsyth: synthetic injections and impednce, still real topology
            synth: everyting synthetic
    """
    start = time.time()
    FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
    logging.basicConfig(format=FORMAT,level=logging.INFO,datefmt='%H:%M:%S')
        
    top = pd.read_csv('./cases/RT_3000.csv')
    # change to zero indexing
    top['f'] -= 1
    top['t'] -= 1
    f_node = top['f'].values
    t_node = top['t'].values

    G = nx.MultiDiGraph()
    G.add_edges_from(zip(f_node,t_node,[{'id':i} for i in range(f_node.shape[0])]))

    Pgfit   = pickle.load(open('./cases/polish2383_wp_power_Pg_pchipfit.pkl','rb'))
    Pdfit   = pickle.load(open('./cases/polish2383_wp_power_Pd_pchipfit.pkl','rb'))
    Fracfit = pickle.load(open('./cases/polish2383_wp_power_frac.pkl','rb'))
    gen_params  = {'vmax': Pgfit['vmax'], 'vmin': Pgfit['vmin'], 'dist': 'pchip', 'params': Pgfit['pchip']}
    load_params = {'vmax': Pdfit['vmax'], 'vmin': Pdfit['vmin'], 'dist': 'pchip', 'params': Pdfit['pchip']}

    Pg,Pd,x,Pg0,Pd0 = pickle.load(open('algorithm_inputs_21-05-2017_0424.pkl', 'rb')) 
    b = -1./x
    p = (Pg - Pd)/100
    p_in = dict(zip(range(G.number_of_nodes()),p))
    b_in = dict(zip(range(G.number_of_edges()),b))
    zones, boundaries, eboundary_map,boundary_edges,n2n = pickle.load(open('zone_dump.pkl','rb'))
    beta,beta_bar,beta_diff,wdump,nu_map,pdump,bdump = pickle.load(open('iteration_5_dump_ph.pkl','rb'))
    
    ####### constant inputs #########
    balance_epsilon = 1e-6
    slack_penalty   = 100
    delta_max       = 60.0*np.pi/180.0
    f_max           = 10
    beta_max        = f_max*0.75
    M               = f_max + delta_max*max(np.abs(b)) + 0.5 #plus half is out of precaution

    
    solvers = []
    beta_bar_final = {}
    beta_final = {}
    for i,(H,boundary,ebound) in enumerate(zip(zones,boundaries,eboundary_map)):
        logging.info('Initializing Zone %d: nodes=%d, edges=%d, boundary_edges=%d', i, H.number_of_nodes(), H.number_of_edges(), len(ebound[1]))
        #ph = hlp.zone_power_sample(H.number_of_nodes(), p_in, len(ebound[1]), beta_max)
        ph = {v: p_in[v] for _,v in pdump[i].items()}
        #bh = {k: b_in[k] for k in random.sample(list(b_in),H.number_of_edges())}
        bh = {v: b_in[v] for _,v in bdump[i].items()}
        Pgh = {}; Pdh = {}
        for k in ph:
            p_in.pop(k)
            Pgh[k] = Pg[k]
            Pdh[k] = Pd[k]
        for k in bh:
            b_in.pop(k)
        imports = 0; exports = 0
        imports_bar= 0; exports_bar = 0
        for k,v in ebound[0]['out'].items():
            for kk in v:
                exports += beta[i][kk]
                exports_bar += beta_bar[kk]
        for k,v in ebound[0]['in'].items():
            for kk in v:
                imports += beta[i][kk]
                imports_bar += beta_bar[kk]
        ph_sum = sum(v for k,v in ph.items())
        logging.info('zone error with beta: %0.3f, zone error with beta_bar; %0.3f', ph_sum + imports - exports, ph_sum + imports_bar - exports_bar) 

        invars = {'G':H,'boundary':boundary, 'ebound':ebound,'p':ph,'b':bh, 'Pg':Pgh, 'Pd':Pdh,\
                'M':M, 'delta_max':delta_max,'f_max':f_max, 'beta_max':beta_max, 'balance_epsilon':balance_epsilon}
        solvers.append(fm.ZoneMILP(i,invars))
    
    #logging.info('Remaining items in p_in: %d',len(p_in))
    #logging.info('Remaining items in b_in: %d, number of boundary edges: %d', len(b_in), len(boundary_edges))
    #
    ####### fixed beta iteration ############
    #logging.info("Starting fixed-beta round")
    #beta_bar_final = {}
    #for solver in solvers:
        solver = solvers[-1]
        logging.info("   Solving zone %d",solver.zone)
        ### set intial solution
        pset_count = 0
        for i,j in solver.Pi.keys():
            if pdump[solver.zone][solver.inv_node_map[i]] == solver.p_map[j]:
                solver.Pi[i,j].start = 1
                solver.Pi[i,j].lb = 1
                solver.Pi[i,j].ub = 1
                pset_count +=1
            else:
                solver.Pi[i,j].start = 0
                #solver.Pi[i,j].lb = 0
                #solver.Pi[i,j].ub = 0
        zset_count = 0
        for i,j in solver.Z.keys():
            if bdump[solver.zone][solver.inv_edge_map[i]] == solver.b_map[j]:
                zset_count += 1
                solver.Z[i,j].start = 1
                solver.Z[i,j].lb = 1
                solver.Z[i,j].ub = 1
            else:
                solver.Z[i,j].start = 0
                solver.Z[i,j].lb = 0
                solver.Z[i,j].ub = 0
        logging.info("    Set %d entries of Z and %d entries of Pi to 1", zset_count,pset_count)
        logging.info("    Populating variables")
        solver.optimize() 
        logging.info("    Solving with fixed beta")
        solver.fixed_beta(beta_bar,gen_params,load_params)
        #import ipdb; ipdb.set_trace()
        #solver.fix_beta(beta_bar)
        #solver.add_balance_slack()
        #solver.balance_slack_objective(slack_penalty)
        #solver.m.write('zone%d.lp' %(solver.zone))
        #sys.exit(0)
        ### make sure there is a solution
        #solver.m.setParam('TimeLimit','default')
        #solver.optimize()
        #logging.info("      Solved with status %d, objective=%0.3f, total slack= %0.3f, slack mean= %0.3g, slack std = %0.3g",solver.m.status,solver.m.objVal, solver.total_slack, solver.slack_stat['mean'], solver.slack_stat['std'])
        ### sanity check that the beta fixe worked
        #beta_final[solver.zone] = solver.beta_val
        #for k,v in solver.beta_val.items():
        #    if k not in beta_bar_final:
        #        beta_bar_final[k] = v/2.
        #    else:
        #        beta_bar_final[k] += v/2.
    
    #gap = 0
    #for z in beta_final:
    #    for k,v in beta_final[z].items():
    #        gap += np.abs(v - beta_bar_final[k])
    #logging.info("      Final Calculated GAP: %0.3g", gap)
    
    ########## Get Final Power Assignment ###########
    p_out     = {}
    b_out     = {}
    alpha_out ={}
    for solver in solvers:
        p_out.update(solver.p_out)
        b_out.update(solver.b_out)
        alpha_out.update(solver.alpha_out)
    
    #Pg_out = np.array([Pg[p_out[i]] for i in range(G.number_of_nodes())])
    #Pd_out = np.array([Pd[p_out[i]] for i in range(G.number_of_nodes())])
    Pg_out = np.array([alpha_out[i]*Pg[p_out[i]] for i in range(G.number_of_nodes())])
    Pd_out = np.array([alpha_out[i]*Pd[p_out[i]] for i in range(G.number_of_nodes())])
    logging.info("After alpha modification: %0.3f <= Pg <= %0.3f", min(Pg_out[Pg_out > 0]), Pg_out.max())
    logging.info("After alpha modification: %0.3f <= Pd <= %0.3f", min(Pd_out[Pd_out > 0]), Pd_out.max())
    
    ####### Assign Susceptance to Inter-Tie Branches ###########
    edge_order = sorted(beta_bar,key=lambda x: abs(beta_bar[x]),reverse=True)
    b_order    = sorted(b_in,key=b_in.get)
    for l in edge_order:
        b_out[l] = b_order.pop(0)
    
    if len(p_out) != G.number_of_nodes():
        import ipdb; ipdb.set_trace()
    if len(b_out) != G.number_of_edges():
        import ipdb; ipdb.set_trace()
    b_out  = np.array([b[b_out[i]]  for i in range(G.number_of_edges())])
    
    ####### DC Powerflow ###########
    import assignment_analysis as asg
    ref = np.argmax((Pg_out - Pd_out)/100)
    pf, Gpf = asg.DC_powerflow((Pg_out - Pd_out)/100, b_out, f_node, t_node, ref)
    logging.info('Max flow pre optimization: %0.3g', max(abs(pf['flows'])) )
    logging.info('Max delta pre optimization: %0.3f degree, %0.3f rad', max(abs(pf['delta']))*180/np.pi, max(abs(pf['delta'])) )
    
    pickle.dump({'Pg': Pg_out, 'Pd': Pd_out, 'b': b_out, 'G': Gpf, 'pf': pf, 'ref': ref},
            open("beta_fix_test_" + timestamp() + ".pkl",'wb'))

if __name__=='__main__':
    main()
