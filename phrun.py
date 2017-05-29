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

def timestamp():
    return datetime.now().strftime('%d-%m-%Y_%H%M')

def main(savename, fdata, mode='synth', method='ph'):
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
        Pg0,Pd0 = power_injections(gen_data,bus_data,equalize=False)
        for i in np.where(Pd0 < 0)[0]:
            Pg0[i] -= Pd0[i]
            Pd0[i] = 0
        gen_params  = {'vmax': np.max(Pg0[Pg0 >0]), 'vmin': np.min(Pg0[Pg0>0])}
        load_params = {'vmax': np.max(Pd0[Pd0 >0]), 'vmin': np.min(Pd0[Pd0>0])}
        Pg,Pd = hlp.injection_equalize_optimization(Pg0,Pd0,gen_params,load_params)
        p = (Pg-Pd)/100 # change to per unit
        #p_in = np.random.permutation(p)
        p_in = dict(zip(range(G.number_of_nodes()),p))
    else:
        #gen_params = {'vmax':800,'vmin':2,'dist':'exp','params':230.4}
        #load_params = {'vmax':275,'vmin':4,'dist':'lognorm','params':(3.4315,0.8363)}
        #gen_params  = {'vmax':2.5e3,'vmin':0.1,'dist':'exp',    'params':77.86}
        #load_params = {'vmax':365,  'vmin':0.1,'dist':'lognorm','params':(2.247,0.8737)}
        #Pg,Pd = injection_sample(G.number_of_nodes(),int_frac=0.23,inj_frac=0.053,gen_only_frac=0.03,gen_params=gen_params,load_params=load_params)
        Pgfit   = pickle.load(open('./cases/polish2383_wp_power_Pg_pchipfit.pkl','rb'))
        Pdfit   = pickle.load(open('./cases/polish2383_wp_power_Pd_pchipfit.pkl','rb'))
        Fracfit = pickle.load(open('./cases/polish2383_wp_power_frac.pkl','rb'))
        gen_params  = {'vmax': Pgfit['vmax'], 'vmin': Pgfit['vmin'], 'dist': 'pchip', 'params': Pgfit['pchip']}
        load_params = {'vmax': Pdfit['vmax'], 'vmin': Pdfit['vmin'], 'dist': 'pchip', 'params': Pdfit['pchip']}
        Pg,Pd,Pg0,Pd0 = injection_sample(G.number_of_nodes(), frac=Fracfit, gen_params=gen_params, load_params=load_params)
        p = (Pg - Pd)/100
        p_in = dict(zip(range(G.number_of_nodes()),p))
    logging.info('%0.3f <= Pg <= %0.3f', min(Pg[Pg>0]), max(Pg[Pg>0]))
    logging.info('%0.3f <= Pd <= %0.3f', min(Pd[Pd>0]), max(Pd[Pd>0]))
    logging.info('sum(Pg - Pd) = %0.3g', sum(Pg - Pd))
    logging.info('intermediate: %0.1f%%, Pg>Pd %0.1f%%, gen_only: %0.1f%%',100*sum((Pg == 0) & (Pd == 0))/Pd.shape[0], 100*sum(((Pg-Pd) > 0) & (Pd != 0))/sum(Pg>0), 100*sum((Pg > 0) & (Pd == 0))/sum(Pg > 0) )

    ######## susceptances #########
    if mode == 'real':
        b = -1/branch_data['BR_X'] 
        b_in = dict(zip(range(G.number_of_edges()),b))
    else:
        #b = get_b_from_dist(branch_num,dist='gamma',params=(1.88734, 0, 0.05856)) 
        #b = get_b_from_dist(G.number_of_edges(),dist='exp',params=(0,0.041),vmin=1e-4,vmax=0.4632) 
        fitb = pickle.load(open('./cases/polish2383_wp_reactance_pchipfit.pkl','rb'))
        b = get_b_from_dist(G.number_of_edges(), dist='kde', params=fitb['pchip'], vmin=fitb['vmin'], vmax=fitb['vmax']) 
        b_in = dict(zip(range(G.number_of_edges()),b))
    
    pickle.dump((Pg,Pd,-1/b,Pg0,Pd0),open('algorithm_inputs_' + timestamp() + '.pkl', 'wb')) 
    ####### constant inputs #########
    balance_epsilon = 1e-6
    slack_penalty   = 100
    delta_max       = 60.0*np.pi/180.0
    f_max           = 10
    beta_max        = f_max*0.75
    M               = f_max + delta_max*max(np.abs(b)) + 0.5 #plus half is out of precaution
    
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

    ####### Partition in Zones #############
    logging.info('Splitting graph into zones')
    zones, boundaries, eboundary_map = zp.get_zones(G,Nmax,Nmin)
    zone_cnt = len(zones)
    logging.info('%d Zones created',zone_cnt)
    ##### sort based on number of boundary edges #####
    boundary_edge_num  = {i:len(eboundary_map[i][1]) for i in range(zone_cnt)}
    boundary_edge_sort = sorted(boundary_edge_num,key=boundary_edge_num.get)
    zones         = [zones[i]         for i in boundary_edge_sort]
    boundaries    = [boundaries[i]    for i in boundary_edge_sort]
    eboundary_map = [eboundary_map[i] for i in boundary_edge_sort]

    for test in zones:
        if not nx.is_connected(nx.Graph(test)):
            import ipdb; ipdb.set_trace()
    boundary_edges,n2n = zp.boundary_edges(G,zones) 
    #pickle.dump((zones, boundaries, eboundary_map,boundary_edges,n2n),open('zone_dump.pkl','wb'))

    for i,(H,boundary,ebound) in enumerate(zip(zones,boundaries,eboundary_map)):
        logging.info('Initializing Zone %d: nodes=%d, edges=%d, boundary_edges=%d', i, H.number_of_nodes(), H.number_of_edges(), len(ebound[1]))
        #ph = {k: p_in[k] for k in random.sample(list(p_in),H.number_of_nodes())}
        ph = hlp.zone_power_sample(H.number_of_nodes(), p_in, len(ebound[1]), beta_max)
        bh = {k: b_in[k] for k in random.sample(list(b_in),H.number_of_edges())}
        Pgh = {}; Pdh = {}
        for k in ph:
            p_in.pop(k)
            Pgh[k] = Pg[k]
            Pdh[k] = Pd[k]
        for k in bh:
            b_in.pop(k)

        invars = {'G':H,'boundary':boundary, 'ebound':ebound,'p':ph,'b':bh, 'Pg':Pgh, 'Pd':Pdh,\
                'M':M, 'delta_max':delta_max,'f_max':f_max, 'beta_max':beta_max, 'balance_epsilon':balance_epsilon}
        solvers.append(fm.ZoneMILP(i,invars))
        
    logging.info('Remaining items in p_in: %d',len(p_in))
    logging.info('Remaining items in b_in: %d, number of boundary edges: %d', len(b_in), len(boundary_edges))

    ########### Main Loop ###########
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
        if method == 'lr':
            logging.info("Iteration %d starting: alpha = %0.3f", iter, alpha)
        elif method == 'ph':
            logging.info("Iteration %d starting: rho = %0.3f", iter, rho)
        for solver in solvers:
            logging.info("   Solving zone %d",solver.zone)
            if iter == 1:
                ### only apply time limit after the first iteration
                solver.m.setParam('TimeLimit',300)
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
            for k,v in solver.beta_val.items():
                if k not in beta_bar:
                    beta_bar[k] = v/2.
                else:
                    beta_bar[k] += v/2.

        ######### Determine Inter-Tie Errors #########
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

        mean_beta_diff = sum(beta_diff.values())/len(beta_diff)
        max_beta_diff  = max(beta_diff.values())
        logging.info("   GAP: %0.3f, MEAN beta_diff: %0.3f, MAX beta_diff: %0.3f", gap, mean_beta_diff, max_beta_diff)

        ######## output of iteration variables ############
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
        if method == 'ph':
            pickle.dump((beta,beta_bar,beta_diff,wdump,nu_map,pdump,bdump),open('iteration_%d_dump_%s_%s.pkl' %(iter,method,mode),'wb'))
        elif method == 'lr':
            pickle.dump((beta,beta_bar,beta_diff,nu,nu_map,pdump,bdump),open('iteration_%d_dump_%s_%s.pkl' %(iter,method,mode),'wb'))

        ######### Terminatio Criteria ##########
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

    ###### fixed beta iteration ############
    logging.info("Starting fixed-beta round")
    #beta_bar_final = {}
    #beta_final = {}
    for solver in solvers:
        logging.info("   Solving zone %d",solver.zone)
        #solver.fix_beta(beta_bar)
        #solver.add_balance_slack()
        #solver.balance_slack_objective(slack_penalty)
        #### make sure there is a solution
        #solver.m.setParam('TimeLimit','default')
        #solver.optimize()
        solver.fixed_beta(beta_bar,gen_params,load_params)
        #logging.info("      Solved with status %d, objective=%0.3f, total slack= %0.3f",solver.m.status,solver.m.objVal, solver.total_slack)
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
    p_out = {}
    b_out = {}
    #theta_out = {}
    alpha_out = {}
    for solver in solvers:
        p_out.update(solver.p_out)
        b_out.update(solver.b_out)
        #theta_out.update(solver.theta_out)
        alpha_out.update(solver.alpha_out)

    Pg_out = np.array([alpha_out[i]*Pg[p_out[i]] for i in range(G.number_of_nodes())])
    Pd_out = np.array([alpha_out[i]*Pd[p_out[i]] for i in range(G.number_of_nodes())])

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
    logging.info('Max flow:  %0.3g', max(abs(pf['flows'])) )
    logging.info('Max delta: %0.3f degree, %0.3f rad', max(abs(pf['delta']))*180/np.pi, max(abs(pf['delta'])) )

    #saveparts = savename.split('.') 
    #pickle.dump({'Pg': Pg_out, 'Pd': Pd_out, 'b': b_out, 'G': Gpf, 'pf': pf, 'ref': ref},
    #        open(saveparts[0] + "_noopt_" + timestamp() + "." + saveparts[1],'wb'))

    #if max(abs(pf['flows'])) > f_max:
    #    M = max(abs(pf['flows'])) + max(max(abs(pf['delta'])), delta_max)*max(np.abs(b_out)) + 0.5 #plus half is out of precaution
    #    invars = {'G':G,'p':(Pg_out-Pd_out)/100, 'b':b_out, 'edge_boundary':set(beta_bar.keys()), \
    #            'f_max':[f_max, max(abs(pf['flows']))], 'M':M, 'balance_epsilon':balance_epsilon, 'delta_max': delta_max}
    #    b_out = fm.intertie_suceptance_assign(invars)
    #    
    #    ##### Rerun DC powerflow ##############
    #    pf, Gpf = asg.DC_powerflow((Pg_out - Pd_out)/100, b_out, f_node, t_node, ref)
    #    logging.info('Max flow post optimization: %0.3g', max(abs(pf['flows'])) )
    #    logging.info('Max delta post optimization: %0.3f degree, %0.3f rad', max(abs(pf['delta']))*180/np.pi, max(abs(pf['delta'])) )

    ###### Saving and logging ######
    saveparts = savename.split('.') 
    pickle.dump({'Pg': Pg_out, 'Pd': Pd_out, 'b': b_out, 'G': Gpf, 'pf': pf, 'ref': ref},
            open(saveparts[0] + timestamp() + "." + saveparts[1],'wb'))
    end = time.time()
    seconds = int(end-start)
    hrs = seconds//3600
    seconds -= hrs*3600
    minutes = seconds//60
    seconds -= minutes*60
    logging.info("Total time: %dhr %dmin %dsec",hrs,minutes,seconds)

if __name__=='__main__':
    import sys
    main(*sys.argv[1:])
