from datetime import datetime
import time
import pandas as pd
import numpy as np
import networkx as nx
import random
import logging
import pickle
from scipy import stats
import helpers as hlp
import assignment_analysis as asg


def main(cnt):
    ### Topology
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
    fitb = pickle.load(open('./cases/polish2383_wp_reactance_pchipfit.pkl','rb'))
    
    delta_out = {}
    flows_out = {}
    max_out   = {'delta':[], 'flows':[]}
    for i in range(100):
        print('-----------ITERATION %d----------' %(i))
        #### Load and Generation
        gen_params  = {'vmax': Pgfit['vmax'], 'vmin': Pgfit['vmin'], 'dist': 'pchip', 'params': Pgfit['pchip']}
        load_params = {'vmax': Pdfit['vmax'], 'vmin': Pdfit['vmin'], 'dist': 'pchip', 'params': Pdfit['pchip']}
        Pg,Pd,Pg0,Pd0 = hlp.injection_sample(G.number_of_nodes(), frac=Fracfit, gen_params=gen_params, load_params=load_params)
        p = (Pg - Pd)/100
        p_in = np.random.permutation(p)
        
        ### Susceptance
        b = hlp.get_b_from_dist(G.number_of_edges(), dist='kde', params=fitb['pchip'], vmin=fitb['vmin'], vmax=fitb['vmax']) 
        b_in = np.random.permutation(b)
        
        ref = np.argmax(p_in)
        pf, _ = asg.DC_powerflow(p_in, b_in, f_node, t_node, ref)
    
        delta_out[i] = pf['delta']*180/np.pi
        flows_out[i] = pf['flows']*100
        max_out['delta'].append(max(abs(pf['delta']))*180/np.pi)
        max_out['flows'].append(max(abs(pf['flows']))*100)
    
    
    delta = pd.DataFrame(delta_out)
    flows = pd.DataFrame(flows_out)
    
    #hist = {}
    #hist['delta_v'],    hist['delta_edges']     = np.histogram(delta.as_matrix().ravel(), bins='auto', density=True)
    #hist['flows_v'],    hist['flows_edges']     = np.histogram(flows.as_matrix().ravel(), bins='auto', density=True)
    #hist['max_delta_v'],hist['max_delta_edges'] = np.histogram(max_out['delta'],          bins='auto', density=True)
    #hist['max_flows_v'],hist['max_flows_edges'] = np.histogram(max_out['delta'],          bins='auto', density=True)
    
    #### save output
    pd.DataFrame(max_out).to_csv('random_assignment_max_out' + cnt +'.csv')
    delta.to_csv('random_assignment_delta' + cnt + '.csv')
    flows.to_csv('random_assignment_flows' + cnt + '.csv')
    #pd.DataFrame(dict([(k,pd.Series(v)) for k,v in hist.items()])).to_csv('random_assignment_distributions.csv',index=False)

if __name__== '__main__':
    import sys
    main(*sys.argv[1:])
