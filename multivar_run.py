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


def main(savename, fdata):

    start = time.time()
    FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
    logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

    logging.info("Saving to: %s",savename)
    logging.info("Topology data: %s", fdata)
    
    input_timestamp = timestamp()

    ##### Load Data ######### 
    bus_data, gen_data, branch_data = hlp.load_data(fdata)

    #### Get Topology ########
    nmap   = dict(zip(bus_data['BUS_I'],range(bus_data.shape[0])))
    f_node = [nmap[i] for i in branch_data['F_BUS']]
    t_node = [nmap[i] for i in branch_data['T_BUS']]

    G = nx.MultiDiGraph()
    id = 0
    for i in branch_data.index:
        if branch_data['BR_STATUS'][i] > 0:
            G.add_edge(f_node[i],t_node[i],attr_dict={'id':id})
            id += 1
    
    N = G.number_of_nodes()
    L = G.number_of_edges()

    logging.info('Number of buses: %d, Number of branches: %d',N,L)

    #### Fit Power and Impedance Data #### 
    import fit_inputs as ftin
    resz = ftin.multivariate_z(branch_data,bw_method=0.01)
    resd,resg,resf = ftin.multivariate_power(bus_data,gen_data)

    ### Sample Power and Impedance ####
    S = hlp.multivar_power_sample(N,resd,resg,resf)
    z = hlp.multivar_z_sample(L,resz)
    log_input_samples(S,z)

    ### get primitive admittance values ####
    Y = hlp.Yparts(z['r'],z['x'],b=z['b'])

    #### optimization ########
    import formulation_multvar as fm
   
    # define constants
    #------------------
    fmax = 9 # per unit maximum real power flow on lin\mathcal{M}_{P^f}e
    dmax = 40*np.pi/180 # angle difference limit over a branch
    htheta = 7 #number of segments for approximating (theta_f - theta_t)^2/2
    umin = np.log(0.9)  # minimum ln(voltage)
    umax = np.log(1.1)  # maximum ln(voltage)
    
    Mpf  = fmax + max(Y['gff'] + Y['gft'])*(1+umax) + max(np.abs(Y['bft']))*dmax
    Mqf  = fmax + max(Y['bff'] + Y['bft'])*(1+umax) + max(np.abs(Y['gft']))*dmax
    Mpt  = fmax + max(Y['gtt'] + Y['gtf'])*(1+umax) + max(np.abs(Y['btf']))*dmax
    Mqt  = fmax + max(Y['btt'] + Y['btf'])*(1+umax) + max(np.abs(Y['gtf']))*dmax
    bigM = max(Mpf,Mqf,Mpt,Mqt)*1.1

    log_optimization_consts(fmax,dmax,htheta,umin,umax,bigM)

    ### solve ####
    vars = fm.single_system(G,fmax,dmax,htheta,umin,umax,z,S,bigM)
    vars['G'] = G

    ###### Saving ######
    saveparts = savename.split('.') 
    pickle.dump(vars,\
            open(saveparts[0] + timestamp() + "inputstamp_" + input_timestamp + "." + saveparts[1],'wb'))
    #### run solution check####
    import multvar_solution_check as solchk
    solchk.rescheck(vars)

    #### log final time ####
    end = time.time()
    hrs,minutes,seconds = timeparts(start,end)
    logging.info("Total time: %dhr %dmin %dsec",hrs,minutes,seconds)

def timestamp():
    return datetime.now().strftime('%d-%m-%Y_%H%M')

def timeparts(start,end):
    seconds = int(end-start)
    hrs = seconds//3600
    seconds -= hrs*3600
    minutes = seconds//60
    seconds -= minutes*60
    return hrs,minutes,seconds 

##### SUBROUTINES ##########
def log_input_samples(S,z):
    logging.info('------ Power Info -------')
    logging.info('Load:')
    logging.info('Total: %0.4f MW, %0.4f MVar', sum(S['Pd']), sum(S['Qd']))
    logging.info('max: %0.4f MW, %0.4f MVar', max(S['Pd']), max(S['Qd']))
    logging.info('min (non 0): %0.4f MW, %0.4f MVar', min(S['Pd'][S['Pd'] != 0]), min(S['Qd'][S['Qd'] != 0]))
    logging.info('Avg: %0.4f WM, %0.4f MVar', np.mean(S['Pd']), np.mean(S['Qd']))
    logging.info('Std: %0.4f WM, %0.4f MVar', np.std(S['Pd']),  np.std(S['Qd']))
    logging.info('Gen Max:')
    logging.info('Total: %0.4f MW, %0.4f MVar', sum(S['Pgmax']), sum(S['Qgmax']))
    logging.info('max: %0.4f MW, %0.4f MVar', max(S['Pgmax']), max(S['Qgmax']))
    logging.info('min (non 0): %0.4f MW, %0.4f MVar', min(S['Pgmax'][S['Pgmax'] != 0]), min(S['Qgmax'][S['Qgmax'] != 0]))
    logging.info('Avg (non 0): %0.4f WM, %0.4f MVar', np.mean(S['Pgmax'][S['Pgmax'] != 0]), np.mean(S['Qgmax'][S['Qgmax'] != 0]))
    logging.info('Std (non 0): %0.4f WM, %0.4f MVar', np.std(S['Pgmax'][S['Pgmax'] != 0]),  np.std(S['Qgmax'][S['Qgmax'] != 0]))
    logging.info('Gen Min:')
    logging.info('Total: %0.4f MW', sum(S['Pgmin']))
    logging.info('max: %0.4f MW', max(S['Pgmin']))
    if np.any(S['Pgmin'] != 0):
        logging.info('min (non 0): %0.4f MW', min(S['Pgmin'][S['Pgmin'] != 0]))
        logging.info('Avg (non 0): %0.4f WM', np.mean(S['Pgmin'][S['Pgmin'] != 0]))
        logging.info('Std (non 0): %0.4f WM', np.std(S['Pgmin'][S['Pgmin'] != 0]))

    logging.info('------Impedance Info----------')
    logging.info('Max         (r,x,b): %0.3g, %0.3g, %0.3g', max(z['r']), max(z['x']), max(z['b']))
    logging.info('Min         (r,x,b): %0.3g, %0.3g, %0.3g', min(z['r']), min(z['x']), min(z['b']))
    logging.info('Min (non 0) (r,x,b): %0.3g, %0.3g, %0.3g', min(z['r'][z['r'] != 0]), min(z['x'][z['x'] != 0]), min(z['b'][z['b'] != 0]))
    logging.info('Avg         (r,x,b): %0.3g, %0.3g, %0.3g', np.mean(z['r']), np.mean(z['x']), np.mean(z['b']))
    logging.info('Std         (r,x,b): %0.3g, %0.3g, %0.3g', np.std(z['r']), np.std(z['x']), np.std(z['b']))

def log_optimization_consts(fmax,dmax,htheta,umin,umax,bigM):
    logging.info('-----Optimization Constants------')
    logging.info('Flow Max (P or Q)    [p.u]: %0.2f',fmax)
    logging.info('Angle Difference Max [rad]: %0.4f',dmax)
    logging.info('htheta: %d',htheta)
    logging.info('u min: %0.4f', umin)
    logging.info('u max: %0.4f', umax)
    logging.info('big M: %0.4g', bigM)

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
