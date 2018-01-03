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
import multvar_init as mvinit
import ea_init as init
import logfun as lg

def main(savename, fdata, Nmax=400, Nmin=50, include_shunts=False, const_rate=True, actual_vars_d=False, actual_vars_g=True, actual_vars_z=True):

    start = time.time()
    FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
    logging.basicConfig(format=FORMAT,level=logging.INFO,datefmt='%H:%M:%S')

    logging.info("Saving to: %s",savename)
    logging.info("Topology data: %s", fdata)
     
    input_timestamp = lg.timestamp()

    #### INPUTS #########################
    truelist = [True,'True','true','t','1']
    Nmax = int(Nmax); Nmin = int(Nmin)
    actual_vars_z = actual_vars_z in truelist
    actual_vars_d = actual_vars_d in truelist
    actual_vars_g = actual_vars_g in truelist
    include_shunts= include_shunts in truelist
    const_rate    = const_rate in truelist

    ##### Define Constants ###############
    C = hlp.def_consts() 

    ##### Load Data ######### 
    bus_data, gen_data, branch_data = hlp.load_data(fdata)
    vmax = bus_data['VM'].max(); vmin = bus_data['VM'].min()
    C['umin'] = min(C['umin'],np.log(vmin))
    C['umax'] = max(C['umax'],np.log(vmax))

    #### Fit Power and Impedance Data #### 
    import fit_inputs as ftin
    resz,C['fmax'] = ftin.multivariate_z(branch_data, bw_method=0.01, actual_vars=actual_vars_z, fmaxin=C['fmax'], const_rate=const_rate)
    resd,resg,resf = ftin.multivariate_power(bus_data, gen_data, actual_vars_d=actual_vars_d, actual_vars_g=actual_vars_g, include_shunts=include_shunts)

    #### Get Topology ########
    G = mvinit.topology(bus_data,branch_data)
    N = G.number_of_nodes()
    L = G.number_of_edges()

    logging.info('Number of buses: %d, Number of branches: %d, Nmax: %d, Nmin: %d',N,L,Nmax,Nmin)

    ### Split Into Zones #####
    # if Nmax is sufficiently large there may be just 1 zone
    zones, boundaries, eboundary_map, e2z = mvinit.zones(G,Nmax,Nmin)
    inputs = mvinit.zone_inputs(zones, boundaries, eboundary_map, resd, resg, resf, resz, lg.log_input_samples)

    ### Main Loop ####################
    Psi = [None,None]
    for i in range(C['generations']):
        Psi[1] = eamutate(Psi[0],C['indivduals'])
        for psi in Psi[i]:
            easolve(psi)
        Psi[0] = easelection(Psi[0] + Psi[1])

    ### Sample Power and Impedance ####
    S = hlp.multivar_power_sample(N,resd,resg,resf)
    z = hlp.multivar_z_sample(L,resz)
    lg.log_input_samples(S,z)

    ### get primitive admittance values ####
    Y = hlp.Yparts(z['r'],z['x'],b=z['b'],tau=z['tap'],phi=z['shift'])
    bigM = hlp.bigM_calc(Y,C['fmax'],C['umax'],C['umax'],C['dmax'])
    lg.log_optimization_consts(C['lossmin'],C['lossterm'],C['fmax'],C['dmax'],C['htheta'],C['umin'],C['umax'],bigM=bigM)

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
