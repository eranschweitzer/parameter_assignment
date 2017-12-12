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
import multvar_init as init


def main(savename, fdata, Nmax=400, Nmin=50, actual_vars_d=False, actual_vars_g=True, actual_vars_z=True):

    start = time.time()
    FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
    logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

    logging.info("Saving to: %s",savename)
    logging.info("Topology data: %s", fdata)
    
    input_timestamp = timestamp()

    #### INPUTS #########################
    truelist = [True,'True','true','t','1']
    actual_vars_z = actual_vars_z in truelist
    actual_vars_d = actual_vars_d in truelist
    actual_vars_g = actual_vars_g in truelist

    ##### Define Constants ###############
    Nmax = int(Nmax); Nmin = int(Nmin)
    fmax = 9            # default per unit maximum real power flow on line
    dmax = 40*np.pi/180 # angle difference limit over a branch
    htheta = 7          # number of segments for approximating (theta_f - theta_t)^2/2
    umin = np.log(0.9)  # minimum ln(voltage)
    umax = np.log(1.05) # maximum ln(voltage)
    lossmin = 0.01      # minimum losses required (fraction = (Pg - Pd)/Pg)
    lossterm= 0.05      # terminate optimization when if losses are at this level or below
    thresholds = {'gap':       5,
                  'mean_diff': 0.05,
                  'max_diff':  0.1,
                  'itermax':   5}
    rho = 1
    
    ##### Load Data ######### 
    bus_data, gen_data, branch_data = hlp.load_data(fdata)

    #### Get Topology ########
    G = init.topology(bus_data,branch_data)
    N = G.number_of_nodes()
    L = G.number_of_edges()

    logging.info('Number of buses: %d, Number of branches: %d, Nmax: %d, Nmin: %d',N,L,Nmax,Nmin)

    #### Fit Power and Impedance Data #### 
    import fit_inputs as ftin
    resz = ftin.multivariate_z(branch_data, bw_method=0.01, actual_vars=actual_vars_z)
    resd,resg,resf = ftin.multivariate_power(bus_data, gen_data, actual_vars_d=actual_vars_d, actual_vars_g=actual_vars_g)

    #### optimization ########
    import formulation_multvar as fm
    i = 0
    if N > Nmax:
        import multvar_solve as slv
        import multvar_output as out

        log_optimization_consts(lossmin,lossterm,fmax,dmax,htheta,umin,umax,thresholds=thresholds)
        solvers,e2z = init.solvers_init(G,Nmax,Nmin,resd,resg,resf,resz,lossmin,lossterm,fmax,dmax,htheta,umin,umax,log_input_samples) 
        while True:
            log_iteration_start(i,rho)
            beta_bar, gamma_bar, ivals = slv.solve(solvers, e2z,logging=log_iterations)
            log_iteration_summary(beta_bar,gamma_bar, ivals)
            flag,msg = slv.termination(i, ivals, thresholds)
            if flag:
                logging.info('===============================')
                logging.info('TERMINATION CRITERIA SATISFIED')
                for part in msg:
                    logging.info("%s", part)
                logging.info('===============================')
                break
            else:
                slv.update(solvers, iter, beta_bar, gamma_bar, rho)
                i += 1
        nvars,lvars = out.getvars(solvers, N, L)
        ##### tie line branch samples #########
        logging.info('Resolving Tie-Line samples')
        import multvar_tieline as tie
        tz = hlp.multivar_z_sample(len(e2z), resz)
        vars = tie.tieassign(G, nvars, lvars, lossmin, lossterm, fmax, dmax, htheta, umin, umax, tz, list(e2z.keys()))
    else:
        ### Sample Power and Impedance ####
        S = hlp.multivar_power_sample(N,resd,resg,resf)
        z = hlp.multivar_z_sample(L,resz,fmaxin=fmax)
        fmax = max(z['rate']) # update fmax
        log_input_samples(S,z)

        ### get primitive admittance values ####
        Y = hlp.Yparts(z['r'],z['x'],b=z['b'])
        bigM = hlp.bigM_calc(Y,fmax,umax,dmax)
        log_optimization_consts(lossmin,lossterm,fmax,dmax,htheta,umin,umax,bigM=bigM)

        ### solve ####
        vars = fm.single_system(G,lossmin,lossterm,fmax,dmax,htheta,umin,umax,z,S,bigM)
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

##### SUBROUTINES ##########
def timestamp():
    return datetime.now().strftime('%d-%m-%Y_%H%M')

def timeparts(start,end):
    seconds = int(end-start)
    hrs = seconds//3600
    seconds -= hrs*3600
    minutes = seconds//60
    seconds -= minutes*60
    return hrs,minutes,seconds 

def log_input_samples(S,z):
    logging.info('------ Power Info -------')
    logging.info('Load:')
    try:
        logging.info('Actual Samples: %s', S['actual_vars_d'])
    except KeyError:
        logging.info('Actual Samples: N/A')
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
    try:
        logging.info('Actual Samples: %s', S['actual_vars_g'])
    except KeyError:
        logging.info('Actual Samples: N/A')
    logging.info('Total: %0.4f MW', sum(S['Pgmin']))
    logging.info('max: %0.4f MW', max(S['Pgmin']))
    if np.any(S['Pgmin'] != 0):
        logging.info('min (non 0): %0.4f MW', min(S['Pgmin'][S['Pgmin'] != 0]))
        logging.info('Avg (non 0): %0.4f WM', np.mean(S['Pgmin'][S['Pgmin'] != 0]))
        logging.info('Std (non 0): %0.4f WM', np.std(S['Pgmin'][S['Pgmin'] != 0]))

    logging.info('------Impedance Info----------')
    try:
        logging.info('Actual Samples: %s', z['actual_vars'])
    except KeyError:
        logging.info('Actual Samples: N/A')
    logging.info('Max         (r,x,b): %0.3g, %0.3g, %0.3g', max(z['r']), max(z['x']), max(z['b']))
    logging.info('Min         (r,x,b): %0.3g, %0.3g, %0.3g', min(z['r']), min(z['x']), min(z['b']))
    logging.info('Min (non 0) (r,x,b): %0.3g, %0.3g, %0.3g', min(z['r'][z['r'] != 0]), min(z['x'][z['x'] != 0]), min(z['b'][z['b'] != 0]))
    logging.info('Avg         (r,x,b): %0.3g, %0.3g, %0.3g', np.mean(z['r']), np.mean(z['x']), np.mean(z['b']))
    logging.info('Std         (r,x,b): %0.3g, %0.3g, %0.3g', np.std(z['r']), np.std(z['x']), np.std(z['b']))

def log_optimization_consts(lossmin,lossterm,fmax,dmax,htheta,umin,umax,bigM=None,thresholds=None):
    logging.info('-----Optimization Constants------')
    logging.info('Flow Max (P or Q)    [p.u]: %0.2f',fmax)
    logging.info('Angle Difference Max [rad]: %0.4f',dmax)
    logging.info('htheta: %d',htheta)
    logging.info('u min: %0.4f', umin)
    logging.info('u max: %0.4f', umax)
    logging.info('minimum losses: %d%%, terminating losses: %d%%', 100*lossmin, 100*lossterm)
    if bigM is not None:
        #logging.info('big M: %0.4g', bigM)
        for k,v in bigM.items():
            logging.info('big M%s: %0.4g', k, v)
    if thresholds is not None:
        for k,v in thresholds.items():
            logging.info('%s threshold: %0.3f',k,v)


def log_iteration_start(i,rho):
    logging.info('---------------------------------------')
    logging.info('ITERATION %d, rho=%0.2f', i,rho)

def log_iterations(s,pre=False):
    if pre:
        logging.info('Solvig Zone %d', s)
    else:
        logging.info("Solved with status %d, objective=%0.3f", s.m.status, s.m.objVal)

def log_iteration_summary(beta_bar,gamma_bar, ivals):
    logging.info('+++++++++++++++++++++++++++++++++++++++')
    logging.info("Iteration summary:")
    for k in ['gap','mean_diff', 'max_diff']:
        logging.info("%s: beta=%0.2f, gamma=%0.2f", k, ivals[k]['beta'], ivals[k]['gamma'])
    logging.info("Average Value statistics:")
    logging.info("max(beta_bar)=%0.2f, mean(beta_bar)=%0.2f, min(beta_bar)=%0.2f", max(beta_bar.values()), np.mean(list(beta_bar.values())), min(beta_bar.values()))
    logging.info("max(gamma_bar)=%0.2f, mean(gamma_bar)=%0.2f, min(gamma_bar)=%0.2f", max(gamma_bar.values()), np.mean(list(gamma_bar.values())), min(gamma_bar.values()))
    logging.info('+++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
