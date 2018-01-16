from datetime import datetime
import time
import logging
import numpy as np
from helpers import model_status, progress 

FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

def timestamp():
    return datetime.now().strftime('%d-%m-%Y_%H%M')

def timeparts(start,end):
    seconds = int(end-start)
    hrs = seconds//3600
    seconds -= hrs*3600
    minutes = seconds//60
    seconds -= minutes*60
    return hrs,minutes,seconds 

def log_total_run(start,end):
    hrs,minutes,seconds = timeparts(start,end)
    logging.info("Total time: %dhr %dmin %dsec",hrs,minutes,seconds)

def log_function_inputs(savename,fdata,**kwargs):
    logging.info('===========================')
    logging.info('Function Inputs')
    logging.info('===========================')
    logging.info("Saving to: %s",savename)
    logging.info("Topology data: %s", fdata)
    for k,v in kwargs.items():
        logging.info("%s: %s", k, v)

def log_topology(N,L,Nmax,Nmin):
    logging.info('Number of buses: %d, Number of branches: %d, Nmax: %d, Nmin: %d',N,L,Nmax,Nmin)

def log_power_samples(S):
    logging.info('------ Power Info -------')
    logging.info('Load:')
    try:
        logging.info('\tActual Samples: %s', S['actual_vars_d'])
    except KeyError:
        logging.info('\tActual Samples: N/A')
    logging.info('\tTotal: %0.4f MW, %0.4f MVar', sum(S['Pd']), sum(S['Qd']))
    logging.info('\tmax: %0.4f MW, %0.4f MVar', max(S['Pd']), max(S['Qd']))
    logging.info('\tmin (non 0): %0.4f MW, %0.4f MVar', min(S['Pd'][S['Pd'] != 0]), min(S['Qd'][S['Qd'] != 0]))
    logging.info('\tAvg: %0.4f WM, %0.4f MVar', np.mean(S['Pd']), np.mean(S['Qd']))
    logging.info('\tStd: %0.4f WM, %0.4f MVar', np.std(S['Pd']),  np.std(S['Qd']))
    logging.info('Gen Max:')
    try:
        logging.info('\tActual Samples: %s', S['actual_vars_g'])
    except KeyError:
        logging.info('\tActual Samples: N/A')
    logging.info('\tTotal: %0.4f MW, %0.4f MVar', sum(S['Pgmax']), sum(S['Qgmax']))
    logging.info('\tmax: %0.4f MW, %0.4f MVar', max(S['Pgmax']), max(S['Qgmax']))
    logging.info('\tmin (non 0): %0.4f MW, %0.4f MVar', min(S['Pgmax'][S['Pgmax'] != 0]), min(S['Qgmax'][S['Qgmax'] != 0]))
    logging.info('\tAvg (non 0): %0.4f WM, %0.4f MVar', np.mean(S['Pgmax'][S['Pgmax'] != 0]), np.mean(S['Qgmax'][S['Qgmax'] != 0]))
    logging.info('\tStd (non 0): %0.4f WM, %0.4f MVar', np.std(S['Pgmax'][S['Pgmax'] != 0]),  np.std(S['Qgmax'][S['Qgmax'] != 0]))
    logging.info('Gen Min:')
    logging.info('\tTotal: %0.4f MW', sum(S['Pgmin']))
    logging.info('\tmax: %0.4f MW', max(S['Pgmin']))
    if np.any(S['Pgmin'] != 0):
        logging.info('\tmin (non 0): %0.4f MW', min(S['Pgmin'][S['Pgmin'] != 0]))
        logging.info('\tAvg (non 0): %0.4f WM', np.mean(S['Pgmin'][S['Pgmin'] != 0]))
        logging.info('\tStd (non 0): %0.4f WM', np.std(S['Pgmin'][S['Pgmin'] != 0]))
    logging.info('Shunt:')
    if S['shunt']['include_shunts']:
        logging.info('\tfraction  (g,b): %0.4f, %0.4f', S['shunt']['Gfrac'], S['shunt']['Bfrac'])
        logging.info('\tmax [p.u] (g,b): %0.4f, %0.4f', S['shunt']['max'][0], S['shunt']['max'][1])
        logging.info('\tmin [p.u] (g,b): %0.4f, %0.4f', S['shunt']['min'][0], S['shunt']['min'][1])
    else:
        logging.info('\tShunts disabled')

def log_branch_samples(z):
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
    logging.info('# off-nominal tap  : %d', sum(z['tap']!=1))
    logging.info('# phase-shifters   : %d', sum(z['shift']!=0))
    logging.info('ratings (min,max)  : %0.3g, %0.3g', min(z['rate']), max(z['rate']))

def log_input_samples(S=None,z=None):
    if S is not None:
        log_power_samples(S)
    if z is not None:
        log_branch_samples(z)


#def log_optimization_consts(lossmin,lossterm,fmax,dmax,htheta,umin,umax,bigM=None,thresholds=None):
def log_optimization_consts(C, bigM=None):
    logging.info('-----Optimization Constants------')
    logging.info('Flow Max (P or Q)    [p.u]: %0.2f',C['fmax'])
    logging.info('Angle Difference Max [rad (deg)]: %0.4f (%0.2f)',C['dmax'], C['dmax']*180/np.pi)
    logging.info('htheta (#, err): %d, %0.2g%%',C['htheta'], C['phi_err']*100)
    logging.info('u min (v min): %0.4f (%0.4f)', C['umin'], np.exp(C['umin']))
    logging.info('u max (v max): %0.4f (%0.4f)', C['umax'], np.exp(C['umax']))
    logging.info('minimum losses: %d%%, terminating losses: %d%%', 100*C['lossmin'], 100*C['lossterm'])
    if 'aug_relax' in C:
        logging.info('Using Polyheral relaxation of augmented Lagrangian. Max Error set to %0.2g%%', C['beta2_err']*100)
    if bigM is not None:
        #logging.info('big M: %0.4g', bigM)
        for k,v in bigM.items():
            logging.info('big M%s: %0.4g', k, v)
    if 'thresholds' in C:
        logging.info('Thresholds:')
        for k,v in C['thresholds'].items():
            logging.info('\t%s threshold: %0.3f',k,v)
    if 'ea' in C:
        logging.info('EA parameters:')
        for k,v in C['ea'].items():
            logging.info('\t%s: %0.2f', k, v)
    if 'solve_kwargs' in C:
        logging.info('kwargs for zones:')
        for k,v in C['solve_kwargs'].items():
            logging.info('\t%s: %s', k, v)


def log_iteration_start(i,rho):
    logging.info('---------------------------------------')
    logging.info('ITERATION %d, rho=%0.2f', i,rho)

def log_iterations(s,pre=False,print_boundary=False):
    if pre:
        logging.info('##### Solving Zone %d ########', s)
    else:
        in_sum   = sum(s.beta[i].X for _,j in s.m._ebound_map['in'].items()  for i in j)
        out_sum  = sum(s.beta[i].X for _,j in s.m._ebound_map['out'].items() for i in j)
        in_sum2  = sum(s.gamma[i].X for _,j in s.m._ebound_map['in'].items()  for i in j)
        out_sum2 = sum(s.gamma[i].X for _,j in s.m._ebound_map['out'].items() for i in j)
        Pg       = sum(s.Pg[i].X for i in s.Pg )
        Qg       = sum(s.Qg[i].X for i in s.Qg )
        Qd       = sum(s.Qd[i].X for i in s.Qd )
        Losses   = (Pg - s.m._pload + in_sum - out_sum)/(Pg + in_sum - out_sum)
        phi_err  = s.phi_error()
        try:
            auglag_err = s.auglag_error()
        except:
            auglag_err = None
        logging.info("Solved with status %s (%d), objective=%0.3f", model_status(s.m), s.m.status, s.m.objVal)
        logging.info("\tgeneration: %0.3g MW, load: %0.3g MW, import: %0.3g MW, export: %0.3g MW", Pg*100, s.m._pload*100, in_sum*100, out_sum*100)
        logging.info("\tgeneration: %0.3g MVAr, load: %0.3g MVar, import: %0.3g MVAr, export: %0.3g MVAr", Qg*100, Qd*100, in_sum2*100, out_sum2*100)
        logging.info("\tphi error (max, min): %0.3g, %0.3g", max(phi_err), min(phi_err))
        if auglag_err is not None:
            logging.info("\tAug. Lagrangian polyhedral relaxation error (beta, gamma) max/min: %0.3g/%0.3g, %0.3g/%0.3g", \
                    max(auglag_err['beta'].values()), min(auglag_err['beta'].values()), max(auglag_err['gamma'].values()), min(auglag_err['gamma'].values()) )
        if print_boundary:
            logging.info("Boundary flows (id: beta, gamma):")
            beta = {i:s.beta[i].X for i in s.beta}
            gamma= {i:s.gamma[i].X for i in s.gamma}
            ids = list(beta.keys())
            txt = ""
            for i,id in enumerate(ids):
                txt += "%d:(%0.3f, %0.3f) " %(id, beta[id], gamma[id])
                if (i % 7) == 7:
                    logging.info("%s", txt)
                    txt = ""
            if txt != "":
                logging.info("%s", txt)

def log_optimization_init(i, T, res=0.1):
    v = progress(i,T, res=res)
    if v is not None:
        logging.info('%0.1f%% Models initialized', v*100)

def log_termination(msg):
    logging.info('===============================')
    logging.info('TERMINATION CRITERIA SATISFIED')
    for part in msg:
        logging.info("%s", part)
    logging.info('===============================')

def log_iteration_summary(beta_bar,gamma_bar, ivals):
    logging.info('+++++++++++++++++++++++++++++++++++++++')
    logging.info("Iteration summary:")
    for k in ['gap','mean_diff', 'max_diff']:
        logging.info("%s: beta=%0.2f, gamma=%0.2f", k, ivals[k]['beta'], ivals[k]['gamma'])
    logging.info("Average Value statistics:")
    logging.info("max(beta_bar)=%0.2f, mean(beta_bar)=%0.2f, min(beta_bar)=%0.2f", max(beta_bar.values()), np.mean(list(beta_bar.values())), min(beta_bar.values()))
    logging.info("max(gamma_bar)=%0.2f, mean(gamma_bar)=%0.2f, min(gamma_bar)=%0.2f", max(gamma_bar.values()), np.mean(list(gamma_bar.values())), min(gamma_bar.values()))
    logging.info('+++++++++++++++++++++++++++++++++++++++')

def log_single_system(s, start=True):
    if start:
        logging.info('Solving single system')
    else:
        log_iterations(s)
    
def log_generation(i, Psi, start=True):
    logging.info('=========================================================')
    if start:
        logging.info('Start of Generation %d', i)
    else:
        logging.info('Generation objectives in range %0.3f -- %0.3f', Psi[0].f, Psi[-1].f)
    logging.info('=========================================================')

def log_individual(i, start=True):
    if start:
        logging.info('-----------------')
        logging.info('Individual %d', i)
        logging.info('-----------------')
