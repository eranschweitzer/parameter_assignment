from datetime import datetime
import time
import logging
import numpy as np

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
    try:
        logging.info('Actual Samples: %s', S['actual_vars_g'])
    except KeyError:
        logging.info('Actual Samples: N/A')
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
    logging.info('Shunt:')
    if S['shunt']['include_shunts']:
        logging.info('fraction  (g,b): %0.4f, %0.4f', S['shunt']['Gfrac'], S['shunt']['Bfrac'])
        logging.info('max [p.u] (g,b): %0.4f, %0.4f', S['shunt']['max'][0], S['shunt']['max'][1])
        logging.info('min [p.u] (g,b): %0.4f, %0.4f', S['shunt']['min'][0], S['shunt']['min'][1])
    else:
        logging.info('Shunts disabled')

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

def log_optimization_consts(lossmin,lossterm,fmax,dmax,htheta,umin,umax,bigM=None,thresholds=None):
    logging.info('-----Optimization Constants------')
    logging.info('Flow Max (P or Q)    [p.u]: %0.2f',fmax)
    logging.info('Angle Difference Max [rad]: %0.4f',dmax)
    logging.info('htheta: %d',htheta)
    logging.info('u min (v min): %0.4f (%0.4f)', umin, np.exp(umin))
    logging.info('u max (v max): %0.4f (%0.4f)', umax, np.exp(umax))
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
