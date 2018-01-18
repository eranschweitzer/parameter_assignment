import sys
from datetime import datetime
import time
import logging
import numpy as np
from helpers import model_status, progress 

def logging_setup(fname=None, level=logging.DEBUG, logger='root', ret=False):
    #FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
    FORMAT = '%(asctime)s: %(message)s'
    DATEFMT = '%H:%M:%S'
    #if logger is None:
    #    if fname is not None:
    #        logging.basicConfig(filename=fname,format=FORMAT,level=level,datefmt=DATEFMT)
    #    else:
    #        logging.basicConfig(format=FORMAT,level=level,datefmt=DATEFMT)
    #else:
    l = logging.getLogger(logger)
    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
    l.setLevel(level)
    if fname is not None:
        fileHandler = logging.FileHandler(fname, mode='a')
        fileHandler.setFormatter(formatter)
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        streamhandler.setLevel(logging.WARNING)
        l.addHandler(fileHandler)
        l.addHandler(streamhandler)
    else:
        streamhandler = logging.StreamHandler(sys.stdout)
        streamhandler.setFormatter(formatter)
    l.addHandler(streamhandler)
    l.propagate = False
    if ret:
        return l

def timestamp():
    return datetime.now().strftime('%d-%m-%Y_%H%M')

def timeparts(start,end):
    seconds = int(end-start)
    hrs = seconds//3600
    seconds -= hrs*3600
    minutes = seconds//60
    seconds -= minutes*60
    return hrs,minutes,seconds 

def log_total_run(start,end, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    hrs,minutes,seconds = timeparts(start,end)
    logger.info("Total time: %dhr %dmin %dsec",hrs,minutes,seconds)

def log_function_inputs(savename,fdata,logger=None, **kwargs):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('===========================')
    logger.info('Function Inputs')
    logger.info('===========================')
    logger.info("Saving to: %s",savename)
    logger.info("Topology data: %s", fdata)
    for k,v in kwargs.items():
        logger.info("%s: %s", k, v)

def log_topology(N,L,Nmax,Nmin, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('Number of buses: %d, Number of branches: %d, Nmax: %d, Nmin: %d',N,L,Nmax,Nmin)

def log_power_samples(S, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('------ Power Info -------')
    logger.info('Load:')
    try:
        logger.info('\tActual Samples: %s', S['actual_vars_d'])
    except KeyError:
        logger.info('\tActual Samples: N/A')
    logger.info('\tTotal: %0.4f MW, %0.4f MVar', sum(S['Pd']), sum(S['Qd']))
    logger.info('\tmax: %0.4f MW, %0.4f MVar', max(S['Pd']), max(S['Qd']))
    logger.info('\tmin (non 0): %0.4f MW, %0.4f MVar', min(S['Pd'][S['Pd'] != 0]), min(S['Qd'][S['Qd'] != 0]))
    logger.info('\tAvg: %0.4f WM, %0.4f MVar', np.mean(S['Pd']), np.mean(S['Qd']))
    logger.info('\tStd: %0.4f WM, %0.4f MVar', np.std(S['Pd']),  np.std(S['Qd']))
    logger.info('Gen Max:')
    try:
        logger.info('\tActual Samples: %s', S['actual_vars_g'])
    except KeyError:
        logger.info('\tActual Samples: N/A')
    logger.info('\tTotal: %0.4f MW, %0.4f MVar', sum(S['Pgmax']), sum(S['Qgmax']))
    logger.info('\tmax: %0.4f MW, %0.4f MVar', max(S['Pgmax']), max(S['Qgmax']))
    logger.info('\tmin (non 0): %0.4f MW, %0.4f MVar', min(S['Pgmax'][S['Pgmax'] != 0]), min(S['Qgmax'][S['Qgmax'] != 0]))
    logger.info('\tAvg (non 0): %0.4f WM, %0.4f MVar', np.mean(S['Pgmax'][S['Pgmax'] != 0]), np.mean(S['Qgmax'][S['Qgmax'] != 0]))
    logger.info('\tStd (non 0): %0.4f WM, %0.4f MVar', np.std(S['Pgmax'][S['Pgmax'] != 0]),  np.std(S['Qgmax'][S['Qgmax'] != 0]))
    logger.info('Gen Min:')
    logger.info('\tTotal: %0.4f MW', sum(S['Pgmin']))
    logger.info('\tmax: %0.4f MW', max(S['Pgmin']))
    if np.any(S['Pgmin'] != 0):
        logger.info('\tmin (non 0): %0.4f MW', min(S['Pgmin'][S['Pgmin'] != 0]))
        logger.info('\tAvg (non 0): %0.4f WM', np.mean(S['Pgmin'][S['Pgmin'] != 0]))
        logger.info('\tStd (non 0): %0.4f WM', np.std(S['Pgmin'][S['Pgmin'] != 0]))
    logger.info('Shunt:')
    if S['shunt']['include_shunts']:
        logger.info('\tfraction  (g,b): %0.4f, %0.4f', S['shunt']['Gfrac'], S['shunt']['Bfrac'])
        logger.info('\tmax [p.u] (g,b): %0.4f, %0.4f', S['shunt']['max'][0], S['shunt']['max'][1])
        logger.info('\tmin [p.u] (g,b): %0.4f, %0.4f', S['shunt']['min'][0], S['shunt']['min'][1])
    else:
        logger.info('\tShunts disabled')

def log_branch_samples(z, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('------Impedance Info----------')
    try:
        logger.info('Actual Samples: %s', z['actual_vars'])
    except KeyError:
        logger.info('Actual Samples: N/A')
    logger.info('Max         (r,x,b): %0.3g, %0.3g, %0.3g', max(z['r']), max(z['x']), max(z['b']))
    logger.info('Min         (r,x,b): %0.3g, %0.3g, %0.3g', min(z['r']), min(z['x']), min(z['b']))
    logger.info('Min (non 0) (r,x,b): %0.3g, %0.3g, %0.3g', min(z['r'][z['r'] != 0]), min(z['x'][z['x'] != 0]), min(z['b'][z['b'] != 0]))
    logger.info('Avg         (r,x,b): %0.3g, %0.3g, %0.3g', np.mean(z['r']), np.mean(z['x']), np.mean(z['b']))
    logger.info('Std         (r,x,b): %0.3g, %0.3g, %0.3g', np.std(z['r']), np.std(z['x']), np.std(z['b']))
    logger.info('# off-nominal tap  : %d', sum(z['tap']!=1))
    logger.info('# phase-shifters   : %d', sum(z['shift']!=0))
    logger.info('ratings (min,max)  : %0.3g, %0.3g', min(z['rate']), max(z['rate']))

def log_input_samples(S=None,z=None, logger=None):
    if S is not None:
        log_power_samples(S, logger=logger)
    if z is not None:
        log_branch_samples(z, logger=logger)


#def log_optimization_consts(lossmin,lossterm,fmax,dmax,htheta,umin,umax,bigM=None,thresholds=None):
def log_optimization_consts(C, bigM=None, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('-----Optimization Constants------')
    logger.info('Flow Max (P or Q)    [p.u]: %0.2f',C['fmax'])
    logger.info('Angle Difference Max [rad (deg)]: %0.4f (%0.2f)',C['dmax'], C['dmax']*180/np.pi)
    logger.info('htheta (#, err): %d, %0.2g%%',C['htheta'], C['phi_err']*100)
    logger.info('u min (v min): %0.4f (%0.4f)', C['umin'], np.exp(C['umin']))
    logger.info('u max (v max): %0.4f (%0.4f)', C['umax'], np.exp(C['umax']))
    logger.info('minimum losses: %d%%, terminating losses: %d%%', 100*C['lossmin'], 100*C['lossterm'])
    if 'aug_relax' in C:
        logger.info('Using Polyheral relaxation of augmented Lagrangian. Max Error set to %0.2g%%', C['beta2_err']*100)
    if bigM is not None:
        #logger.info('big M: %0.4g', bigM)
        for k,v in bigM.items():
            logger.info('big M%s: %0.4g', k, v)
    if 'thresholds' in C:
        logger.info('Thresholds:')
        for k,v in C['thresholds'].items():
            logger.info('\t%s threshold: %0.3f',k,v)
    if 'ea' in C:
        logger.info('EA parameters:')
        for k,v in C['ea'].items():
            logger.info('\t%s: %0.2f', k, v)
    if 'solve_kwargs' in C:
        logger.info('kwargs for zones:')
        for k,v in C['solve_kwargs'].items():
            logger.info('\t%s: %s', k, v)


def log_iteration_start(i,rho, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('---------------------------------------')
    logger.info('ITERATION %d, rho=%0.2f', i,rho)

def log_iterations(s,pre=False,print_boundary=False, logger=None, zone=None):
    if logger is None:
        logger = logging.getLogger('root')
    if pre:
        logger.info('##### Solving Zone %d ########', s)
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
        if zone is None:
            logger.info("Solved with status %s (%d), objective=%0.3f", model_status(s.m), s.m.status, s.m.objVal)
        else:
            logger.info("(zone %d) Solved with status %s (%d), objective=%0.3f", zone, model_status(s.m), s.m.status, s.m.objVal)
        logger.info("\tgeneration: %0.3g MW, load: %0.3g MW, import: %0.3g MW, export: %0.3g MW", Pg*100, s.m._pload*100, in_sum*100, out_sum*100)
        logger.info("\tgeneration: %0.3g MVAr, load: %0.3g MVar, import: %0.3g MVAr, export: %0.3g MVAr", Qg*100, Qd*100, in_sum2*100, out_sum2*100)
        logger.info("\tphi error (max, min): %0.3g, %0.3g", max(phi_err), min(phi_err))
        if auglag_err is not None:
            logger.info("\tAug. Lagrangian polyhedral relaxation error (beta, gamma) max/min: %0.3g/%0.3g, %0.3g/%0.3g", \
                    max(auglag_err['beta'].values()), min(auglag_err['beta'].values()), max(auglag_err['gamma'].values()), min(auglag_err['gamma'].values()) )
        if print_boundary:
            logger.info("Boundary flows (id: beta, gamma):")
            beta = {i:s.beta[i].X for i in s.beta}
            gamma= {i:s.gamma[i].X for i in s.gamma}
            ids = list(beta.keys())
            txt = ""
            for i,id in enumerate(ids):
                txt += "%d:(%0.3f, %0.3f) " %(id, beta[id], gamma[id])
                if (i % 7) == 7:
                    logger.info("%s", txt)
                    txt = ""
            if txt != "":
                logger.info("%s", txt)

def log_optimization_init(i, T, res=0.1, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    v = progress(i,T, res=res)
    if v is not None:
        logger.info('%0.1f%% Models initialized', v*100)

def log_termination(msg, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('===============================')
    logger.info('TERMINATION CRITERIA SATISFIED')
    for part in msg:
        logger.info("%s", part)
    logger.info('===============================')

def log_iteration_summary(beta_bar,gamma_bar, ivals, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('+++++++++++++++++++++++++++++++++++++++')
    logger.info("Iteration summary:")
    for k in ['gap','mean_diff', 'max_diff']:
        logger.info("%s: beta=%0.2f, gamma=%0.2f", k, ivals[k]['beta'], ivals[k]['gamma'])
    logger.info("Average Value statistics:")
    logger.info("max(beta_bar)=%0.2f, mean(beta_bar)=%0.2f, min(beta_bar)=%0.2f", max(beta_bar.values()), np.mean(list(beta_bar.values())), min(beta_bar.values()))
    logger.info("max(gamma_bar)=%0.2f, mean(gamma_bar)=%0.2f, min(gamma_bar)=%0.2f", max(gamma_bar.values()), np.mean(list(gamma_bar.values())), min(gamma_bar.values()))
    logger.info('+++++++++++++++++++++++++++++++++++++++')

def log_single_system(s, start=True, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    if start:
        logger.info('Solving single system')
    else:
        log_iterations(s, logger=logger)
    
def log_generation(i, Psi, start=True, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('=========================================================')
    if start:
        logger.info('Start of Generation %d', i)
    else:
        logger.info('Generation objectives in range %0.3f -- %0.3f', Psi[0].f, Psi[-1].f)
    logger.info('=========================================================')

def log_individual(i, start=True, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    if start:
        logger.info('-----------------')
        logger.info('Individual %d', i)
        logger.info('-----------------')

def log_zones_split(pre=True, num=None, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    if pre:
        logger.info('Splitting graph into zones')
    else:
        logger.info('%d zones created', num)

def log_zone_init(i, H, ebound, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('-------------------')
    logger.info('Initializing Zone %d: %d nodes, %d edges, %d boundary edges', i, H.number_of_nodes(), H.number_of_edges(), len(ebound[1]))
    logger.info('-------------------')

def log_callback(model, solcnt, in_sum, out_sum, Pg, criteria, phiconst, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('(zone %d) Current solution: solcnt: %d, solmin: %d, sum(beta_in)=%0.2f, sum(beta_out)=%0.2f, sum(Pg)=%0.2f, sum(load)=%0.2f, criteria=%0.3g, phiconst=%d', model._zone, solcnt, model._solmin, in_sum, out_sum, Pg, model._pload, criteria, phiconst)

def log_calback_terminate(where, why, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    logger.info('      terminating in %s due to %s', where, why)
