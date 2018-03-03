import pickle
import numpy as np
import networkx as nx
import pandas as pd
import helpers as hlp
import logging

#FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
#logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

def rescheck(data, G=None, maps=None, ebound_map=None, logger=None):
    if logger is None:
        logger = logging.getLogger('root')
    if G is None:
        G = data['G']
    N = G.number_of_nodes()
    L = G.number_of_edges()
    Y = hlp.Yparts(data['r'],data['x'],b=data['b'],tau=data['tap'],phi=data['shift'])

    Pf = np.empty(L)
    Qf = np.empty(L)
    Pt = np.empty(L)
    Qt = np.empty(L)
    delta = np.empty(L)
    for _n1,_n2,_l in G.edges_iter(data='id'):
        n1,n2,l = map_values(maps,_n1,_n2,_l)
        Pf[l] =  Y['gff'][l]*(1 + data['u'][n1]) + Y['gft'][l]*(1 - data['phi'][l] + data['u'][n2]) + Y['bft'][l]*(data['theta'][n1] - data['theta'][n2])
        Qf[l] = -Y['bff'][l]*(1 + data['u'][n1]) - Y['bft'][l]*(1 - data['phi'][l] + data['u'][n2]) + Y['gft'][l]*(data['theta'][n1] - data['theta'][n2])
        Pt[l] =  Y['gtt'][l]*(1 + data['u'][n2]) + Y['gtf'][l]*(1 - data['phi'][l] + data['u'][n1]) - Y['btf'][l]*(data['theta'][n1] - data['theta'][n2])
        Qt[l] = -Y['btt'][l]*(1 + data['u'][n2]) - Y['btf'][l]*(1 - data['phi'][l] + data['u'][n1]) - Y['gtf'][l]*(data['theta'][n1] - data['theta'][n2])
        delta[l] = data['theta'][n1] - data['theta'][n2]

    phierr = 0.5*delta**2 - data['phi']
    phierr_idx = np.argmax(np.abs(phierr))

    Pfn = np.zeros(N)
    Qfn = np.zeros(N)
    Ptn = np.zeros(N)
    Qtn = np.zeros(N)
    for _n1,_n2,_l in G.edges_iter(data='id'):
        n1,n2,l = map_values(maps,_n1,_n2,_l)
        Pfn[n1] += data['Pf'][l]
        Ptn[n2] += data['Pt'][l]
        Qfn[n1] += data['Qf'][l]
        Qtn[n2] += data['Qt'][l]
    
    beta = np.zeros(N)
    gamma= np.zeros(N)
    if ebound_map is not None:
        for i in range(N):
            beta[i]  = sum( data['beta'][l]  for l in ebound_map['in'].get( maps['rnmap'][i],[]) ) - \
                       sum( data['beta'][l]  for l in ebound_map['out'].get(maps['rnmap'][i],[]) ) 
            gamma[i] = sum( data['gamma'][l] for l in ebound_map['in'].get( maps['rnmap'][i],[]) ) - \
                       sum( data['gamma'][l] for l in ebound_map['out'].get(maps['rnmap'][i],[]) ) 
        if 'beta_p' in data:
            beta_abs  = np.empty(len(data['beta']))
            gamma_abs = np.empty(len(data['gamma']))
            for i,l in enumerate(data['beta']):
                beta_abs[i]  = data['beta'][l]  - data['beta_p'][l]  + data['beta_n'][l] 
                gamma_abs[i] = data['gamma'][l] - data['gamma_p'][l] + data['gamma_n'][l]
        if 'beta2' in data:
            aug_lag_error = {'beta': {}, 'gamma': {}}
            for l in data['beta2']:
                aug_lag_error['beta'][l]  = (data['beta'][l] - data['beta_bar'][l])**2 - data['beta2'][l]
                aug_lag_error['gamma'][l] = (data['gamma'][l] - data['gamma_bar'][l])**2 - data['gamma2'][l]

    if 'GS' not in data:
        data['GS'] = 0
    if 'BS' not in data:
        data['BS'] = 0
    else:
        bs_abs = np.abs(data['BS']) - (data['BSp'] + data['BSn'])
    balance = {}
    balance['P'] = data['Pg'] - data['GS'] - data['Pd'] - Pfn - Ptn + beta
    balance['Q'] = data['Qg'] + data['BS'] - data['Qd'] - Qfn - Qtn + gamma

    Plim = {}
    Plim['max'] = (data['Pg']*100 - data['Pgmax']) > 1e-6
    Plim['min'] = (data['Pgmin'] - data['Pg']*100) > 1e-6
    Qlim = {}
    Qlim['max'] = ( data['Qg']*100 - data['Qgmax'])> 1e-6
    Qlim['min'] = (-data['Qgmax'] - data['Qg']*100)> 1e-6

    ### slacks
    for l in ['sf','sd']:
        if l not in data:
            data[l] = np.zeros(L)
    if 'su' not in data:
        data['su'] = np.zeros(N)
    Flim = {}
    for fl in ['Pf', 'Qf', 'Pt', 'Qt']:
        Flim[fl] = (data[fl] < -data['rate'] - data['sf'] - 1e-6 ) & (data[fl] > data['rate'] + data['sf'] + 1e-6)
        #Flim[fl] = np.abs(data[fl] + data['sfn'] - data['sfp']) - data['rate']
    
    ### sil
    sil = hlp.calc_sil(**data)
    Sf  = np.sqrt(Pf**2 + Qf**2)
    silavg = np.mean([Sf[i]/sil[i] for i in sil])

    ### Qabs
    abs_err = {}
    for fl in ['Qf', 'Qt', 'Pf']:
        if fl+'abs' in data:
            abs_err[ fl+'abs'] = np.abs(data[fl+'abs'] - np.abs(data['Qf']))
        else:
            abs_err[fl+'abs'] = None
    #if np.any(Flim['Qf']) :
    #    import ipdb; ipdb.set_trace()

    
    logger.info('+++++++++++++++++++++++++++++++')
    logger.info('Solution Verification')
    logger.info('+++++++++++++++++++++++++++++++')
    logger.info('Total load: %0.4f MW, Total gen: %0.4f MW' ,100*sum(data['Pd']),100*sum(data['Pg']))
    logger.info('Total load: %0.4f MVar, Total gen: %0.4f MVar' ,100*sum(data['Qd']),100*sum(data['Qg']))
    if 'Qgslack' in data:
        logger.info('Slack to make sum(Qg) > 0: %0.3g', data['Qgslack'])
    if ebound_map is not None:
        logger.info('Total imports/exports (+/-): %0.4f MW, %0.4f MVAr', 100*beta.sum(), 100*gamma.sum())
    logger.info('Losses [MW]: %0.3g, %0.3f%%' ,100*(sum(data['Pg'] + beta - data['Pd'])), 100*sum(data['Pg'] + beta - data['Pd'])/sum(data['Pd']))
    logger.info('Maximum |Pf|: %0.4f MW,   Maximum |Pt|: %0.4f MW' ,100*max(np.abs(data['Pf'])), 100*max(np.abs(data['Pt'])))
    logger.info('Maximum |Qf|: %0.4f MVar, Maximum |Qt|: %0.4f Mvar' ,100*max(np.abs(data['Qf'])), 100*max(np.abs(data['Qt'])))
    logger.info('Maximum Pf error: %0.3g' ,max(np.abs(data['Pf'] - Pf)))
    logger.info('Maximum Qf error: %0.3g' ,max(np.abs(data['Qf'] - Qf)))
    logger.info('Maximum Pt error: %0.3g' ,max(np.abs(data['Pt'] - Pt)))
    logger.info('Maximum Qt error: %0.3g' ,max(np.abs(data['Qt'] - Qt)))
    logger.info('Maximum |P balance|: %0.3g' , max(np.abs(balance['P'])))
    logger.info('Maximum |Q balance|: %0.3g' , max(np.abs(balance['Q'])))
    logger.info('Maximum angle difference: %0.2f deg' ,max(np.abs(delta))*180/np.pi)
    logger.info('Maximum differentce between phi and delta^2/2 (diff, delta, phi): %0.3g, %0.3g, %0.3g' ,max(np.abs(phierr)), delta[phierr_idx], data['phi'][phierr_idx])
    logger.info('# of occurances where delta^2/2 - phi < 0: %d', sum(phierr < -1e-5))
    logger.info('Maximum v: %0.3f, Minimum v: %0.3f' , max(np.exp(data['u'])), min(np.exp(data['u'])))
    logger.info('Pmax violations: %d, Pmin violations: %d' ,sum(Plim['max']), sum(Plim['min']))
    logger.info('Qmax violations: %d, Qmin violations: %d' ,sum(Qlim['max']), sum(Qlim['min']))
    logger.info*('Average Fraction of SIL loading: %0.3f', silavg)
    logger.info('Flow limits:')
    logger.info('Sum Real Power Violations (including slacks) (Pf, Pt): %d, %d', sum(Flim['Pf']), sum(Flim['Pt']))
    logger.info('Sum Reactive Power Violations (including slacks) (Qf, Qt): %d, %d', sum(Flim['Qf']), sum(Flim['Qt']))
    logger.info('Flow Magnitude Vars:')
    for fl in ['Qfabs', 'Qtabs', 'Pfabs']:
        if abs_err[fl] is not None:
            logger.info('\tMagnitude of Maximum error %s: %0.3g', fl, max(abs_err[fl]))
        else:
            logger.info('\t%s: N/A', fl)
    logger.info('Shunts:')
    if not isinstance(data['GS'],int):
        logger.info('\tNumber of Gsh: %d', sum(data['GS'] != 0))
        if np.any(data['GS'] != 0):
            logger.info('\tGS (min, max): %0.3f, %0.3f', min(data['GS'][data['GS'] != 0]), max(data['GS'][data['GS'] != 0]))
    else:
        logger.info('\tNo Gsh')
    if not isinstance(data['BS'],int):
        logger.info('\tNumber of Bsh: %d', sum(data['BS'] != 0))
        if np.any(data['BS'] != 0):
            logger.info('\tBS (min, max): %0.3f, %0.3f', min(data['BS'][data['BS'] != 0]), max(data['BS'][data['BS'] != 0]))
            logger.info('\tMax error in |BS| constraints: %0.3g', max(abs(bs_abs)))
    else:
        logger.info('\tNo Bsh')
    logger.info('Slacks:')
    logger.info('\t (Max, sum) Flow Slack: %0.3f, %0.3f', max(data['sf']), sum(data['sf']))
    logger.info('\t (Max, sum) u Slack: %0.3f, %0.3f', max(data['su']), sum(data['su']))
    logger.info('\t (Max, sum) delta Slack: %0.3f, %0.3f', max(data['sd']), sum(data['sd']))
    if 'beta_p' in data:
        logger.info('Boundary Value Test:')
        logger.info('\tbeta - beta_p + beta_n (max, min): %0.3g, %0.3g', max(beta_abs), min(beta_abs))
        logger.info('\tgamma - gamma_p + gamma_n (max, min): %0.3g, %0.3g', max(gamma_abs), min(gamma_abs))
    if 'beta2' in data:
        logger.info('Polyhedral Relaxation of Augmented Lagrangian Test:')
        logger.info('\tbeta (max, min): %0.3g, %0.3g', max(aug_lag_error['beta'].values()), min(aug_lag_error['beta'].values()) )
        logger.info('\tgamma (max, min): %0.3g, %0.3g', max(aug_lag_error['gamma'].values()), min(aug_lag_error['gamma'].values()) )
    
def map_values(maps,n1,n2,l):
    if maps is None:
        return (n1,n2,l)
    else:
        return (maps['nmap'][n1], maps['nmap'][n2], maps['lmap'][l])

if __name__ == '__main__':
    import sys
    fname = sys.argv[1] 
    data = pickle.load(open(fname,'rb'))
    rescheck(data)
