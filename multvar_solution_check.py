import pickle
import numpy as np
import networkx as nx
import pandas as pd
import helpers as hlp
import logging

FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

def rescheck(data, G=None, maps=None, ebound_map=None):
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
    
    #if np.any(Flim['Qf']) :
    #    import ipdb; ipdb.set_trace()

    
    logging.info('+++++++++++++++++++++++++++++++')
    logging.info('Solution Verification')
    logging.info('+++++++++++++++++++++++++++++++')
    logging.info('Total load: %0.4f MW, Total gen: %0.4f MW' ,100*sum(data['Pd']),100*sum(data['Pg']))
    logging.info('Total load: %0.4f MVar, Total gen: %0.4f MVar' ,100*sum(data['Qd']),100*sum(data['Qg']))
    if ebound_map is not None:
        logging.info('Total imports/exports (+/-): %0.4f MW, %0.4f MVAr', 100*beta.sum(), 100*gamma.sum())
    logging.info('Losses [MW]: %0.3g, %0.3f%%' ,100*(sum(data['Pg'] + beta - data['Pd'])), 100*sum(data['Pg'] + beta - data['Pd'])/sum(data['Pd']))
    logging.info('Maximum |Pf|: %0.4f MW,   Maximum |Pt|: %0.4f MW' ,100*max(np.abs(data['Pf'])), 100*max(np.abs(data['Pt'])))
    logging.info('Maximum |Qf|: %0.4f MVar, Maximum |Qt|: %0.4f Mvar' ,100*max(np.abs(data['Qf'])), 100*max(np.abs(data['Qt'])))
    logging.info('Maximum Pf error: %0.3g' ,max(np.abs(data['Pf'] - Pf)))
    logging.info('Maximum Qf error: %0.3g' ,max(np.abs(data['Qf'] - Qf)))
    logging.info('Maximum Pt error: %0.3g' ,max(np.abs(data['Pt'] - Pt)))
    logging.info('Maximum Qt error: %0.3g' ,max(np.abs(data['Qt'] - Qt)))
    logging.info('Maximum |P balance|: %0.3g' , max(np.abs(balance['P'])))
    logging.info('Maximum |Q balance|: %0.3g' , max(np.abs(balance['Q'])))
    logging.info('Maximum angle difference: %0.2f deg' ,max(np.abs(delta))*180/np.pi)
    logging.info('Maximum differentce between phi and delta^2/2: %0.3g' ,max(np.abs(phierr)))
    logging.info('Maximum v: %0.3f, Minimum v: %0.3f' , max(np.exp(data['u'])), min(np.exp(data['u'])))
    logging.info('Pmax violations: %d, Pmin violations: %d' ,sum(Plim['max']), sum(Plim['min']))
    logging.info('Qmax violations: %d, Qmin violations: %d' ,sum(Qlim['max']), sum(Qlim['min']))
    logging.info('Flow limits:')
    logging.info('Sum Real Power Violations (including slacks) (Pf, Pt): %d, %d', sum(Flim['Pf']), sum(Flim['Pt']))
    logging.info('Sum Reactive Power Violations (including slacks) (Qf, Qt): %d, %d', sum(Flim['Qf']), sum(Flim['Qt']))
    logging.info('Shunts:')
    if not isinstance(data['GS'],int):
        logging.info('\tNumber of Gsh: %d', sum(data['GS'] != 0))
        if np.any(data['GS'] != 0):
            logging.info('\tGS (min, max): %0.3f, %0.3f', min(data['GS'][data['GS'] != 0]), max(data['GS'][data['GS'] != 0]))
    else:
        logging.info('\tNo Gsh')
    if not isinstance(data['BS'],int):
        logging.info('\tNumber of Bsh: %d', sum(data['BS'] != 0))
        if np.any(data['BS'] != 0):
            logging.info('\tBS (min, max): %0.3f, %0.3f', min(data['BS'][data['BS'] != 0]), max(data['BS'][data['BS'] != 0]))
    else:
        logging.info('\tNo Bsh')
    logging.info('Slacks:')
    logging.info('\t Max Flow Slack: %0.3f', max(data['sf']))
    logging.info('\t Max u Slack: %0.3f', max(data['su']))
    logging.info('\t Max delta Slack: %0.3f', max(data['sd']))
    if 'beta_p' in data:
        logging.info('Boundary Value Test:')
        logging.info('\tbeta - beta_p + beta_n (max, min): %0.3g, %0.3g', max(beta_abs), min(beta_abs))
        logging.info('\tgamma - gamma_p + gamma_n (max, min): %0.3g, %0.3g', max(gamma_abs), min(gamma_abs))
    if 'beta2' in data:
        logging.info('Polyhedral Relaxation of Augmented Lagrangian Test:')
        logging.info('\tbeta (max, min): %0.3g, %0.3g', max(aug_lag_error['beta'].values()), min(aug_lag_error['beta'].values()) )
        logging.info('\tgamma (max, min): %0.3g, %0.3g', max(aug_lag_error['gamma'].values()), min(aug_lag_error['gamma'].values()) )
    
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
