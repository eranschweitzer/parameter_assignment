import pickle
import numpy as np
import networkx as nx
import pandas as pd
import helpers as hlp

def rescheck(data):
    
    N = data['G'].number_of_nodes()
    L = data['G'].number_of_edges()
    Y = hlp.Yparts(data['r'],data['x'],b=data['b'])

    Pf = np.empty(L)
    Qf = np.empty(L)
    Pt = np.empty(L)
    Qt = np.empty(L)
    delta = np.empty(L)
    for n1,n2,l in data['G'].edges_iter(data='id'):
        Pf[l] =  Y['gff'][l]*(1 + data['u'][n1]) + Y['gft'][l]*(1 - data['phi'][l] + data['u'][n2]) - Y['bft'][l]*(data['theta'][n2] - data['theta'][n1])
        Qf[l] = -Y['bff'][l]*(1 + data['u'][n1]) - Y['bft'][l]*(1 + data['phi'][l] + data['u'][n2]) + Y['gft'][l]*(data['theta'][n2] - data['theta'][n1])
        Pt[l] =  Y['gtt'][l]*(1 + data['u'][n2]) + Y['gtf'][l]*(1 - data['phi'][l] + data['u'][n1]) - Y['btf'][l]*(data['theta'][n1] - data['theta'][n2])
        Qt[l] = -Y['btt'][l]*(1 + data['u'][n2]) - Y['btf'][l]*(1 + data['phi'][l] + data['u'][n1]) + Y['gtf'][l]*(data['theta'][n1] - data['theta'][n2])
        delta[l] = data['theta'][n1] - data['theta'][n2]

    phierr = 0.5*delta**2 - data['phi']

    Pfn = np.zeros(N)
    Qfn = np.zeros(N)
    Ptn = np.zeros(N)
    Qtn = np.zeros(N)
    for n1,n2,l in data['G'].edges_iter(data='id'):
        Pfn[n1] += data['Pf'][l]
        Ptn[n2] += data['Pt'][l]
        Qfn[n1] += data['Qf'][l]
        Qtn[n2] += data['Qt'][l]

    balance = {}
    balance['P'] = data['Pg'] - data['Pd'] - Pfn - Ptn
    balance['Q'] = data['Qg'] - data['Qd'] - Qfn - Qtn

    Plim = {}
    Plim['max'] = (data['Pg']*100 - data['Pgmax']) > 1e-6
    Plim['min'] = (data['Pgmin'] - data['Pg']*100) > 1e-6
    Qlim = {}
    Qlim['max'] = ( data['Qg']*100 - data['Qgmax'])> 1e-6
    Qlim['min'] = (-data['Qgmax'] - data['Qg']*100)> 1e-6

    print('Total load: %0.4f MW, Total gen: %0.4f MW' %(100*sum(data['Pd']),100*sum(data['Pg'])))
    print('Total load: %0.4f MVar, Total gen: %0.4f MVar' %(100*sum(data['Qd']),100*sum(data['Qg'])))
    print('Losses [MW]: %0.3g, %0.3f%%' %(100*(sum(data['Pg'] - data['Pd'])), sum(data['Pg'] - data['Pd'])/sum(data['Pg'])))
    print('Maximum |Pf|: %0.4f MW,   Maximum |Pt|: %0.4f MW' %(100*max(np.abs(data['Pf'])), 100*max(np.abs(data['Pt']))))
    print('Maximum |Qf|: %0.4f MVar, Maximum |Qt|: %0.4f Mvar' %(100*max(np.abs(data['Qf'])), 100*max(np.abs(data['Qt']))))
    print('Maximum Pf error: %0.3g' %(max(np.abs(data['Pf'] - Pf))))
    print('Maximum Qf error: %0.3g' %(max(np.abs(data['Qf'] - Qf))))
    print('Maximum Pt error: %0.3g' %(max(np.abs(data['Pt'] - Pt))))
    print('Maximum Qt error: %0.3g' %(max(np.abs(data['Qt'] - Qt))))
    print('Maximum |P balance|: %0.3g' %( max(np.abs(balance['P']))))
    print('Maximum |Q balance|: %0.3g' %( max(np.abs(balance['Q']))))
    print('Maximum angle difference: %0.2f deg' %(max(np.abs(delta))*180/np.pi))
    print('Maximum differentce between phi and delta^2/2: %0.3g' %(max(np.abs(phierr))))
    print('Maximum v: %0.3f, Minimum v: %0.3f' %( max(np.exp(data['u'])), min(np.exp(data['u']))))
    print('Pmax violations: %d, Pmin violations: %d' %(sum(Plim['max']), sum(Plim['min'])))
    print('Qmax violations: %d, Qmin violations: %d' %(sum(Qlim['max']), sum(Qlim['min'])))

if __name__ == '__main__':
    import sys
    fname = sys.argv[1] 
    data = pickle.load(open(fname,'rb'))
    rescheck(data)
