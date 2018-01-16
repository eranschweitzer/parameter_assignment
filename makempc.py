import numpy as np
import pickle
from scipy import io

def savempc(dataname, savename, expand_rate=False, vlim_precision=2, **kwargs):

    if type(dataname) is str:
        data = pickle.load(open(dataname,'rb'))
    else:
        data = dataname
    
    if 'vars' in data:
        for i, vars in enumerate(data['vars']):
            savempc({'G': data['G'], **vars}, update_savename(savename, i), expand_rate=expand_rate, vlim_precision=vlim_precision)
        import sys
        sys.exit(0)

    N = data['G'].number_of_nodes()
    L = data['G'].number_of_edges()
    
    ###### Slacks #########
    sf = try_extract(data,'sf',L)

    ################
    # Branch Matrix
    ################
    branch = np.zeros((L,13))

    for n1,n2,l in data['G'].edges_iter(data='id'):
        branch[l,0] = n1+1                       # from bus
        branch[l,1] = n2+1                       # to bus
        branch[l,2] = data['r'][l]               # resistance (p.u.)
        branch[l,3] = data['x'][l]               # reactance (p.u.)
        branch[l,4] = data['b'][l]               # changing susceptance
        #branch[l,5] = data['rate'][l]*100        # Rate A
        branch[l,5] = set_rate(data['rate'][l], sf[l], expand_rate) # Rate A
        branch[l,6] = 0                          # Rate B
        branch[l,7] = 0                          # Rate C
        branch[l,8] = data['tap'][l]             # off nominal tap
        branch[l,9] = data['shift'][l]*180/np.pi # phase shift
        branch[l,10]= 1               # branch status
        branch[l,11]= 0               # minimum angle difference
        branch[l,12]= 0               # maximum angle difference

    ###################
    # Generator Matrix
    ##################
    G    = sum(data['Pgmax'] > 0)
    gidx = np.where(data['Pgmax'] > 0)[0]
    gen  = np.zeros((G,21))
    for i,g in enumerate(gidx):
        gen[i,0] = g + 1    # Generator Bus
        if data['Pg'][g]*100 > 1:
            gen[i,1] = data['Pg'][g]*100 # Real power [MW]
            gen[i,2] = data['Qg'][g]*100 # Reactive power [MVar]
            gen[i,7] = 1                 # Generator Status
        else:
            gen[i,1] = 0 # Real power [MW]
            gen[i,2] = 0 # Reactive power [MVar]
            gen[i,7] = 0 # Generator Status
        gen[i,3] =  data['Qgmax'][g]  # maximum reactive power [MVar]
        gen[i,4] = -data['Qgmax'][g]  # minimum reactive power [MVar]
        gen[i,5] = np.exp(data['u'][g]) # voltage magnitude setpoint [p.u.]
        gen[i,6] = 100             # MVA base
        gen[i,8] = data['Pgmax'][g]   # Maximum real power output [MW]
        gen[i,9] = data['Pgmin'][g]   # Minimum real power output [MW]
        # the rest are opf related, neglected for now

    
    ############
    # BUS TYPES
    ############
    # pick ref bus as largest capacity generator that is also on
    ref     = int(gen[np.argmax(gen[:,7]*gen[:,8]),0] - 1)
    refang  = data['theta'][ref] # angle of ref bus
    pvbuses = gen[gen[:,7] > 0, 0] - 1

    vmax, vmin = vlim(data['u'], precision=vlim_precision)
    #############
    # Bus Matrix
    ############
    bus  = np.zeros((N,13))
    for i in data['G'].nodes_iter():
        bus[i,0] = i + 1 #bus id
        if i == ref:
            bus[i,1] = 3 #slack bus
        elif i in pvbuses:
            bus[i,1] = 2 #PV bus
        else:
            bus[i,1] = 1 #PQ bus
        bus[i,2] = 100*data['Pd'][i]       # real power
        bus[i,3] = 100*data['Qd'][i]       # reactive power
        try:
            bus[i,4] = 100*data['GS'][i]
        except (KeyError, TypeError):
            bus[i,4] = 0                       # shunt conductance
        try:
            bus[i,5] = 100*data['BS'][i]
        except (KeyError, TypeError):
            bus[i,5] = 0                       # shunt susceptance
        bus[i,6] = 1                       # bus area
        bus[i,7] = np.exp(data['u'][i])    # voltage magnitude
        bus[i,8] = (data['theta'][i] - refang)*180/np.pi # bus angle (shifted so ref is at 0)
        bus[i,9] = 220                     # base kV
        bus[i,10]= 1                       # Zone
        bus[i,11]= vmax                    # maximum voltage magnitude (p.u.)
        bus[i,12]= vmin                    # minimum voltage magnitude (p.u.)

    ##################
    # Generator Cost
    ##################
    # for now all generators are given the same price and linear cost
    gencost = np.zeros((G,6))
    gencost[:,0] = 2        # polynomial cost function
    gencost[:,3] = 2        # number of cost coefficients
    gencost[:,4] = 10       # cost is (arbitrarily set to $10/MW

    io.savemat(savename,{'baseMVA':float(100),'bus':bus,'branch':branch,'gen':gen,'gencost':gencost})

def set_rate(rate, slack, expand_rate):
    if expand_rate:
        return np.ceil((rate + slack)*100)
    else:
        return np.ceil(rate*100)

def vlim(u, precision=2):
    vmax = np.exp(np.max(u))
    vmin = np.exp(np.min(u))
    return np.ceil(round(vmax*(10**(precision+2)))/10**2)/(10**precision), np.floor(round(vmin*(10**(precision+2)))/10**2)/(10**precision)

def update_savename(savename, i):
    saveparts = savename.split('.')
    return saveparts[0] + '_ind%d' %(i) + '.' +  saveparts[1]

def try_extract(d, key, N):
    """ try to extract key from dictionary d. If it fails return zeros array of length N """
    try:
        return d[key]
    except KeyError:
        return np.zeros(N)

if __name__ == '__main__':
    import sys
    dataname = sys.argv[1]
    savename = sys.argv[2]
    kwargs = {}
    try:
        kwargs['expand_rate'] = sys.argv[3] in ['True', 'true', 't', '1']
    except IndexError:
        pass
    try:
        kwargs['vlim_precision'] = int(sys.argv[4])
    except IndexError:
        pass
    savempc(dataname,savename, **kwargs)
