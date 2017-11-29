import numpy as np
import pickle
from scipy import io

def savempc(dataname,savename):
    data = pickle.load(open(dataname,'rb'))
    
    bus = np.zeros((data['G'].number_of_nodes(),13))
    
    for i,d in data['G'].nodes_iter(data=True):
        bus[i,0]     = i+1 #bus id
        if d['ref']:
            bus[i,1] = 3 #slack bus
#        elif data['Pg'][i] > data['Pd'][i]:
#            bus[i,1] = 2 #PV bus
        else:
            bus[i,1] = 1 #PQ bus
        bus[i,2]     = data['Pd'][i] #real power MW
        bus[i,3]     = 0 #data['Pd'][i]*np.tan(np.arccos(0.8 + np.random.random()*0.2))  #reactive power MVar
        bus[i,4]     = 0             #shunt conductance
        bus[i,5]     = 0             #shunt susceptance
        bus[i,6]     = 1             #bus area
        bus[i,7]     = 1             #voltage magnitude
        bus[i,8]     = 0             #voltage angle
        bus[i,9]     = 220           #base kV
        bus[i,10]    = 1             #Zone
        bus[i,11]    = 0.8           #maximum voltage magnitude (p.u.)
        bus[i,12]    = 1.2           #minimum voltage magnitude (p.u.)
    
    branch = np.zeros((data['G'].number_of_edges(),13))
    
    for i,(u,v,d) in enumerate(data['G'].edges_iter(data=True)):
        branch[i,0]  = u+1      #from bus
        branch[i,1]  = v+1      #to bus
        branch[i,2]  = d['x']*0.1 #resistance (p.u.)
        branch[i,3]  = d['x'] #reactance (p.u.)
        samp = np.random.random()
        if samp < 0.9:
            branch[i,4] = (0 + np.random.random()*0.2)*branch[i,3]
        elif (samp >= 0.9) and (samp < 0.99):
            branch[i,4] = branch[i,3]*(0.75+ np.random.random())
        else:
            branch[i,4] = branch[i,3]*(15 + np.random.random()*10)
        branch[i,10] = 1      #branch status
    
    gen_id = np.where(data['Pg']  > 0)[0]
    gen = np.zeros((len(gen_id),10))
    for i,b_id in enumerate(gen_id):
        gen[i,0] =  b_id+1 #bus id
        gen[i,1] =  data['Pg'][b_id] #real power output (MW)
        gen[i,2] =  0 #data['Pg'][b_id]*np.tan(np.arccos(0.8 + np.random.random()*0.2))  #reactive power MVar 
        gen[i,3] =  sum(data['Pg'])  #max reactive power output
        gen[i,4] = -sum(data['Pg'])  #min reacitve power output 
        gen[i,7] = 1 #in service
        gen[i,8] = gen[i,1]*1.2 #maximum power output
    
    io.savemat(savename,{'baseMVA':float(100),'bus':bus,'branch':branch,'gen':gen})

if __name__=='__main__':
    import sys
    savempc(*sys.argv[1:])
