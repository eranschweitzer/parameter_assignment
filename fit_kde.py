import sys
import helpers as hlp
import numpy as np
import pickle

def main(fname):
    bus_data, gen_data, branch_data = hlp.load_data(fname)
    Pd = bus_data['PD'].values
    x = branch_data['BR_X'].values
    Pg = np.zeros(bus_data.shape[0])
    for bus,v in zip(gen_data['GEN_BUS'],gen_data['PG']):
        Pg[bus] += v
    
    fit = hlp.analyze_statistics(Pg,Pd,x)
    
    pickle.dump(fit,open('%s_kdefit.pkl' %(fname), 'wb'))

if __name__ == "__main__":
    main(sys.argv[1])
