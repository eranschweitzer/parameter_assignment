import time
import numpy as np
import networkx as nx
import random
import logging
import pickle
import helpers as hlp
import multvar_init as mvinit
import ea_init as init
import logfun as lg
import ea


def main(fname):
    lgslv = {'log_iteration_start':lg.log_iteration_start, 
             'log_iterations': lg.log_iterations, 
             'log_iteration_summary':lg.log_iteration_summary,
             'log_termination': lg.log_termination,
             'log_single_system': lg.log_single_system}

    Psi = [None, pickle.load(open(fname,'rb'))]
    Psi[1].inputs['globals']['consts']['rho']=100
    Psi[1].inputs['globals']['consts']['aug_relax'] = True
    Psi[1].inputs['globals']['consts']['hbeta'] = 25
    Psi[1].inputs['globals']['consts']['beta2_err'] = 0.01
    Psi[1].inputs['globals']['consts']['htheta'] = hlp.polyhedral_h(Psi[1].inputs['globals']['consts']['dmax'], 1e-3)
    Psi[0] = ea.EAgeneration(Psi[1].inputs)
    Psi[1].initialize_optimization()
    for ind, psi in enumerate(Psi[1].iter()):
        lg.log_individual(ind)
        psi.solve(Psi[1].inputs,logging=lgslv, solck=True, print_boundary=False, write_model=False, fname='debug/polish2383wp', rho_update='sqrt')
    

if __name__=="__main__":
    import sys
    main(*sys.argv[1:])
