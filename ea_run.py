import time
import numpy as np
import networkx as nx
import random
import pickle
import helpers as hlp
import multvar_init as mvinit
import ea_init as init
import logfun as lg

def main(savename, fdata, topology=None, Nmax=400, Nmin=50, include_shunts=False, const_rate=False, actual_vars_d=False, actual_vars_g=True, actual_vars_z=True, debug=False, logfile=None, **kwargs):

    fin = locals().copy()
    del fin['savename']; del fin['fdata']; del fin['kwargs']
    fin.update(**kwargs)
    start = time.time()
    timestamps = {}
    timestamps['start'] = lg.timestamp()

    lg.logging_setup(fname=logfile)
    lg.log_function_inputs(savename,fdata,**fin) 

    #### INPUTS #########################
    truelist = [True,'True','true','t','1']
    Nmax = int(Nmax); Nmin = int(Nmin)
    actual_vars_z = actual_vars_z in truelist
    actual_vars_d = actual_vars_d in truelist
    actual_vars_g = actual_vars_g in truelist
    include_shunts= include_shunts in truelist
    const_rate    = const_rate in truelist
    debug         = debug in truelist
    topology      = hlp.none_test(topology)

    ##### Define Constants ###############
    C = hlp.def_consts() 
    C['ea'] = {'generations': 5,
               'individuals':10,
               'ea_select':5,
               'mutate_probability':0.05}
    C['aug_relax'] = False
    C['beta2_err'] = 0.01
    C['Qlims']     = True
    C['sil'] = {'usesil': False, 'Sf2Pf': 1.1, 'siltarget': 1.0}
    C['solve_kwargs'] = {'remove_abs': True,
                         'solck': False,
                         'print_boundary': False,
                         'write_model': False,
                         'fname': 'debug/mymodel',
                         'rho_update': 'sqrt'}
    C['savempc'] = {'savempc': True, 'mpcpath': 'mpc/', 'expand_rate': True, 'vlim_precision': 2}
    C['parallel_opt'] = {'parallel':False, 'parallel_zones': False, 'workers': None, 'dump_path': 'pickle_data' }
    C['gurobi_config'] = {'Threads': 60, 'MIPgap': 0.15, 'LogFile': '/tmp/GurobiMultivar.log', 'MIPFocus': 1}
    C['random_solve'] = False
    C['rndslv_params'] = {'rep_max': 10, 'timelimit': 300}
    hlp.update_consts(C,fin)

    C['htheta'] = hlp.polyhedral_h(C['dmax'], C['phi_err'])
    
    ##### Load Data ######### 
    bus_data, gen_data, branch_data = hlp.load_data(fdata)
    vmax = bus_data['VM'].max(); vmin = bus_data['VM'].min()
    C['umin'] = min(C['umin'],np.log(vmin))
    C['umax'] = max(C['umax'],np.log(vmax))

    #### Fit Power and Impedance Data #### 
    import fit_inputs as ftin
    resz,C['fmax'] = ftin.multivariate_z(branch_data, bw_method=0.01, actual_vars=actual_vars_z, fmaxin=C['fmax'], const_rate=const_rate)
    resd,resg,resf = ftin.multivariate_power(bus_data, gen_data, actual_vars_d=actual_vars_d, actual_vars_g=actual_vars_g, include_shunts=include_shunts)

    #### Get Topology ########
    if topology is None:
        G = mvinit.topology(bus_data,branch_data)
    else:
        G = mvinit.topology_generator(type=topology,**fin) 
    N = G.number_of_nodes()
    L = G.number_of_edges()
    if C['random_solve']:
        Nmax = N
        C['ea']['generations'] = 1
    lg.log_topology(N,L,Nmax,Nmin)

    ### Split Into Zones #####
    # if Nmax is sufficiently large there may be just 1 zone
    zones, boundaries, eboundary_map, e2z = mvinit.zones(G,Nmax,Nmin)
    ssamples = mvinit.zone_inputs(zones, boundaries, eboundary_map, resd, resg, resf, resz, lg.log_input_samples, Sonly=True)
    zsamples = hlp.multivar_z_sample(L, resz)
    lg.log_input_samples(z=zsamples)

    #### form inputs dictionary ####
    inputs = {'globals': {'G':G, 'consts':C, 'z': zsamples, 'e2z':e2z}, 'locals':ssamples}
    for i,(H,ebound) in enumerate(zip(zones,eboundary_map)):
        inputs['locals'][i].update({'z':zsamples, 'G': H, 'ebound':ebound[1], 'ebound_map':ebound[0]})

    lg.log_optimization_consts(C)
    lgslv = {'log_iteration_start':lg.log_iteration_start, 
             'log_iterations': lg.log_iterations, 
             'log_iteration_summary':lg.log_iteration_summary,
             'log_termination': lg.log_termination,
             'log_single_system': lg.log_single_system}
    inputs['globals']['consts']['logging'] = lgslv
    inputs['globals']['consts']['saving'] = {'savename': savename, 'logpath': 'logs/'}
    ### Main Loop ####################
    import ea
    Psi = [ea.EAgeneration(inputs), None]
    for i in range(C['ea']['generations']):
        lg.log_generation(i, Psi[0], start=True)
        Psi[1] = Psi[0].mutate(C['ea']['individuals'], pm=C['ea']['mutate_probability'])
        if debug:
            debug_dump(savename, Psi, i, timestamps['start'])
        if not C['parallel_opt']['parallel']:
            Psi[1].initialize_optimization(logging=lg.log_optimization_init)
            for ind, psi in enumerate(Psi[1].iter()):
                lg.log_individual(ind)
                psi.solve(Psi[1].inputs,logging=lgslv, **C['solve_kwargs'])
        elif (C['parallel_opt']['workers'] is None) or (C['parallel_opt']['workers'] == 'self'):
            Psi[1].parallel_wrap()
        else:
            lg.log_outsource_announce(C['parallel_opt']['workers'])
            Psi[1].outsource(logfile=logfile)
        Psi[0] += Psi[1]
        Psi[0].selection(C['ea']['ea_select'])
        lg.log_generation(i, Psi[0], start=False)

    #### saving
    end = time.time()
    timestamps['end'] = lg.timestamp()
    Psi[0].save(savename, timestamps)
    lg.log_total_run(start,end)
    if C['savempc']['savempc']:
        mpcsv = hlp.savepath_replace(savename, C['savempc']['mpcpath'])
        Psi[0].save(mpcsv, timestamps, ftype='mpc', **C['savempc'])

def parse_inputs(fname):
    out = {}
    with open(fname,'r') as f:
        for l in f:
            if l[0] == '#':
                continue
            parts = l.strip().replace(' ' ,'').split(':')
            if len(parts) > 1:
                subparts = parts[1].split(',')
                if len(subparts) > 1:
                    out[parts[0]] = [p for p in subparts]
                else:
                    out[parts[0]] = parts[1]
    savename = out.pop('savename')
    fdata    = out.pop('fdata')
    return savename, fdata, out

def debug_dump(savename, Psi, gen, tstamp):
    if '/' in savename:
        s = savename.split('/')[-1].split('.')[0]
    else:
        s = savename.split('.')[0]
    pickle.dump(Psi[1], open('debug/' + s + '_gen_' + str(gen) + '_' + tstamp + '.pkl','wb'))

if __name__ == '__main__':
    import sys
    savename, fdata, kwargs = parse_inputs(sys.argv[1])
    #main(*sys.argv[1:])
    main(savename, fdata, **kwargs)
