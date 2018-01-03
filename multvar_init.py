import networkx as nx
import helpers as hlp
import logging

FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

def topology(bus_data,branch_data):
    nmap   = dict(zip(bus_data['BUS_I'],range(bus_data.shape[0])))
    f_node = [nmap[i] for i in branch_data['F_BUS']]
    t_node = [nmap[i] for i in branch_data['T_BUS']]

    G = nx.MultiDiGraph()
    id = 0
    for i in branch_data.index:
        if branch_data['BR_STATUS'][i] > 0:
            G.add_edge(f_node[i],t_node[i],attr_dict={'id':id})
            id += 1
    return G

def zones(G,Nmax,Nmin):
    import zone_splitting as zp

    ############ Partition Into Zones #############
    logging.info('Splitting graph into zones')
    zones, boundaries, eboundary_map = zp.get_zones(G,Nmax,Nmin)
    logging.info('%d zones created', len(zones))
    ##### sort based on number of boundary edges #####
    boundary_edge_num  = {i:len(eboundary_map[i][1]) for i in range(len(zones))}
    boundary_edge_sort = sorted(boundary_edge_num,key=boundary_edge_num.get)
    zones              = [zones[i]         for i in boundary_edge_sort]
    boundaries         = [boundaries[i]    for i in boundary_edge_sort]
    eboundary_map      = [eboundary_map[i] for i in boundary_edge_sort]
    #### map from edge id to list of zones it is incident upon
    e2z = ebound2zones([x[1] for x in eboundary_map])
    return zones, boundaries, eboundary_map, e2z

def zone_inputs(zones,boundaries,eboundary_map,resd,resg,resf,resz,log_samples):
    """ initialize zone inputs WITHOUT creating the optimization model 
        This is throught for a situation where Z is known during the optimization
        Therefore bigM is not needed, and is taken out here.
    """
    inputs = {'S':[], 'z':[]}
    for i,(H,boundary,ebound) in enumerate(zip(zones,boundaries,eboundary_map)):
        logging.info('-------------------')
        logging.info('Initializing Zone %d: %d nodes, %d edges, %d boundary edges', i, H.number_of_nodes(), H.number_of_edges(), len(ebound[1]))
        logging.info('-------------------')
        ### Sample Power and Impedance ####
        inputs['S'].append(hlp.multivar_power_sample(H.number_of_nodes(),resd,resg,resf))
        inputs['z'].append(hlp.multivar_z_sample(H.number_of_edges(), resz))
        log_samples(inputs['S'][-1],inputs['z'][-1])
    return inputs
        
def solvers_init(G,Nmax,Nmin,resd,resg,resf,resz,lossmin,lossterm,fmax,dmax,htheta,umin,umax,log_samples):
    import zone_splitting as zp
    import formulation_multvar as fm

    ############ Partition Into Zones #############
    logging.info('Splitting graph into zones')
    zones, boundaries, eboundary_map = zp.get_zones(G,Nmax,Nmin)
    logging.info('%d zones created', len(zones))
    ##### sort based on number of boundary edges #####
    boundary_edge_num  = {i:len(eboundary_map[i][1]) for i in range(len(zones))}
    boundary_edge_sort = sorted(boundary_edge_num,key=boundary_edge_num.get)
    zones              = [zones[i]         for i in boundary_edge_sort]
    boundaries         = [boundaries[i]    for i in boundary_edge_sort]
    eboundary_map      = [eboundary_map[i] for i in boundary_edge_sort]
    solvers = []
    for i,(H,boundary,ebound) in enumerate(zip(zones,boundaries,eboundary_map)):
        logging.info('-------------------')
        logging.info('Initializing Zone %d: %d nodes, %d edges, %d boundary edges', i, H.number_of_nodes(), H.number_of_edges(), len(ebound[1]))
        logging.info('-------------------')
        ### Sample Power and Impedance ####
        S = hlp.multivar_power_sample(H.number_of_nodes(),resd,resg,resf)
        z = hlp.multivar_z_sample(H.number_of_edges(), resz)
        log_samples(S,z)
        ### get primitive admittance values ####
        Y = hlp.Yparts(z['r'],z['x'],b=z['b'],tau=z['tap'],phi=z['shift'])
        bigM = hlp.bigM_calc(Y,fmax,umax,umin,dmax)
        #logging.info('big M: %0.4g', bigM)
        for k,v in bigM.items():
            logging.info('big M%s: %0.4g', k, v)
        #### initialize zone solver #####
        solvers.append(fm.ZoneMILP(H,lossmin,lossterm,fmax,dmax,htheta,umin,umax,z,S,bigM,ebound[1],ebound[0]))
    return solvers, ebound2zones([x[1] for x in eboundary_map])

def ebound2zones(ebound):
    """ dictionar
        keys: branch ids of boundary edges
        values: list pairs of zones the boundary edge is incident upon
    """
    out = {}
    for z in range(len(ebound)):
        for l in ebound[z]:
            if l in out:
                out[l].append(z)
            else:
                out[l] = [z]
    return out
