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
        z = hlp.multivar_z_sample(H.number_of_edges(),resz)
        log_samples(S,z)
        ### get primitive admittance values ####
        Y = hlp.Yparts(z['r'],z['x'],b=z['b'])
        bigM = hlp.bigM_calc(Y,fmax,umax,dmax)
        logging.info('big M: %0.4g', bigM)
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
