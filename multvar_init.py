import networkx as nx
import numpy as np
import helpers as hlp
import logging
import logfun as lg

#FORMAT = '%(asctime)s %(levelname)7s: %(message)s'
#logging.basicConfig(format=FORMAT,level=logging.DEBUG,datefmt='%H:%M:%S')

def topology_generator(type='ER', **kwargs):
    if type=='ER':
        n = int(kwargs.get('N', 2000))
        deg_avg = float(kwargs.get('deg_avg', 2.5))
        p = float(kwargs.get('p', deg_avg/float(n)))
        option = int(kwargs.get('ERoption', 2))
        return random_graph(n,p, deg_avg, option)
    elif type == 'RT':
        ftop = kwargs.get('topology_file', None)
        option  = int(kwargs.get('RToption', 1))
        deg_avg = float(kwargs.get('deg_avg', 2.5))
        if ftop is None:
            raise(Exception('No topology_file specified for RT topology'))
        return RTsmallworld(ftop, option=option, deg_avg=deg_avg)
    else:
        raise(Exception('unknow type %s' %(type)))

def RTsmallworld(ftop, option=1, deg_avg=None):
    import pandas as pd
    top = pd.read_csv(ftop)
    # change to zero indexing
    top['f'] -= 1
    top['t'] -= 1
    f_node = top['f'].values
    t_node = top['t'].values
    G = nx.MultiDiGraph()
    G.add_edges_from(zip(f_node,t_node,[{'id':i} for i in range(f_node.shape[0])]))
    if option == 1:
        return G
    elif option == 2:
        if deg_avg is not None:
            return RTsmallworld_deg_force(G, deg_avg)
        else:
            raise(Exception('For RTsmallworld option 2, avg_deg must be specified'))
    else:
        raise(Exception('option of r RTsmall World must be either 1 or 2'))

def RTsmallworld_deg_force(G, deg_avg):
    Gtmp = nx.Graph(G)
    target_branches = round(deg_avg*Gtmp.number_of_nodes()/2)
    while Gtmp.number_of_edges() > target_branches:
        cut_vert = list(nx.articulation_points(Gtmp))
        degg1 = [k for k,v in Gtmp.degree().items() if v > 1]
        while True:
            n1 = degg1[np.random.randint(len(degg1))]
            n2 = Gtmp.neighbors(n1)[np.random.randint(Gtmp.degree(n1))]
            if (Gtmp.degree(n2) > 1) and (not ((n1 in cut_vert) and (n2 in cut_vert) ) ):
                break
        Gtmp.remove_edge(n1, n2)

    assert nx.number_connected_components(Gtmp) == 1

    G2 = nx.MultiDiGraph()
    id = 0
    for u, v in Gtmp.edges_iter():
        if np.random.rand() < 0.5:
            G2.add_edge(u, v, attr_dict={'id':id})
        else:
            G2.add_edge(v, u, attr_dict={'id':id})
        id += 1
    return G2



def random_graph(n,p, deg_avg, option):
    """ create random graph and pick largest connected component """
    if option == 1:
        ER = nx.convert_node_labels_to_integers(sorted(nx.connected_component_subgraphs(nx.fast_gnp_random_graph(n=n, p=p)), key=len, reverse=True)[0])
    elif option == 2:
        ER  = nx.fast_gnp_random_graph(n=n, p=p)
        comp = list(nx.connected_components(ER))
        ## randomly permute compoent order
        comp = [comp[i] for i in np.random.permutation(len(comp))]
        
        ## add a branch connecting each comonent to the next to ensure connectivity
        for i in range(len(comp)-1):
            n1idx = np.random.randint(len(comp[i]))
            n2idx = np.random.randint(len(comp[i+1]))
            for j, n1 in enumerate(comp[i]):
                if j == n1idx:
                    break
            for j, n2 in enumerate(comp[i+1]):
                if j == n2idx:
                    break
            ER.add_edge(n1, n2)
        target_branches = round(deg_avg*n/2)
        while ER.number_of_edges() > target_branches:
            cut_vert = list(nx.articulation_points(ER))
            degg1 = [k for k,v in ER.degree().items() if v > 1]
            while True:
                n1 = degg1[np.random.randint(len(degg1))]
                n2 = ER.neighbors(n1)[np.random.randint(ER.degree(n1))]
                if (ER.degree(n2) > 1) and (not ((n1 in cut_vert) and (n2 in cut_vert) ) ):
                    break
            ER.remove_edge(n1, n2)

    assert nx.number_connected_components(ER) == 1
    
    G = nx.MultiDiGraph()
    id = 0
    for u, v in ER.edges_iter():
        if np.random.rand() < 0.5:
            G.add_edge(u, v, attr_dict={'id':id})
        else:
            G.add_edge(v, u, attr_dict={'id':id})
        id += 1
    return G
        
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
    lg.log_zones_split(pre=True)
    zones, boundaries, eboundary_map = zp.get_zones(G,Nmax,Nmin)
    lg.log_zones_split(pre=False, num=len(zones))
    ##### sort based on number of boundary edges #####
    boundary_edge_num  = {i:len(eboundary_map[i][1]) for i in range(len(zones))}
    boundary_edge_sort = sorted(boundary_edge_num,key=boundary_edge_num.get)
    zones              = [zones[i]         for i in boundary_edge_sort]
    boundaries         = [boundaries[i]    for i in boundary_edge_sort]
    eboundary_map      = [eboundary_map[i] for i in boundary_edge_sort]
    #### map from edge id to list of zones it is incident upon
    e2z = ebound2zones([x[1] for x in eboundary_map])
    return zones, boundaries, eboundary_map, e2z

def zone_inputs(zones,boundaries,eboundary_map,resd,resg,resf,resz,log_samples,Sonly=False):
    """ initialize zone inputs WITHOUT creating the optimization model 
        This is throught for a situation where Z is known during the optimization
        Therefore bigM is not needed, and is taken out here.
    """
    inputs = []
    for i,(H,boundary,ebound) in enumerate(zip(zones,boundaries,eboundary_map)):
        lg.log_zone_init(i, H, ebound)
        ### Sample Power and Impedance ####
        inputs.append({})
        inputs[i]['S'] = hlp.multivar_power_sample(H.number_of_nodes(),resd,resg,resf)
        if not Sonly:
            inputs[i]['z'] = hlp.multivar_z_sample(H.number_of_edges(), resz)
            log_samples(S=inputs[i]['S'],z=inputs[i]['z'])
        else:
            log_samples(S=inputs[i]['S'])
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

if __name__=='__main__':
    import ipdb; ipdb.set_trace()
    G = topology_generator(type='ER', deg_avg=2.3, N=2383, ERoption=2)
