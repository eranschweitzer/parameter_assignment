import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
#from networkx.nx_pydot import graphviz_layout
from scipy import sparse
import pickle
from helpers import load_data,power_injections
import formulation as fm


def get_zones(G,Nmax,Nmin,debug=False):
    
    if debug:
        iter = 1
        writer = pd.ExcelWriter('python_debug.xlsx', engine='xlsxwriter')
        
    sub_problem_nodes = [np.array(G.nodes())] 
    while np.any([len(i) > Nmax for i in sub_problem_nodes]):
        nbunch = sub_problem_nodes.pop(np.argmax([len(i) for i in sub_problem_nodes]))
        try:
            Gtmp  = nx.Graph(G.subgraph(nbunch))
            L = nx.laplacian_matrix(Gtmp, nodelist=nbunch).asfptype()
            eigs, v = sparse.linalg.eigsh(L, k=2, which='SM')
            fiedler_vect = v[:,1]
        except nx.NetworkXError:
            import ipdb; ipdb.set_trace()
        sub_problem_nodes.append(nbunch[np.where(fiedler_vect <= 0)[0]])
        sub_problem_nodes.append(nbunch[np.where(fiedler_vect  > 0)[0]])
        
        if debug:
            df = pd.DataFrame({0: sub_problem_nodes[0]})
            for k in range(1,len(sub_problem_nodes)):
                df = pd.concat([df, pd.DataFrame({k: sub_problem_nodes[k]})], axis=1)
            df.to_excel(writer,'iter%d' %(iter))
            iter +=1 
        nbunch = None
        i = 0
        initial_length = len(sub_problem_nodes)
        while i < initial_length:
            nbunch = sub_problem_nodes.pop(0)
            if nx.number_connected_components(nx.Graph(G.subgraph(nbunch))) > 1:
                for cnodes in sorted(nx.connected_components(nx.Graph(G.subgraph(nbunch))), key=len, reverse=True):
                    sub_problem_nodes.append(np.array(list(cnodes)))
            else:
                sub_problem_nodes.append(nbunch)
            i += 1
        sub_problem_nodes.sort(key=len)
        # now append nodes in compoents that are too small to their neighbors in other components
        nbunch = None
        i = 0
        initial_length = len(sub_problem_nodes)
        while i < initial_length:
            nbunch = sub_problem_nodes.pop(0)
            if len(nbunch) < Nmin:
                neighbors = []
                for nn in nbunch:
                    neighbors += nx.Graph(G).neighbors(nn)
                comp_id = None
                for nnn in neighbors:
                    try:
                        comp_id = [nnn in sub for sub in sub_problem_nodes].index(True)
                        break
                    except ValueError:
                        pass
                if comp_id is None:
                    import ipdb; ipdb.set_trace()
                sub_problem_nodes[comp_id] = np.concatenate([sub_problem_nodes[comp_id],nbunch])
                
            else:
                sub_problem_nodes.append(nbunch)
            i += 1
            
    if debug:
        writer.save()
    #return list(zip([G.subgraph(i) for i in sub_problem_nodes],[boundary_nodes(G,i) for i in sub_problem_nodes])) 
    return [G.subgraph(i) for i in sub_problem_nodes],[boundary_nodes(G,i) for i in sub_problem_nodes],[boundary_edge_map(G,i) for i in sub_problem_nodes]

def boundary_nodes(G,nbunch):
    """ get boundary nodes that are IN nbunch """
    eboundary = nx.edge_boundary(nx.Graph(G),nbunch)
    nboundary = []
    for u,v in eboundary:
        if (u in nbunch) and (v not in nbunch):
            if u not in nboundary:
                # avoid duplicate entries
                nboundary.append(u)
        elif (u not in nbunch) and (v in nbunch):
            if v not in nboundary:
                # avoids duplicate entries
                nboundary.append(v)
        else:
            raise Exception("Error in edge boundary")
    return nboundary

def boundary_edge_map(G,nbunch):
    nboundary = boundary_nodes(G,nbunch)
    eboundary = {'in': {i:[] for i in nboundary}, 'out': {i:[] for i in nboundary}}
    eboundary_id = []
    for node in nboundary:
        for u,v,l in G.out_edges_iter([node],data=True):
            if v not in nbunch:
                eboundary['out'][u].append(l['id'])
                eboundary_id.append(l['id'])
        for u,v,l in G.in_edges_iter([node],data=True):
            if u not in nbunch:
                eboundary['in'][v].append(l['id'])
                eboundary_id.append(l['id'])
    return eboundary,eboundary_id

def boundary_edges(G,zones):
    """ return set of boundary edges between the zones 
    zones should be a list of subgraphs of G """
    edges = set()
    for z in zones:
        edges.update(nx.edge_boundary(G,z.nodes()))
    nodes = set()
    n2n = {}
    for u,v in edges:
        nodes.update({u})
        nodes.update({v})
        #try:
        #    n2n[u] += [v]
        #except KeyError:
        #    n2n[u] = [v]
        #
        #try:
        #    n2n[v] += [u]
        #except KeyError:
        #    n2n[v] = [u]
    Gbound = nx.Graph(G.subgraph(nodes))
    for comp in nx.connected_components(Gbound):
        for nn in comp:
            n2n[nn] = comp.difference({nn})
    return edges,n2n

if __name__=='__main__':
#    bus_data,gen_data,branch_data = load_data('../cases/case118')
#    Pg,Pd = power_injections(gen_data,bus_data)
#    p = (Pg-Pd)/100 # change to per unit
#    b = -1/branch_data['BR_X'].values 
#    ref = bus_data.loc[bus_data['BUS_TYPE']==3,'BUS_I'].values[0]
#    
#    
#    p_full = {'in':np.random.permutation(p),'out':np.zeros(p.shape[0]),'ind': np.ones(p.shape[0])}
#    b_full = {'in':np.random.permutation(b),'out':np.zeros(b.shape[0]),'ind': np.ones(b.shape[0])}
#
#    G = nx.MultiDiGraph()
#    G.add_edges_from(zip(branch_data['F_BUS'],branch_data['T_BUS'],[{'id':i} for i in branch_data.index]))
#    for u,v,k in G.edges_iter(data='id'):
#        print(u,v,k)
#
#    ipdb.set_trace()
#    Nmax = 50
#    zones,boundaries = get_zones(G,Nmax)
#    eboundary,n2n = boundary_edges(G,zones)

    bus_data,gen_data,branch_data = load_data('./cases/polish2383_wp')
    f_node = branch_data['F_BUS'].values
    t_node = branch_data['T_BUS'].values
    
    G = nx.MultiDiGraph()
    G.add_edges_from(zip(f_node,t_node,[{'id':i} for i in range(f_node.shape[0])]))
    
    
    Nmax = 400; Nmin = 50;
#    zones,boundaries,_,_= pickle.load(open('../polish_debug/zone_dump.pkl','rb'))
    zones2,boundaries2,edge_maps = get_zones(G,Nmax,Nmin,debug=False)
#    ipdb.set_trace()
    boundary_edges,n2n  = boundary_edges(G,zones)
