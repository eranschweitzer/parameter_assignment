import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
#from networkx.nx_pydot import graphviz_layout
from scipy import sparse
import pickle


def DC_powerflow(p,b,f,t,ref):
    """ calculate DC powerflow flows based on the 
    vector of injections, branch susceptances
    and list of from and to buses of lines
    """
    row_ind = np.concatenate([np.arange(f.shape[0]), np.arange(t.shape[0])])
    col_ind = np.concatenate([f,t])
    data = np.concatenate([np.ones(f.shape[0]), -1*np.ones(t.shape[0])])

    slack_mask = col_ind == ref
    col_ind_slack = col_ind[~slack_mask]
    row_ind_slack = row_ind[~slack_mask]
    col_ind_slack[col_ind_slack > ref] -= 1
    #row_ind_slack[row_ind_slack > ref] -= 1
    data_slack = data[~slack_mask]

    M = sparse.csr_matrix((data,(row_ind,col_ind)),shape=[b.shape[0],p.shape[0]])
    M_slack = sparse.csr_matrix((data_slack,(row_ind_slack,col_ind_slack)),shape=[b.shape[0],p.shape[0] - 1])

    B = M_slack.transpose().dot(sparse.diags(b,0)).dot(M_slack)
    if ref == 0:
        p_no_slack = p[1:]
    elif ref == p.shape[0] - 1:
        p_no_slack = p[:-1]
    else:
        p_no_slack = np.concatenate([p[:ref],p[ref+1:]])

    theta = -1*sparse.linalg.spsolve(B,p_no_slack)
    
    #add slack back in
    if ref == 0:
        theta = np.concatenate([[0],theta])
    elif ref == p.shape[0] - 1:
        theta = np.concatenate([theta,[0]])
    else:
        theta = np.concatenate([theta[:ref],[0],theta[ref:]])
    
    delta = M.dot(theta)
    flows = -1*sparse.diags(b,0).dot(delta)
    p_calc = M.transpose().dot(flows)
    pf = {'theta':theta,'delta':delta,'flows':flows,'p_calc':p_calc} 
    return pf, make_graph(p,b,f,t,ref,pf)

def make_graph(p,b,f,t,ref,pf):
    """ create graph """
    G = nx.MultiDiGraph()
    for i in range(len(p)):
            G.add_node(i,attr_dict={'ref':i==ref,'p':p[i],'theta':pf['theta'][i]})
    for i in range(len(b)):
        G.add_edge(f[i],t[i],attr_dict={'flow':pf['flows'][i],'delta':pf['delta'][i],\
                'f':f[i],'t':t[i], 'x':-1/b[i]})
    return G


def write_graph(path,G=None,p=None,b=None,f=None,t=None,ref=None):
    if G is None:
        pf,G = DC_powerflow(p,b,f,t,ref)
    edges = {'Source':[], 'Target':[],'Type':[], 'flow':[], 'delta':[]}
    for u, v, d in G.edges_iter(data=True):
        edges['Type'].append('Directed')
        if d['flow'] >= 0:
            edges['Source'].append(u)
            edges['Target'].append(v)
        else:
            edges['Source'].append(v)
            edges['Target'].append(u)
        edges['flow'].append(abs(d['flow']))
        #edges['flow_sign'].append(d['flow'] >= 0 )
        edges['delta'].append(abs(d['delta']))
        #edges['delta_sign'].append(d['delta'] >= 0 )
    nodes = {'id':[], 'label':[], 'p':[], 'p_sign':[], 'theta':[], 'theta_sign':[]}
    for u, d in G.nodes_iter(data=True):
        nodes['id'].append(u)
        nodes['p'].append(abs(d['p']))
        nodes['p_sign'].append(np.sign(d['p']))
        nodes['theta'].append(abs(d['theta']))
        nodes['theta_sign'].append(np.sign(d['theta']))
        if d['ref']:
            nodes['label'].append("ref")
        else:
            nodes['label'].append("")
    def stringizer(x):
        if type(x) is bool:
            return str(int(x))
        elif type(x) is np.bool_:
            return str(int(x))
        elif type(x) is np.int64:
            return str(x)
        elif x is None:
            return "None"
        elif type(x) is np.float64:
            return str(float(x))
        elif type(x) is np.ndarray:
            if x.shape[0] == 1:
                return str(x[0])
            else:
                return str([x[i] for i in x])
        else:
            import ipdb; ipdb.set_trace()
    pd.DataFrame(edges).to_csv(path + '_edges.csv',index_col=False)
    pd.DataFrame(nodes).to_csv(path + '_nodes.csv',index_col=False)
    #nx.write_gml(G,path,stringizer=stringizer)
    #file = open(path,'w')
    #try:
    #    for line in nx.generate_gml(G,stringizer=stringizer):
    #        import ipdb; ipdb.set_trace()
    #        file.write((line + '\n'))
    #except nx.NetworkXError:
    #    pass
    #file.close()

def graph_plot(G,pos=None,node_size=100,edge_vmax=None,edge_vmin=None,vmax=None,vmin=None,robust_labels=False,noload_labels=False):

    if pos is None:
        pos = graphviz_layout(G,prog='neato',args="-Gremincross=true")
        #Gtmp= nx.Graph()
        #Gtmp.add_edges_from(G.edges_iter())
        #ipdb.set_trace()
        #pos = graphviz_layout(Gtmp,prog='fdp', args="-Gremincross=true")

    node_cmap = plt.get_cmap('PRGn')
    edge_cmap = plt.get_cmap('YlOrRd')
    edge_color = np.zeros(G.number_of_edges())
    edge_labels = {}
    i = 0
    for u,v,d in G.edges(data=True):
        edge_color[i] = np.abs(d['flow'])
        edge_labels[u,v] = "%0.1f" %(np.abs(d['flow'])*100)
        i += 1

    if edge_vmax is None:
        edge_vmax = edge_color.max()
    if edge_vmin is None:
        edge_vmin = edge_color.min()
    
    node_color = np.zeros(G.number_of_nodes())
    node_labels = {}
    i = 0
    for n,d in G.nodes(data=True):
        node_color[i] = d['p']
        if robust_labels:
            if d['ref']:
                node_labels[n] = 'ref, %0.1f' %(d['p']*100)
            else:
                node_labels[n] = '%0.1f' %(d['p']*100)
        else:
            if d['ref']:
                node_labels[n] = 'ref'
            elif d['p'] == 0:
                if noload_labels:
            #    node_labels[n] = '0'
                    node_labels[n] = '0,\nd=%d' %(G.degree(n))
        i += 1
    if vmax is None:
        vmax = node_color.max()
    if vmin is None:
        vmin = node_color.min()

    # use symmetric vmax,vmin so that 0 is nicely in the middle
    vlim = max(np.abs(vmax),np.abs(vmin))

    fig,ax = plt.subplots(1,figsize=(10,10))
    nx.draw_networkx(G,ax=ax,pos=pos,node_color=node_color,node_size=node_size,\
            cmap= node_cmap,edge_color=edge_color, edge_cmap=edge_cmap,\
            vmax = vlim, vmin=-vlim, edge_vmax=edge_vmax,edge_vmin=edge_vmin,\
            width = 3, labels = node_labels)
    if robust_labels:
        nx.draw_networkx_edge_labels(G,pos,ax=ax,edge_labels=edge_labels)
    ax.set_axis_off()


    gradient = np.linspace(0, 1, num=256)
    gradient = np.vstack((gradient, gradient))
    ax1 = fig.add_axes([.1,.05,.8,.025])
    ax1.imshow(gradient, aspect='auto', cmap=node_cmap)
    ax1.xaxis.tick_top()
    ax1.set_xticks([(vmin+vlim)/(2*vlim)*255,128,(1-(vmax-vlim)/(2*vlim))*255])
    ax1.set_xticklabels(100*np.array([vmin,0,vmax]))
    ax1.set_xlabel('Node Injection [MW]')
    ax1.xaxis.set_label_position('top') 
    ax1.set_yticks([])
    ax2 = fig.add_axes([.1,.025,.8,.025])
    ax2.imshow(gradient, aspect='auto', cmap=edge_cmap)
    ax2.set_xticks([0,127,255])
    ax2.set_xticklabels(100*np.array([edge_vmin,(edge_vmax-edge_vmin)/2,edge_vmax]))
    ax2.set_xlabel('Branch Flow [MW]')
    ax2.set_yticks([])
    return fig,ax,pos 


if __name__=='__main__':
    import run_file as rn

    ####### original case
    bus_data,gen_data,branch_data = rn.load_data('../cases/polish2383_wp')
    Pg,Pd = rn.power_injections(gen_data,bus_data)
    p = (Pg-Pd)/100 # change to per unit
    b = -1/branch_data['BR_X'] 
    ref = bus_data.loc[bus_data['BUS_TYPE']==3,'BUS_I'].values[0]
    
    realpf = DC_powerflow(p,b.values,branch_data['F_BUS'].values,branch_data['T_BUS'].values,ref)
    Greal = make_graph(p,b.values,branch_data['F_BUS'].values,branch_data['T_BUS'].values,ref,realpf)
    fig,ax,pos = graph_plot(Greal)
    fig.savefig('Real_case.png',dpi=300)
    ###### allocation
    alloc = pickle.load(open('../data/assignment_results.pkl','rb'))
    p_alloc = alloc['p_in'][alloc['power_perm'].astype(int)]
    b_alloc = alloc['b_in'][alloc['susceptance_perm'].astype(int)]
    ref_alloc = np.argmax(p_alloc)

    respf = DC_powerflow(p_alloc,b_alloc,branch_data['F_BUS'].values,branch_data['T_BUS'].values,ref_alloc)

    Galloc = make_graph(p_alloc,b_alloc,branch_data['F_BUS'].values,branch_data['T_BUS'].values,ref_alloc,respf)
    fig,ax,pos = graph_plot(Galloc,pos=pos)
    fig.savefig('Reallocated_case.png',dpi=300)
    

    fig, ax = plt.subplots(2)
    ######### line flows
    h  = ax[0].hist(realpf['flows']*100,bins='auto',label='real')
    ax[0].hist(respf['flows']*100,bins=h[1],alpha = 0.5,label='allocation')
    ax[0].set_xlabel('Line Flows [MW]')
    ax[0].legend()

    ######### line delta
    h  = ax[1].hist(realpf['delta']*180/np.pi,bins='auto',label='real')
    ax[1].hist(respf['delta']*180/np.pi,bins=h[1],alpha=0.5,label='allocation')
    ax[1].set_xlabel('Angle Differences [degrees]')
    ax[1].legend()
     
