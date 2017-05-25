import pickle
import formulation as fm
import helpers as hlp
import pandas as pd
import numpy as np
import networkx as nx

data = pickle.load(open('polish2383kde18-05-2017_2255.pkl','rb'))
Pg_out = data['Pg']
Pd_out = data['Pd']
b_out  = data['b']


top = pd.read_csv('./cases/RT_3000.csv')
# change to zero indexing
top['f'] -= 1
top['t'] -= 1
f_node = top['f'].values
t_node = top['t'].values
G = nx.MultiDiGraph()
G.add_edges_from(zip(f_node,t_node,[{'id':i} for i in range(f_node.shape[0])]))


beta,beta_bar,beta_diff,wdump,nu_map,pdump,bdump = pickle.load(open('iteration_5_dump_ph.pkl','rb'))

####### DC Powerflow ###########
import assignment_analysis as asg
ref = np.argmax((Pg_out - Pd_out)/100)
pf, Gpf = asg.DC_powerflow((Pg_out - Pd_out)/100, b_out, f_node, t_node, ref)

print('max flow pre optimization: %0.3g' %(max(abs(pf['flows']))) )
print('max delta pre optimization: %0.3f degree, %0.3f rad' %(max(abs(pf['delta']))*180/np.pi, max(abs(pf['delta']))) )

#### Optimization ########
balance_epsilon = 1e-4
delta_max = 60.0*np.pi/180.
f_max = (10,max(abs(pf['flows'])))
M = max(abs(pf['flows'])) + max(max(abs(pf['delta'])), delta_max)*max(np.abs(b_out)) + 0.5 #plus half is out of precaution
invars = {'G':G,'p':(Pg_out-Pd_out)/100, 'b':b_out, 'edge_boundary':set(beta_bar.keys()), \
        'f_max':f_max, 'M':M, 'balance_epsilon':balance_epsilon, 'delta_max': delta_max}
b_out = fm.intertie_suceptance_assign(invars)

##### Rerun DC powerflow ##############
pf, Gpf = asg.DC_powerflow((Pg_out - Pd_out)/100, b_out, f_node, t_node, ref)
print('max flow post optimization: %0.3g' %(max(abs(pf['flows']))) )
print('max delta post optimization: %0.3f degree, %0.3f rad' %(max(abs(pf['delta']))*180/np.pi, max(abs(pf['delta']))) )

pickle.dump({'Pg': Pg_out, 'Pd': Pd_out, 'b': b_out, 'G': Gpf, 'pf': pf, 'ref': ref},open('susceptance_optimization_test.pkl','wb'))
