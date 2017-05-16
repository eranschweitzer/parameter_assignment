import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import sparse,stats
import pickle
import sys
#sys.path.append('/Users/eran/Dropbox/ASU/SINE/python')
sys.path.append('../../../../../SINE/python')
import my_functions as my
import helpers as hp
import assignment_analysis as asg


bus_data,gen_data,branch_data = hp.load_data('../cases/polish2383_wp')
#Pg,Pd = hp.power_injections(gen_data,bus_data)
#p = (Pg-Pd)/100 # change to per unit
#b = -1/branch_data['BR_X'].values
f = branch_data['F_BUS'].values
t = branch_data['T_BUS'].values
#ref = bus_data.loc[bus_data['BUS_TYPE']==3,'BUS_I'].values[0]
#
#realpf,Greal = asg.DC_powerflow(p,b,f,t,ref)
#asg.write_graph('polish2383wp',G=Greal)

data    = pickle.load(open('../data/polish2383wp_ph_real.pkl','rb'))
Pg   = np.array(data['Pg'])
Pd   = np.array(data['Pd'])
b    = np.array(data['b'])
p    = (Pg-Pd)/100 # change to per unit
ref  = np.argmax(p)
asg.write_graph('polish_shuffle',p=p,b=b,f=f,t=t,ref=ref)

#data = pickle.load(open('../data/RT3000_ph.pkl','rb'))
#Pg   = np.array(data['Pg'])
#Pd   = np.array(data['Pd'])
#b    = np.array(data['b'])
#p    = (Pg-Pd)/100 # change to per unit
#ref  = np.argmax(p)
#top  = pd.read_csv("../cases/RT_3000.csv")
## change to zero indexing
#top['f'] -= 1
#top['t'] -= 1
#f_node = top['f'].values
#t_node = top['t'].values
#
#asg.write_graph('RT3000',p=p,b=b,f=f_node,t=t_node,ref=ref)
