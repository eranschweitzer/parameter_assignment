import pandas as pd
import numpy as np

for i in range(1,4):
    df = pd.read_csv('random_assignment_max_out%d.csv' %i)
    if i == 1:
        delta = df['delta'].values
        flows = df['flows'].values
    else:
        delta = np.concatenate([delta,df['delta'].values])
        flows = np.concatenate([flows,df['flows'].values])

hist = {}
hist['delta_v'],    hist['delta_edges']     = np.histogram(delta, bins='auto', density=True)
hist['flows_v'],    hist['flows_edges']     = np.histogram(flows, bins='auto', density=True)

pd.DataFrame(dict([(k,pd.Series(v)) for k,v in hist.items()])).to_csv('random_assignment_max_distributions.csv',index=False)
