import numpy as np

def solve(solvers,e2z,logging=None):
    beta_bar = {l:0 for l in e2z}
    gamma_bar= {l:0 for l in e2z}
    for i, s in enumerate(solvers):
        if logging is not None:
            logging(i,pre=True)
        s.optimize()
        if logging is not None:
            logging(s)
        #### update avg. values ####
        for l in s.beta:
            beta_bar[l] += s.beta[l].X/2
            gamma_bar[l]+= s.gamma[l].X/2

    ####### calculate errors ##########
    gap = {'beta':0, 'gamma':0}
    beta_diff = {}
    gamma_diff= {}
    for l in beta_bar:
        beta_diff[l]  = np.abs(solvers[e2z[l][0]].beta[l].X  - solvers[e2z[l][1]].beta[l].X)
        gamma_diff[l] = np.abs(solvers[e2z[l][0]].gamma[l].X - solvers[e2z[l][1]].gamma[l].X)
        for i in e2z[l]:
            gap['beta'] += (solvers[i].beta[l].X  - beta_bar[l])**2
            gap['gamma']+= (solvers[i].gamma[l].X - gamma_bar[l])**2
    mean_diff = {'beta':  sum(beta_diff.values())/len(beta_diff),
                 'gamma': sum(gamma_diff.values())/len(gamma_diff)}
    max_diff  = {'beta': max(beta_diff.values()), 'gamma': max(gamma_diff.values())}
    return beta_bar, gamma_bar, {'gap':gap, 'mean_diff':mean_diff, 'max_diff':max_diff} 

def update(solvers, iter, beta_bar, gamma_bar, rho):
    for s in solvers:
        if iter == 0:
            s.remove_abs_vars()
            s.m._solmin = 1
        s.objective_update(beta_bar, gamma_bar, rho)


def termination(iter,vals,thresholds):
    msg = []
    for t in ['gap', 'mean_diff', 'max_diff']:
        if (vals[t]['beta'] <= thresholds[t]) and (vals[t]['gamma'] <= thresholds[t]):
            msg.append('%s thresholds satisfied' %(t))
    if iter == thresholds['itermax']:
        msg.append('Maximum Iteration')
    return (len(msg)>0), msg
