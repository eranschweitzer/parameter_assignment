import sys
import helpers as hlp
import numpy as np
import pickle
from scipy import interpolate, optimize, stats

def analyze_power_statistics(Pg, Pd, fit='pchip', print_out=True):
    """ analyze the power injections and per unit reactance of a case 
    assumes that Pg and Pd are the same size vectors """
    Gnum = sum(Pg > 0)
    p = Pg - Pd
    frac  = {}
    Pgout = {}
    Pdout = {}
    Pgout['vmin'] = min(Pg[Pg > 0])
    Pgout['vmax'] = max(Pg[Pg > 0])
    Pdout['vmin'] = min(Pd[Pd > 0])
    Pdout['vmax'] = max(Pd[Pd > 0])
    frac['intermediate'] = sum(p == 0)/p.shape[0]
    frac['Pd>Pg']    = sum((Pg > 0) & (Pd > Pg))/Gnum
    frac['Pd<Pg']    = sum((Pg > 0) & (Pd < Pg) & (Pd !=0))/Gnum
    frac['Pg']       = Gnum/p.shape[0]
    frac['net_inj']  = sum(p > 0)/p.shape[0]
    frac['gen_only'] = sum((Pg > 0) & (Pd == 0))/Gnum

    g90per = np.percentile(Pg[Pg > 0],90)
    corr_coeff, corr_coeff_pval = stats.pearsonr(Pg[Pg > 0],Pd[Pg > 0])
    corr_coeff90, corr_coeff90_pval = stats.pearsonr(Pg[(Pg > 0) & (Pg < g90per)],Pd[(Pg > 0) & (Pg < g90per)])
    
    if fit == 'kde':
        Pgout['kde'] = kde_fit(Pg[Pg > 0])
        Pdout['kde'] = kde_fit(Pd[Pd > 0])
    elif fit == 'pchip':
        Pgout['pchip'] = hlp.PchipDist(Pg[Pg > 0])
        Pdout['pchip'] = hlp.PchipDist(Pd[Pd > 0])
    else:
        print('Only kde and pchip fit supported')
        sys.exit(0)
    if print_out:
        for s,v in {'Pg':Pgout,'Pd': Pdout}.items():
            print("%0.3g <= %s <= %0.3g" %(v['vmin'],s,v['vmax']))
        for key,val in frac.items():
            print("Percent %s nodes = %0.2f%%" %(key,val*100))
        print("correlation between generation and load: %0.3g, p-value: %0.3g" %(corr_coeff, corr_coeff_pval))
        print("correlation between 90%% of generation and load: %0.3g, p-value: %0.3f, 90th percentile of Pg: %0.3g" %(corr_coeff90, corr_coeff90_pval, g90per))
    return Pgout, Pdout, frac 

def analyze_reactance_statistics(x, fit='pchip', print_out=True):
    vmin = min(x)
    vmax = max(x)
    if fit == 'kde':
        fit_obj  = kde_fit(x)
    elif fit == 'pchip':
        fit_obj = hlp.PchipDist(x)
    else:
        print('Only kde and pchip fit supported')
        sys.exit(0)
    if print_out:
        print("%0.3g <= x <= %0.3g" %(vmin, vmax))
    return {'vmax': vmax, 'vmin': vmin, fit:fit_obj}

def multivariate_power(bus_data,gen_data,gen_cost=None,bw_method='scott',actual_vars_d=False,actual_vars_g=True,mvabase=100, Bfracmin=0.1, Bshmin=3.0, include_shunts=True):

    N = bus_data.shape[0]
    """ Load """
    resd = {}
    resd['kde'] = kde_fit(bus_data.loc[:,['PD','QD']].values.transpose(), bw_method=bw_method)
    resd['max'] = bus_data.loc[:,['PD','QD']].max(axis=0).values
    resd['min'] = bus_data.loc[:,['PD','QD']].min(axis=0).values
    resd['actual_vars'] = actual_vars_d
    resd['shunt'] = {}
    resd['shunt']['include_shunts'] = include_shunts
    resd['shunt']['max'] = bus_data.loc[:,['GS','BS']].max(axis=0).values/mvabase
    resd['shunt']['min'] = bus_data.loc[:,['GS','BS']].min(axis=0).values/mvabase
    resd['shunt']['Gfrac']= sum(bus_data['GS'] != 0)/N
    resd['shunt']['Bfrac']= sum(bus_data['BS'] != 0)/N

    if resd['shunt']['Bfrac'] == 0:
        resd['shunt']['Bfrac'] = Bfracmin
        resd['shunt']['max'][1] = Bshmin

    """ gen """
    order = dict(zip(range(4),['Pgmax','Pgmin','Qgmax', 'Pcost']))
    inkde = list(range(4))
    vdefault = {}
    x = {}
    genmap = dict(zip(gen_data['GEN_BUS'].unique(),range(len(gen_data['GEN_BUS'].unique()))))
    GBnum = len(genmap) 
    x['Pgmax'] = np.zeros(GBnum)
    x['Pgmin'] = np.empty(GBnum)
    x['Pgmin'][:] = np.inf
    x['Qgmax'] = np.zeros(GBnum)
    x['Pcost'] = np.zeros(GBnum)
    if gen_cost is None:
        tmpcost = 10*np.ones(gen_data.shape[0])
    else:
        tmpcost = gen_cost['COST1']
    for bus,pmax,pmin,qmax,status, cost in zip(gen_data['GEN_BUS'],gen_data['PMAX'],gen_data['PMIN'],gen_data['QMAX'],gen_data['GEN_STATUS'], tmpcost):
        #if status > 0: include all generators, not just those that are on.
        x['Pgmax'][genmap[bus]] += pmax
        x['Pgmin'][genmap[bus]] = np.minimum(x['Pgmin'][genmap[bus]],pmin)
        x['Qgmax'][genmap[bus]] += qmax
        x['Pcost'][genmap[bus]] += cost*pmax
    x['Pcost'] /= x['Pgmax'] # weighted average
    x['Pcost'][np.isnan(x['Pcost'])] = 0
    for i in range(4-1,-1,-1):
        if np.all(x[order[i]] == x[order[i]][0]):
            vdefault[order[i]] = x[order[i]][0]
            x.pop(order[i],None)
            inkde.pop(i)
    resg = {}
    resg['kde'] = kde_fit(np.vstack((x[order[i]] for i in inkde )), bw_method=bw_method)
    resg['max'] = np.array([np.max(x[order[i]]) for i in inkde])
    resg['min'] = np.array([np.min(x[order[i]]) for i in inkde])
    resg['order'] = order
    resg['inkde'] = inkde
    resg['vdefault'] = vdefault
    resg['actual_vars'] = actual_vars_g
    
    resf = {}
    resf['intermediate'] = sum(~bus_data['BUS_I'].isin(gen_data['GEN_BUS']) & np.all(bus_data.loc[:,['PD','QD']] == 0,axis=1))/N
    resf['gen']          = GBnum/N
    resf['gen_only']     = sum(bus_data['BUS_I'].isin(gen_data['GEN_BUS']) & np.all(bus_data.loc[:,['PD','QD']] == 0,axis=1))/N
    resf['Qd_Pd']        = sum(bus_data['QD'] > bus_data['PD'])/N
    resf['Qg_Pg']        = sum(gen_data['QMAX'] > gen_data['PMAX'])/GBnum
    resf['PgAvg']        = gen_data['PMAX'].sum()/GBnum
    resf['QgAvg']        = gen_data['QMAX'].sum()/GBnum

    return resd,resg,resf

def set_fmax(data, fmaxin):
    """ pick a maximum rating, which will be given to all branches with no limit (rate=0).
        This is taken as the maximum of a user-input fmaxin, and the maximum value in the data.
    """
    return max(fmaxin, data.max())

def multivariate_z(branch_data,bw_method='scott',actual_vars=True, mvabase=100., fmaxin=9., const_rate=False):

    fields = ['r','x','b','rate','tap','shift'] ;nf = len(fields)
    order = dict(zip(range(nf),fields))
    inkde = list(range(nf))
    vdefault = {}
    M = sum(branch_data.loc[:,'BR_STATUS'] > 0)
    fmax   = set_fmax(branch_data['RATE_A']/mvabase,fmaxin)
    x = {}
    for i,k in order.items():
        x[k] = np.empty(M)
    ptr = 0
    for R,X,B,rate,tap,shift,status in zip(branch_data['BR_R'], branch_data['BR_X'], branch_data['BR_B'], branch_data['RATE_A'], branch_data['TAP'], branch_data['SHIFT'],branch_data['BR_STATUS']):
        if status > 0:
            x['r'][ptr] = R; x['x'][ptr] = X ; x['b'][ptr] = B; 
            x['shift'][ptr] = shift*np.pi/180 # convert to radians from degrees
            if tap == 0:
                x['tap'][ptr] = 1.
            else:
                x['tap'][ptr] = tap
            if (rate == 0) or const_rate:
                x['rate'][ptr] = fmax
            else:
                x['rate'][ptr] = rate/mvabase
            ptr += 1
    for i in range(nf-1,-1,-1):
        if np.all(x[order[i]] == x[order[i]][0]):
            vdefault[order[i]] = x[order[i]][0]
            x.pop(order[i],None)
            inkde.pop(i)
    res = {}
    res['kde']      = kde_fit(np.vstack(x[order[i]] for i in inkde), bw_method=bw_method)
    res['actual_vars'] = actual_vars
    res['max']      = np.array([np.max(x[order[i]]) for i in inkde])
    res['min']      = np.array([np.min(x[order[i]]) for i in inkde])
    res['order']    = order
    res['inkde']    = inkde
    res['vdefault'] = vdefault
    res['xmean']    = np.mean(x['x'])
    try:
        res['RgX']  = sum(x['r'][i] > x['x'][i] for i in range(M))/M
    except KeyError:
        res['RgX']  = 0
    try:
        res['BgX']  = sum(x['b'][i] > x['x'][i] for i in range(M))/M
        res['b0']       = sum(x['b'] == 0)/M
        res['bmean']    = np.mean(x['b'])
    except KeyError:
        res['BgX']   = 0
        res['b0']    = 0
        res['bmean'] = 0
    return res, fmax

def kde_fit(x,bw_method='scott'):
    """ returns a kde object fit to the values in x """
    return stats.gaussian_kde(x,bw_method=bw_method)

def pchip_test(data,str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    P = hlp.PchipDist(data)
    p = P.P.derivative()
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.hist(data,bins='auto',normed=True)
    ax.plot(np.linspace(P.min,P.max),P.pdf(np.linspace(P.min,P.max)))
    t = 'integral = %0.2f' %(p.integrate(a=P.min,b=P.max))
    ax.text(0.5,0.9,t,verticalalignment='top',transform=ax.transAxes)
    ax = fig.add_subplot(312)
    ax.hist(data,bins='auto',normed=True,cumulative=True)
    ax.plot(np.linspace(P.min,P.max),P(np.linspace(P.min,P.max)))

    ax = fig.add_subplot(313)
    s = P.resample(len(data))
    _,bins,_ = ax.hist(data,bins='auto',normed=True)
    ax.hist(s, bins=bins, normed=True)

    fig.savefig('pchip_fit_test_%s.png' %(str))

def main(fname, fit='pchip'):
    bus_data, gen_data, branch_data = hlp.load_data(fname)
    Pd = bus_data['PD'].values
    x = branch_data['BR_X'].values
    Pg = np.zeros(bus_data.shape[0])
    for bus,v in zip(gen_data['GEN_BUS'],gen_data['PG']):
        Pg[bus] += v
    
    #pchip_test(Pg[Pg>0],'gen')
    #pchip_test(Pd[Pd>0],'load')
    #pchip_test(x,'reactance')
    #sys.exit(0)
    Pgout, Pdout, frac = analyze_power_statistics(Pg, Pd, fit=fit)
    pickle.dump(Pgout,open('%s_power_Pg_%sfit.pkl' %(fname, fit), 'wb'))
    pickle.dump(Pdout,open('%s_power_Pd_%sfit.pkl' %(fname, fit), 'wb'))
    pickle.dump(frac,open('%s_power_frac.pkl' %(fname), 'wb'))

    fitx = analyze_reactance_statistics(x)
    pickle.dump(fitx,open('%s_reactance_%sfit.pkl' %(fname, fit), 'wb'))

if __name__ == "__main__":
    fname = sys.argv[1]
    try:
        fit = sys.argv[2]
    except IndexError:
        fit = 'pchip'
    import ipdb; ipdb.set_trace()
    bus_data, gen_data, branch_data, gen_cost = hlp.load_data(fname)
    resz = multivariate_z(branch_data)
    resd,resg,resf = multivariate_power(bus_data,gen_data, gen_cost=gen_cost)
    ipdb.set_trace()
    sys.exit(0)
    Pd = bus_data['PD'].values
    x = branch_data['BR_X'].values
    Pg = np.zeros(bus_data.shape[0])
    for bus,v in zip(gen_data['GEN_BUS'],gen_data['PG']):
        Pg[bus] += v
    
    #pchip_test(Pg[Pg>0],'gen')
    #pchip_test(Pd[Pd>0],'load')
    #pchip_test(x,'reactance')
    #sys.exit(0)
    Pgout, Pdout, frac = analyze_power_statistics(Pg, Pd, fit=fit)
    pickle.dump(Pgout,open('%s_power_Pg_%sfit.pkl' %(fname, fit), 'wb'))
    pickle.dump(Pdout,open('%s_power_Pd_%sfit.pkl' %(fname, fit), 'wb'))
    pickle.dump(frac,open('%s_power_frac.pkl' %(fname), 'wb'))
                                                                         
    fitx = analyze_reactance_statistics(x)
    pickle.dump(fitx,open('%s_reactance_%sfit.pkl' %(fname, fit), 'wb'))
