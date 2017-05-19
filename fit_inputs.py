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

def kde_fit(x):
    """ returns a kde object fit to the values in x """
    return stats.gaussian_kde(x)

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
