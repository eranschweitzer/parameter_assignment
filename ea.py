import pickle
import numpy as np
import gurobipy as gb
import formulation_ea as fm
import multvar_solve as slv
import multvar_solution_check as chk

def mutate(Psi0,K,pm=0.05):
    Psi = EAgeneration(Psi0.inputs)
    wmax = 0
    if not Psi0.isnone():
        Psi0.order()
        wmax = Psi0.wmax
    for i in range(K):
        r = np.random.random()
        if (Psi0.isnone()) or (r > wmax):
            Psi.append( EAindividual(Psi.inputs['globals']['G'].number_of_edges()) )
        else:
            x = select_psi(r,Psi0).copy()
            x.permute(pm) 
            Psi.append(x)
    return Psi

def select_psi(r,Psi):
    """ IMPORTNAT!!!! it is assumed that Psi is ordered! """
    i = 0
    while Psi[i].w < r:
        i += 1
    return Psi[i]

class EAindividual(object):
    def __init__(self, L, Z=None):
        if Z is None:
            self.Z = np.random.permutation(L)
        else:
            self.Z = Z.copy()
        self.w    = None
        self.f    = None
        self.opt  = None
        self.vars = None
        self.zones = []
        self.Slabels = ['Pgmax', 'Pgmin', 'Qgmax', 'Pd', 'Qd']

    def initialize_zones(self, inputs):
        ### initalize zones
        for i,v in enumerate(inputs['locals']):
            self.zones.append(EAzone(v, inputs['globals'], self.Z, zone=i))
    
    def copy(self):
        return EAindividual(self.Z.shape[0], Z=self.Z)
    
    def permute(self,pm):
        for i in range(self.Z.shape[0]):
            if np.random.random() < pm:
                swap         = np.random.randint(len(self.Z))
                _v           = self.Z[i]
                self.Z[i]    = self.Z[swap]
                self.Z[swap] = _v

    def solve(self,inputs,logging=None,**kwargs):
        if logging is not None:
            try:
                log_iterations = logging['log_iterations']
            except KeyError:
                log_iterations = None
        if len(self.zones) == 0:
            raise(BaseException("No Zones initialized"))
        elif len(self.zones) == 1:
            self.opt = self.zones[0]
            if logging is not None:
                logging['log_single_system'](self.opt, start=True)
            self.opt.optimize()
            if logging is not None:
                logging['log_single_system'](self.opt, start=False)
        else:
            i = 0
            while True:
                if logging is not None:
                    logging['log_iteration_start'](i,inputs['globals']['consts']['rho'])
                beta_bar, gamma_bar, ivals = slv.solve(self.zones, inputs['globals']['e2z'],logging=log_iterations,**kwargs)
                if logging is not None:
                    logging['log_iteration_summary'](beta_bar,gamma_bar, ivals)
                flag,msg = slv.termination(i, ivals, inputs['globals']['consts']['thresholds'])
                if flag:
                    if logging is not None:
                        logging['log_termination'](msg)
                    break
                else:
                    slv.update(self.zones, i, beta_bar, gamma_bar, inputs['globals']['consts']['rho'], **kwargs)
                    i += 1
                    if i == 1:
                        for zone in self.zones:
                            zone.set_timelimit(1500)
            if logging is not None:
                logging['log_single_system'](self.opt, start=True)
            self.fullsolve(inputs)
            if logging is not None:
                logging['log_single_system'](self.opt, start=False)
        self.set_f()
        self.set_vars(inputs['globals']['z']) 
        self.sol_check(G=inputs['globals']['G'])

    def sol_check(self,**kwargs):
        chk.rescheck(self.vars,**kwargs)

    def fullsolve(self, inputs):
        ## set nodal variables
        self.joint_s(inputs['globals']['G'].number_of_nodes())
        self.opt = fm.ZoneMILP(inputs['globals']['G'], inputs['globals']['consts'], {'z':inputs['globals']['z'], 'S': self._S}, self.Z, nperm=True)
        self.opt.optimize()

    def joint_s(self, N):
        self._S = {}
        #### initialize empty arrays of size N
        for l in self.Slabels:
            self._S[l] = np.empty(N)
        ### populate arrays with placed values from zones
        for i,zone in enumerate(self.zones):
            _v = zone.getvars(Sonly=True)
            for k in self._S:
                if k in ['Pd','Qd']:
                    mult = 100
                else:
                    mult = 1
                self._S[k][zone.rnmap] = _v[k]*mult
        ### copy the rest of keys of S that are no already in vars
        for k,v in self.zones[0].S.items():
            if k not in self._S:
                self._S[k] = v

    def set_f(self):
        self.f = self.opt.objective
    
    def set_vars(self, z):
        self.vars = {}
        _vars = self.opt.getvars()
        for k,v in _vars.items():
            self.vars[k] = np.empty(v.shape[0])
            if v.shape[0] == self.opt.N:
                self.vars[k][self.opt.rnmap] = _vars[k]
            elif v.shape[0] == self.opt.L:
                self.vars[k][self.opt.rlmap] = _vars[k]
            else:
                raise(ValueError("Inocrrect vector shape. key: %s" %(k)))
        ### add power inputs if they were not optimization variables.
        for l in self.Slabels:
            if l not in self.vars:
                self.vars[l] = self._S[l]
        ### add branch variables
        for k,v in z.items():
            try:
                if v.shape[0] == self.opt.L:
                    self.vars[k] = v[self.Z]
            except AttributeError:
                pass

class EAzone(fm.ZoneMILP):
    def __init__(self, locals, globals, zperm, zone=0):
        super().__init__(locals['G'], globals['consts'], {'z':globals['z'], 'S': locals['S']}, zperm, 
                         ebound=locals['ebound'], ebound_map=locals['ebound_map'], zone=zone)
    def sol_check(self):
        vars = self.getvars(includez=True)
        ebound_map = self.m._ebound_map
        vars['beta'] = {i:self.beta[i].X for i in self.beta}
        vars['gamma'] = {i:self.gamma[i].X for i in self.gamma}
        try:
            vars['beta_p'] = {i:self.beta_p[i].X for i in self.beta_p}
            vars['beta_n'] = {i:self.beta_n[i].X for i in self.beta_n}
            vars['gamma_p'] = {i:self.gamma_p[i].X for i in self.gamma_p}
            vars['gamma_n'] = {i:self.gamma_n[i].X for i in self.gamma_n}
        except gb.GurobiError:
            pass
        try:
            vars['beta2']     = {i:self.beta2[i].X     for i in self.beta2}
            vars['gamma2']    = {i:self.gamma2[i].X    for i in self.beta2}
            vars['beta_bar']  = {i:self.beta_bar[i]    for i in self.beta2}
            vars['gamma_bar'] = {i:self.gamma_bar[i]   for i in self.beta2}
        except AttributeError:
            pass
        maps = {'nmap':self.nmap, 'lmap': self.lmap, 'rnmap': self.rnmap, 'rlmap': self.rlmap}
        chk.rescheck(vars,G=self.G, maps=maps, ebound_map=ebound_map)


class EAgeneration(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.Psi    = None

    def __iadd__(self,b):
        """ !!!!IMPORTANT!!!!
        It is assumed that the same inputs were used to
        generate both generations
        """
        if self.inputs is not b.inputs:
            raise(BaseException("When adding generations they must have the same inputs"))

        if self.Psi is None:
            self.Psi = []
        self.Psi.extend(b.Psi)
        return self

    def __getitem__(self,x):
        return self.Psi[x]
    
    def iter(self):
        return self.Psi
    
    def isnone(self):
        return self.Psi is None

    def append(self, x):
        if self.Psi is None:
            self.Psi = []
        self.Psi.append(x)

    def initialize_optimization(self, logging=None, res=0.1):
        T = len(self.Psi)
        for i, psi in enumerate(self.Psi):
            if logging is not None:
                logging(i,T, res=res)
            psi.initialize_zones(self.inputs)

    def mutate(self,K,**kwargs):
        return mutate(self,K,**kwargs)
    
    def selection(self,kappa):
        """ select the best kappa individuals in Psi
        Since the ordering routine places the individuals in order
        we simply order and then remove all entries in the list
        that are greater than kappa """
        self.order()
        del self.Psi[kappa:len(self.Psi)]

    def order(self):
        """ invert objective and find the largest one """
        C = 0
        for psi in self.Psi:
            psi.w = 1./psi.f
            C = max(C,psi.w)
        
        """ sorting order in descending order (hence the -psi.w) """
        idx = np.argsort([-psi.w for psi in self.Psi])
        self.Psi = [self.Psi[i] for i in idx]

        """ normalize by the largest weight and form cumulative weight"""
        for i,psi in enumerate(self.Psi):
            if i > 0:
                psi.w = psi.w/C + self.Psi[i-1].w
            else:
                psi.w = psi.w/C
        """ round sum up to the nearest integer """
        C = np.ceil(self.Psi[-1].w)

        """ renormalize """
        for psi in self.Psi:
            psi.w = psi.w/C

        self.wmax = self.Psi[-1].w

    def save(self, filename, timestamps):
        saveparts = filename.split('.') 
        _tmp = {'G': self.inputs['globals']['G'], 'vars':[] }
        for psi in self.iter():
            _tmp['vars'].append(psi.vars)
        pickle.dump(_tmp,\
                open(saveparts[0] + timestamps['end'] + "inputstamp_" + timestamps['start'] + "." + saveparts[1],'wb'))
