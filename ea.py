import os
import sys
import pickle
import numpy as np
import gurobipy as gb
import formulation_ea as fm
import helpers as hlp
import multvar_solve as slv
import multvar_solution_check as chk
import makempc as mmpc
import multiprocessing as mlt
import subprocess
import logfun as lg

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
        self.ind  = None
        self.zones = []
        self.Slabels = ['Pgmax', 'Pgmin', 'Qgmax', 'Pd', 'Qd']

    def initialize_zones(self, inputs):
        ### initalize zones
        for i,v in enumerate(inputs['locals']):
            self.zones.append(EAzone(v, inputs['globals'], self.Z, zone=i))
    
    def copy(self):
        return EAindividual(self.Z.shape[0], Z=self.Z)

    def cleanup(self):
        self.opt = None
        self.zones = []
    
    def permute(self,pm):
        for i in range(self.Z.shape[0]):
            if np.random.random() < pm:
                swap         = np.random.randint(len(self.Z))
                _v           = self.Z[i]
                self.Z[i]    = self.Z[swap]
                self.Z[swap] = _v

    def add_id(self,i):
        self.ind = i

    def solve(self,inputs,logging=None, parallel=False, parallel_zones=False, **kwargs):
        if logging is not None:
            try:
                log_iterations = logging['log_iterations']
            except KeyError:
                log_iterations = None
            if 'logger' not in logging:
                logging['logger'] = None
        else:
            log_iterations = None
        if len(self.zones) == 0:
            if parallel_zones is True:
                self.parsolve(inputs,logging=logging,**kwargs)
                if logging is not None:
                    logging['log_single_system'](self.opt, start=True, logger=logging['logger'])
                self.fullsolve(inputs, logger=logging['logger'], calcS=False)
                if logging is not None:
                    logging['log_single_system'](self.opt, start=False, logger=logging['logger'])
            else:
                raise(BaseException("No Zones initialized"))
        elif len(self.zones) == 1:
            self.opt = self.zones[0]
            if logging is not None:
                logging['log_single_system'](self.opt, start=True, logger=logging['logger'])
            self.opt.optimize()
            if logging is not None:
                logging['log_single_system'](self.opt, start=False, logger=logging['logger'])
        else:
            i = 0
            while True:
                if logging is not None:
                    method = kwargs.get("rho_update", 'None')
                    logging['log_iteration_start'](i, slv.rho_modify(inputs['globals']['consts']['rho'],i, method), logger=logging['logger'] )
                beta_bar, gamma_bar, ivals = slv.solve(self.zones, inputs['globals']['e2z'],logging=log_iterations, logger=logging['logger'], **kwargs)
                if logging is not None:
                    logging['log_iteration_summary'](beta_bar,gamma_bar, ivals, logger=logging['logger'])
                flag,msg = slv.termination(i, ivals, inputs['globals']['consts']['thresholds'])
                if flag:
                    if logging is not None:
                        logging['log_termination'](msg, logger=logging['logger'])
                    break
                else:
                    slv.update(self.zones, i, beta_bar, gamma_bar, inputs['globals']['consts']['rho'], **kwargs)
                    i += 1
                    if i == 1:
                        for zone in self.zones:
                            zone.set_timelimit(1500)
            if logging is not None:
                logging['log_single_system'](self.opt, start=True, logger=logging['logger'])
            self.fullsolve(inputs, logger=logging['logger'])
            if logging is not None:
                logging['log_single_system'](self.opt, start=False, logger=logging['logger'])
        self.set_f()
        self.set_vars(inputs['globals']['z']) 
        self.sol_check(G=inputs['globals']['G'], logger=logging['logger'])
        if parallel:
            self.cleanup()

    def sol_check(self,**kwargs):
        chk.rescheck(self.vars,**kwargs)

    def fullsolve(self, inputs, logger=None, calcS=True):
        ## set nodal variables
        if calcS:
            self.joint_s(inputs['globals']['G'].number_of_nodes())
        self.opt = fm.ZoneMILP(inputs['globals']['G'], inputs['globals']['consts'], {'z':inputs['globals']['z'], 'S': self._S}, self.Z, nperm=True)
        self.opt.optimize(logger=logger)

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
        ### copy the rest of keys of S that are not already in vars
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

    def parsolve(self, inputs, logging=None, **kwargs):
       
        e2z   = inputs['globals']['e2z']
        zones = []
        conns = []
        for i,v in enumerate(inputs['locals']):
            parent_conn, child_conn = mlt.Pipe()
            conns.append(parent_conn)
            zones.append( mlt.Process(target=parzone, args=(v, inputs['globals'], self.Z, i, logging, kwargs, child_conn)) )
    
        ### start processes
        for z in zones:
            z.start()
        
        iter = 0 
        while True:
            if logging is not None:
                method = kwargs.get("rho_update", 'None')
                logging['log_iteration_start'](iter, slv.rho_modify(inputs['globals']['consts']['rho'],iter, method), logger=logging['logger'] )
            beta_bar = {l:0 for l in e2z}
            gamma_bar= {l:0 for l in e2z}
    
            ### wait for beta and gamma
            beta = {}; gamma = {}
            for i, conn in enumerate(conns):
                data = conn.recv()
                if data is False:
                    break
                beta[i] = data['beta']; gamma[i] = data['gamma']
            if data is False:
                break
    
            ### calculate averages
            for i in beta:
                for l in beta[i]:
                    beta_bar[l]  += beta[i][l]/2
                    gamma_bar[l] += gamma[i][l]/2
    
            ### calculate errors
            gap = {'beta':0, 'gamma':0}
            beta_diff = {}; gamma_diff= {}
            for l in beta_bar:
                beta_diff[l]   = np.abs( beta[e2z[l][0]][l] - beta[e2z[l][1]][l] )
                gamma_diff[l]  = np.abs( gamma[e2z[l][0]][l] - gamma[e2z[l][1]][l] )
                for i in e2z[l]:
                    gap['beta'] += (beta[i][l]  - beta_bar[l])**2
                    gap['gamma']+= (gamma[i][l] - gamma_bar[l])**2
            mean_diff = {'beta':  sum(beta_diff.values())/len(beta_diff),
                         'gamma': sum(gamma_diff.values())/len(gamma_diff)}
            max_diff  = {'beta': max(beta_diff.values()), 'gamma': max(gamma_diff.values())}
    
            if logging is not None:
                logging['log_iteration_summary'](beta_bar,gamma_bar, {'gap':gap, 'mean_diff':mean_diff, 'max_diff':max_diff}, logger=logging['logger'])
                logging['log_iteration_summary'](beta_bar,gamma_bar, {'gap':gap, 'mean_diff':mean_diff, 'max_diff':max_diff}, ind=self.ind, iter=iter)
            ### check termination
            flag, msg = slv.termination(iter, {'gap':gap, 'mean_diff':mean_diff, 'max_diff':max_diff}, inputs['globals']['consts']['thresholds']) 
    
            ### send termination flag
            for conn in conns:
                conn.send(flag)
            if flag:
                if logging is not None:
                    logging['log_termination'](msg, logger=logging['logger'])
                break
    
            #### send values for update
            for conn in conns:
                conn.send({'beta_bar':beta_bar, 'gamma_bar': gamma_bar})
            iter += 1

        # if any of the zones failed terminate all processes and restart after 
        # randomly changing the Z permutation
        if data is False:
            lg.log_reset(self.ind)
            for p in zones:
                p.terminate()
            for conn in conns:
                conn.close()
            self.Z = np.random.permutation(self.Z)
            self.parsolve(inputs,logging=logging,**kwargs)
            return

        ### receive S variables from zones (mod. of joint_s)
        self._S = {}
        #### initialize empty arrays of size N
        for l in self.Slabels:
            self._S[l] = np.empty(inputs['globals']['G'].number_of_nodes())
        for i, conn in enumerate(conns):
            data = conn.recv()
            for k in self._S:
                if k in ['Pd','Qd']:
                    mult = 100
                else:
                    mult = 1
                self._S[k][data['rnmap']] = data['S'][k]*mult
            conn.close()
        ### copy the rest of keys of S that are not already in vars
        for k,v in inputs['locals'][0]['S'].items():
            if k not in self._S:
                self._S[k] = v
    
def parzone(locals, globals, zperm, zone, logging, kwargs, conn):
    #### INITIALIZE ####
    s = fm.ZoneMILP(locals['G'], globals['consts'], {'z':globals['z'], 'S': locals['S']}, zperm, ebound=locals['ebound'], ebound_map=locals['ebound_map'], zone=zone)
    iter = 0 
    while True:
        #### solve zone ####
        s.optimize(logger=logging['logger'],**kwargs)
        if s.m.status not in [2,11,9]:
            conn.send(False)
            return
        with mlt.Lock():
            logging['log_iterations'](s,logger=logging['logger'], zone=zone)
        if kwargs.get('solck', False):
            with mlt.Lock():
                s.sol_check()

        conn.send({'beta': s.get_beta(), 'gamma': s.get_gamma()})
        
        ### check whether to exit
        flag = conn.recv()
        if flag:
            break

        ### wait to receive the average values
        data = conn.recv()

        ### update model
        if iter == 0:
            s.m._solmin = 1
            if kwargs.get("remove_abs", True):
                s.remove_abs_vars()
        method = kwargs.get("rho_update", 'None')
        s.objective_update(data['beta_bar'], data['gamma_bar'], slv.rho_modify(globals['consts']['rho'], iter, method) )
        iter += 1
        if iter == 1:
            s.set_timelimit(1500)

    ### send variables
    conn.send({'S':s.getvars(Sonly=True), 'rnmap': s.rnmap})

class EAzone(fm.ZoneMILP):
    def __init__(self, locals, globals, zperm, zone=0):
        super().__init__(locals['G'], globals['consts'], {'z':globals['z'], 'S': locals['S']}, zperm, 
                         ebound=locals['ebound'], ebound_map=locals['ebound_map'], zone=zone)


class EAgeneration(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.Psi    = None

    def __iadd__(self,b):
        """ !!!!IMPORTANT!!!!
        It is assumed that the same inputs were used to
        generate both generations
        """
        #if self.inputs is not b.inputs:
        #    raise(BaseException("When adding generations they must have the same inputs"))

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

    def set_ind_id(self):
        for i, psi in enumerate(self.Psi):
            psi.add_id(i)

    def split(self, N):
        """ split generation into N copies, as equal as possible """
        base_load = len(self.Psi)//int(N)
        extra     = len(self.Psi) % N
        out = []
        for i in range(N):
            out.append(EAgeneration(self.inputs))
            out[-1].Psi = self.Psi[0:base_load]; del self.Psi[0:base_load]
            if extra > 0:
                out[-1].Psi.append(self.Psi.pop(0))
                extra -= 1
        return out

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

    def save(self, filename, timestamps, ftype='pkl', **kwargs):
        saveparts = filename.split('.') 
        base_str = saveparts[0] + timestamps['end'] + "inputstamp_" + timestamps['start'] + "."

        _tmp = {'G': self.inputs['globals']['G'], 'vars':[] }
        for psi in self.iter():
            _tmp['vars'].append(psi.vars)

        if ftype == 'pkl':
            pickle.dump(_tmp, open( base_str + "pkl", 'wb') )
        elif ftype == 'mpc':
            mmpc.savempc(_tmp, base_str + "mat", **kwargs)
    
    def parallel_solve(self, psi, ind, conn, s, base):
        #psi = args[0]; ind = args[1]
        with s:
            lg.log_parind(ind, start=True)
            if base is None:
                base = self.inputs['globals']['consts']['saving']['logpath']
            fname = hlp.savepath_replace(self.inputs['globals']['consts']['saving']['savename'], base).split('.')[0] + "_ind" + str(ind) + ".log"
            logger = lg.logging_setup(fname=fname, logger='ind%d' %(ind), ret=True)
            lgslv = self.inputs['globals']['consts'].get("logging",None)
            try:
                lgslv['logger'] = logger
            except TypeError:
                pass
            if not self.inputs['globals']['consts']['parallel_opt']['parallel_zones']:
                psi.initialize_zones(self.inputs)
            psi.solve(self.inputs, logging=lgslv, parallel=True, parallel_zones=self.inputs['globals']['consts']['parallel_opt']['parallel_zones'], **self.inputs['globals']['consts']['solve_kwargs'])
            lg.log_parind(ind, start=False)
        conn.send(psi)
        #return psi

    def parallel_wrap(self, logname=None):
        #p = mlt.Pool(min(5,len(self.Psi)))
        if logname is not None:
            lg.logging_setup(fname=logname)
            logname = "/".join(logname.split('/')[:-1]) + '/'
        s = mlt.Semaphore(min(5,len(self.Psi)))
        conns = []
        jobs  = []
        for i, psi in enumerate(self.Psi):
            try:
                ind = psi.ind
            except AttributeError:
                ind = i
            parent_conn, child_conn = mlt.Pipe()
            conns.append(parent_conn)
            jobs.append( mlt.Process( target=self.parallel_solve, args=(psi, ind, child_conn, s, logname), daemon=False ) )

        for j in jobs:
            j.start()

        for i, conn in enumerate(conns):
            self.Psi[i] = conn.recv()

    def outsource(self,logfile=None):
        """ - split up individuals into processes and pickle
            - ssh into the workers in the list and start their process (possible run own process as well)
            - wait until the data has been written i.e. all processes completed
            - read in data, merge lists and return """
        basepath = "/".join(os.path.realpath(__file__).split('/')[:-1]) + '/'
        if logfile is not None:
            if logfile[0] != '/':
                _tmp = logfile.split('.')[0]
                logfile = basepath + _tmp + '_'
        workers   = self.inputs['globals']['consts']['parallel_opt']['workers']
        dump_path = self.inputs['globals']['consts']['parallel_opt']['dump_path']
        if dump_path[-1] != '/':
            dump_path += '/'
        if dump_path[0] != '/':
            dump_path = basepath + dump_path

        if 'self' in workers:
            # make sure self is the last job
            del workers[workers.index('self')]
            workers.append('self')
        self.set_ind_id() # set id so there won't be issues with logging files
        parts = self.split(len(workers))
        #### save pickle files
        savenames = [dump_path + w + '.pkl' for w in workers if w != 'self']
        for i, w in enumerate(workers):
            if w != 'self':
                pickle.dump(parts[i], open(savenames[i],'wb'))
         
        ### start subprocesses
        proc = []
        for i, w in enumerate(workers):
            if w != 'self':
                lg.log_outsource(w,parts[i],savenames[i])
                proc.append(subprocess.Popen(["ssh", w, 'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/gurobi751/linux64/lib GRB_LICENSE_FILE=/opt/gurobi.lic', sys.executable, os.path.realpath(__file__), savenames[i], logfile + w + '.log']))
                pass #start prosses
            else:
                parts[i].parallel_wrap()
                self += parts[i]
        ### wait for processes to exit
        for i, p in enumerate(proc):
            p.wait()
            lg.log_outsource_wait(workers[i],p.returncode)

        ### read in data
        for i, w in enumerate(workers):
            if w != 'self':
                lg.log_outsource_collect(w)
                self += pickle.load(open(savenames[i],'rb'))


if __name__ == '__main__':
    Psi = pickle.load(open(sys.argv[1],'rb'))
    Psi.parallel_wrap(logname=sys.argv[2])
    pickle.dump(Psi, open(sys.argv[1],'wb'))
    sys.exit(0)
