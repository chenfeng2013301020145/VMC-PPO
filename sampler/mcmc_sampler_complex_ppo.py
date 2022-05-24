# encoding: utf-8
# compatible with 2d sysytem
import sys
sys.path.append('..')

import torch
import numpy as np
import multiprocessing
import os

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse_indices, counts = torch.unique(x, sorted=True, return_inverse=True, 
                                           return_counts=True, dim=dim)
    perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype,
                        device=inverse_indices.device)
    inverse, perm = inverse_indices.flip([0]), perm.flip([0])
    indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, indices, counts
 
def _get_unique_states(states, sym_states, logphis, thetas):
    """
    Returns the unique states, corresponding psis and the counts.
    """
    _, indices, counts = np.unique(states[:,0,:], return_index=True, return_counts=True, axis=0)
    states = states[indices]
    sym_states = sym_states[indices]
    logphis = logphis[indices]
    thetas = thetas[indices]
    return states, sym_states, logphis, thetas, counts

class MCsampler():
    def __init__(self,**kwargs):
        self._state_size = kwargs.get('state_size')
        self._model = kwargs.get('model')
        self._n_sample = kwargs.get('n_sample', 1000)
        self._update_operator = kwargs.get('updator')
        self._op = kwargs.get('operator')
        self._get_init_state = kwargs.get('get_init_state')
        self._init_type = kwargs.get('init_type', 'rand')
        self._threads = kwargs.get('threads',1)
        self._update_size = self._op._update_size
        self._dimension = len(self._state_size) - 1
        self.warmup_length = 500
        self.acceptance = False
        self.cal_ops = True

        Dp = self._state_size[-1]
        if self._dimension == 1:
            N = self._state_size[0]
            self._single_state_shape = [Dp, N]
            self.N = N
        else:
            length = self._state_size[0]
            width = self._state_size[1]
            self._single_state_shape = [Dp, length, width]
            self.N = length*width
        self.pow_list = np.arange(self.N-1, -1, -1)
        self._updator = self._update_operator(self._state_size)
        self._state0, self._state0_v = self._get_init_state(self._state_size, kind=self._init_type, n_size=self._threads)
            
    def find_states_and_ops(self, states):
        with torch.no_grad():
            n_sample = states.shape[0]
            update_states = np.zeros([n_sample, self._update_size] + self._single_state_shape)
            update_psis = np.zeros([n_sample, self._update_size, 2])
            update_coeffs = np.zeros([n_sample, self._update_size])
            efflens = np.zeros([n_sample], dtype=np.int64)
            
            if self.cal_ops:
                for i,state in enumerate(states):
                    update_states[i], update_coeffs[i], efflen = self._op.find_states(state)
                    efflens[i] = efflen
                    ustates = update_states[i,:efflen,:].reshape([-1]+self._single_state_shape)
                    upsis = self._model(torch.from_numpy(ustates).float())
                    update_psis[i,:efflen,:] = upsis.numpy().reshape([1, efflen, 2])
            else:
                for i,state in enumerate(states):
                    update_states[i], update_coeffs[i], efflen = self._op.find_states(state)
                    efflens[i] = efflen
                    ustates = update_states[i,:efflen,:].reshape([-1]+self._single_state_shape)
                    ustates = self._model.pick_sym_config(torch.from_numpy(ustates)).numpy()
                    update_states[i,:efflen,:] = ustates.reshape(update_states[i,:efflen,:].shape)
        return update_states, update_psis, update_coeffs, efflens
    
    def _generate_updates(self, states, threads):
        """
        Generates updated states and coefficients for an Operator.

        Args:
            states: The states with shape (batch size, shape of state).
            operator: The operator used for updating the states.
            state_size: shape of a state in states
            update_size: number of update_states

        Returns:
            The updated states and their coefficients. The shape of the updated
            states is (batch size, num of updates, shape of state), where num of
            updates is the largest number of updated states among all given states.
            If a state has fewer updated states, its updates are padded with the
            original state.

        """
        torch.set_num_threads(1)

        ustates = np.empty(threads, np.ndarray)
        #sym_ustates = np.empty(threads, np.ndarray)
        upsis = np.empty(threads, np.ndarray)
        ucoeffs = np.empty(threads, np.ndarray)
        efflens = np.empty(threads, np.ndarray)
        
        pool = multiprocessing.Pool(threads)
        results = []
        cnt = 0
        
        for state in states:
            results.append(pool.apply_async(self.find_states_and_ops, (state,)))
        pool.close()
        pool.join()

        for cnt, res in enumerate(results):
            ustates[cnt], upsis[cnt], ucoeffs[cnt], efflens[cnt] = res.get()

        return (np.concatenate(ustates, axis=0).reshape([-1, self._update_size]+self._single_state_shape),
                np.concatenate(upsis, axis=0).reshape([-1, self._update_size, 2]),
                np.concatenate(ucoeffs, axis=0).reshape([-1, self._update_size]), 
                np.concatenate(efflens, axis=0).reshape(-1))
       
    def get_single_sample(self, state: np.ndarray, sym_states_i, logphi_i, theta_i, mask, rand):
        # size of sym_state: (batch_size, sym_N, Dp, L, W)
        with torch.no_grad():
            state_f = state
            while np.sum(np.abs(state_f - state)) < 1e-4:
                state_f = self._updator._get_update(state, mask)
                mask = self._updator.generate_mask(1)

            psi_f = self._model(torch.from_numpy(state_f[None,...]).float())
            logphi_f = psi_f[:,0].numpy()
            theta_f = psi_f[:,1].numpy()
            delta_logphi = logphi_f - logphi_i

            # if delta_logphi>0 or rand<=np.exp(delta_logphi*2.0) or self.acceptance:
            if rand <= min(np.exp(delta_logphi*2.0), 1.) or self.acceptance:
                #sym_state_f, _ = self._model.symmetry(torch.from_numpy(state_f[None,...]).float())
                sym_state_f = self._model.pick_sym_config(torch.from_numpy(state_f[None,...]))
                sym_states_f, _ = self._model.symmetry(sym_state_f)
                sym_states_f = sym_states_f.numpy()
                return state_f, sym_states_f, logphi_f, theta_f
            else:
                return state, sym_states_i, logphi_i, theta_i
    
    def warmup_sample(self, state: np.ndarray, logphi_i, theta_i, mask, rand):
        with torch.no_grad():
            state_f = state
            while np.sum(np.abs(state_f - state)) < 1e-4:
                state_f = self._updator._get_update(state, mask)
                mask = self._updator.generate_mask(1)

            psi_f = self._model(torch.from_numpy(state_f[None,...]).float())
            logphi_f = psi_f[:,0].numpy()
            theta_f = psi_f[:,1].numpy()
            delta_logphi = logphi_f - logphi_i

            if delta_logphi>0 or rand<=np.exp(delta_logphi*2.0) or self.acceptance:
            # if rand <= min(np.exp(delta_logphi*2.0), 1.) or self.acceptance:
                return state_f, logphi_f, theta_f
            else:
                return state, logphi_i, theta_i
          
    def _mh_sampler(self, n_sample_per_thread: int, state0, seed_number):
        """
        Importance sampling with Metropolis-Hasting algorithm

        Returns:
            state_sample_per_thread: (n_sample_per_thread, Dp, N)
            sym_state_sample_per_thread: (n_sample_per_thread, symN, Dp, N)
            logphi_sample_per_thread: (n_sample_per_thread)
            theta_sample_per_thread: (n_sample_per_thread)
        """
        # empty tensor storing the sample data
        state_sample_per_thread = np.zeros([n_sample_per_thread] + self._single_state_shape)
        sym_states_sample_per_thread = np.zeros([n_sample_per_thread, self._model.sym_N] + self._single_state_shape)
        logphi_sample_per_thread = np.zeros(n_sample_per_thread)
        theta_sample_per_thread = np.zeros(n_sample_per_thread)

        with torch.no_grad():
            np.random.seed(seed_number)
            masks = self._updator.generate_mask(n_sample_per_thread+self.warmup_length)
            rands = np.random.rand(n_sample_per_thread+self.warmup_length).astype(np.float32)

            state = state0
            psi = self._model(torch.from_numpy(state0[None,...]).float())
            logphi = psi[:,0].numpy()
            theta = psi[:,1].numpy()
            i = 0
            cnt = 0
            
            # warmup section
            while cnt < self.warmup_length:
                state, logphi, theta = self.warmup_sample(state, logphi, theta, masks[cnt], rands[cnt])
                cnt += 1
            
            sym_state = self._model.pick_sym_config(torch.from_numpy(state[None,...]))
            sym_states, _ = self._model.symmetry(sym_state)
            sym_states = sym_states.numpy()
            # sample section
            while i < n_sample_per_thread:
                state, sym_states, logphi, theta \
                       = self.get_single_sample(state, sym_states, logphi, theta, masks[cnt], rands[cnt])     
                state_sample_per_thread[i] = state
                sym_states_sample_per_thread[i] = sym_states
                logphi_sample_per_thread[i] = logphi
                theta_sample_per_thread[i] = theta
                i += 1
                cnt += 1

        return state_sample_per_thread, sym_states_sample_per_thread,\
               logphi_sample_per_thread, theta_sample_per_thread

    def warmup_mc_chain(self, n_sample_per_thread: int, state0, seed_number):        
        with torch.no_grad():
            np.random.seed(seed_number)
            masks = self._updator.generate_mask(n_sample_per_thread)
            rands = np.random.rand(n_sample_per_thread)
            
            state = state0
            psi = self._model(torch.from_numpy(state0[None,...]).float())
            logphi = psi[:,0].numpy()
            theta = psi[:,1].numpy()
            
            i = 0
            while i < n_sample_per_thread:
                state, logphi, theta = self.warmup_sample(state, logphi, theta, masks[i], rands[i])
                i += 1
        return state
    
    def first_warmup(self, steps):
        torch.set_num_threads(1)

        pool = multiprocessing.Pool(self._threads)
        state_list = np.zeros([self._threads] + self._single_state_shape)
                
        results = []
        seed_list = np.random.choice(10000, size=self._threads)*2
        for i in range(self._threads):
            results.append(pool.apply_async(self.warmup_mc_chain,
                        (steps, self._state0[i], seed_list[i], )))
        pool.close()
        pool.join()

        cnt = 0
        for res in results:
            state_list[cnt] = res.get()
            cnt += 1
            
        self._state0 = state_list
        return 
    
    def get_new_samples(self):
        """
        Returns:
            Sample states: state_list
            logphis of the sample state: logphi_list
            thetas of the sample state: theta_list
        """
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        #torch.set_num_interop_threads(1)

        #tic = time.time()
        pool = multiprocessing.Pool(self._threads)
        state_list = np.zeros([self._threads, self._n_sample] + self._single_state_shape)
        sym_states_list = np.zeros([self._threads, self._n_sample, self._model.sym_N] + self._single_state_shape)
        logphi_list = np.zeros([self._threads, self._n_sample])
        theta_list = np.zeros([self._threads, self._n_sample])
        
        results = []
        seed_list = np.random.choice(10000, size=self._threads, replace=False)
        for i in range(self._threads):
            results.append(pool.apply_async(self._mh_sampler,
                    (self._n_sample, self._state0[i], seed_list[i], )))
        pool.close()
        pool.join()

        cnt = 0
        for res in results:
            state_list[cnt], sym_states_list[cnt], logphi_list[cnt], theta_list[cnt] = res.get()
            cnt += 1
           
        states = state_list.reshape([-1]+self._single_state_shape)
        sym_states = sym_states_list.reshape([-1, self._model.sym_N]+self._single_state_shape)
        logphis = logphi_list.reshape(-1)
        thetas = theta_list.reshape(-1)
        states, sym_states, logphis, thetas, counts = _get_unique_states(states, sym_states, logphis, thetas)
        if self.acceptance:
            counts = np.ones(np.shape(counts))

        # print(np.shape(sym_states))
        # sym_ss = np.unique(sym_states.reshape([-1]+ self._single_state_shape), axis=0)
        # print(len(sym_ss))
        

        total_threads = os.cpu_count()//2
        self._eff_n_sample = (len(states)//total_threads)*total_threads
        mod_n_sample = len(states) - self._eff_n_sample
        state_threads = np.empty(total_threads, np.ndarray)
        len_per_thread = len(states)//total_threads
        
        pre_pos = 0
        for j in range(total_threads):
            if j < mod_n_sample:
                length = len_per_thread + 1
                state_threads[j] = states[pre_pos:pre_pos+length,:]
                pre_pos += length
            else:
                state_threads[j] = states[pre_pos:pre_pos+len_per_thread,:]
                pre_pos += len_per_thread
        
        ustates, upsis, ucoeffs, efflens \
            = self._generate_updates(state_threads, total_threads)
        
        # update the initial sampling state
        # index = np.argmax(logphi_list, axis=1)
        self._state0 = state_list[:,-1,:]
        self.single_state0 = state_list[-1,-1,:]
        # np.random.shuffle(self._state0)
        efflen = np.max(efflens)

        return (states, sym_states, logphis, thetas, counts, 
                ustates[:,:efflen,:],
                upsis[:,:efflen,:], 
                ucoeffs[:,:efflen], efflens)

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    seed = 2860
    torch.manual_seed(seed)
    np.random.seed(seed)

    import os
    pwd = os.getcwd()
    father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
    sys.path.append('father_path')
    from updators.state_swap_updator import updator
    from ops.HS_spin2d import Heisenberg2DTriangle, get_init_state, value2onehot
    from algos.core import mlp_cnn_sym
    from algos.core import c6rotation, transpose
    from utils_ppo import SampleBuffer
    import time 


    logphi_model = mlp_cnn_sym([4,4,2], 3, [8,6,4], stride0=[1], stride=[1], pbc=True, relu_type='cReLU', bias=True, momentum=[0,0], sym_funcs=[c6rotation], gcnn=True)

    _ham = Heisenberg2DTriangle(state_size=[4,4,2], pbc=True)
    MHsampler = MCsampler(state_size=[4,4,2], model=logphi_model,
                          get_init_state=get_init_state, n_sample=2500, threads=10,
                          updator=updator, operator=_ham)

    print(MHsampler._state0_v)
    tic = time.time()
    states, logphis, thetas, counts, update_states, update_psis, update_coeffs, efflens\
                        = MHsampler.get_new_samples()
    print(time.time() - tic)
    print(sum(counts))
    #print(counts)
    logps = np.unique(logphis)

    cpu = torch.device("cpu")
    buffer = SampleBuffer(cpu, [4,4,2])
    buffer.update(states, logphis, thetas, counts, update_states, 
                      update_psis, update_coeffs, efflens)

    print(len(states))
