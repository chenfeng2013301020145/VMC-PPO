# encoding: utf-8
# compatible with 2d sysytem

import torch
import numpy as np
import multiprocessing
import os

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def _generate_updates(state, operator):
    """
    Generates updated states and coefficients for an Operator.

    Args:
        states: The states with shape (shape of state).
        operator: The operator used for updating the states.

    Returns:
        The updated states and their coefficients. The shape of the updated
        states is (num of updates, shape of state), where num of
        updates is the largest number of updated states among all given states.
        If a state has fewer updated states, its updates are padded with the
        original state.

    """
    ustates, ucoeffs = operator.find_states(state)
    return ustates, ucoeffs

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
        self._warmup = False
        self.basis_warmup_sample = 1000
        self.accept = False

        Dp = self._state_size[-1]
        if self._dimension == 1:
            N = self._state_size[0]
            self._single_state_shape = [Dp, N]
        else:
            length = self._state_size[0]
            width = self._state_size[1]
            self._single_state_shape = [Dp, length, width]
        
        self._updator = self._update_operator(self._state_size)
        self.single_state0, self._state0_v = self._get_init_state(self._state_size, kind=self._init_type, n_size=1)
        # self._state0 = self.warmup_sample(n_sample = 20000)
    
    def first_warmup(self):
        self._state0 = self.warmup_sample(n_sample = 20000)
        return 
    
    def get_single_sample(self, state, logphi_i, mask, rand, update_states, update_coeffs):
        with torch.no_grad():
            state_f = self._updator._get_update(state, mask)
            psi_f = self._model(torch.from_numpy(state_f[None,...]).float()).numpy()

            logphi_f = psi_f[:,0]
            delta_logphi = logphi_f - logphi_i

            if delta_logphi>0 or rand<=np.exp(delta_logphi*2.0) or self.accept:
                update_states, update_coeffs = _generate_updates(state_f, self._op)
                return state_f, logphi_f, update_states, update_coeffs
            else:
                return state, logphi_i, update_states, update_coeffs
    
    def warmup_sample(self, n_sample):
        masks = self._updator.generate_mask(n_sample)
        rands = np.random.rand(n_sample)
        state0 = self.single_state0.squeeze()
        output_state0 = []
        with torch.no_grad():
            psi = self._model(torch.from_numpy(state0[None,...]).float()).numpy()
            logphi_i = psi[:,0]
            state = state0

            cnt = 0
            while cnt < n_sample:
                state_f = self._updator._get_update(state, masks[cnt])
                psi_f = self._model(torch.from_numpy(state_f[None,...]).float()).numpy()

                logphi_f = psi_f[:,0]
                delta_logphi = logphi_f - logphi_i
                if delta_logphi>0 or rands[cnt]<=np.exp(delta_logphi*2.0):
                    state = state_f
                    logphi_i = logphi_f

                if cnt > n_sample - self._threads - 1:
                    output_state0.append(state)
                cnt += 1
            #print(np.array(output_state0).shape)
        #return np.repeat(state[None,...], self._threads, axis=0)
        return np.array(output_state0)

    def _mh_sampler(self, n_sample_per_thread: int, state0, seed_number):
        """
        Importance sampling with Metropolis-Hasting algorithm
        
        Returns: 
            state_sample_per_thread: (n_sample_per_thread, Dp, N)
            logphi_sample_per_thread: (n_sample_per_thread)
            theta_sample_per_thread: (n_sample_per_thread)
        """
        # empty tensor storing the sample data
        state_sample_per_thread = np.zeros([n_sample_per_thread] + self._single_state_shape)
        logphi_sample_per_thread = np.zeros(n_sample_per_thread)
        us_sample_per_thread = np.zeros([n_sample_per_thread, self._update_size] + self._single_state_shape)
        uc_sample_per_thread = np.zeros([n_sample_per_thread, self._update_size])
        
        with torch.no_grad():
            np.random.seed(seed_number)
            masks = self._updator.generate_mask(n_sample_per_thread)
            rands = np.random.rand(n_sample_per_thread)
            
            state = np.squeeze(state0)
            psi = self._model(torch.from_numpy(state0[None,...]).float()).numpy()
 
            logphi = psi[:,0]
            i = 0
            update_states, update_coeffs = _generate_updates(state, self._op)
            
            while i < n_sample_per_thread:
                state, logphi, update_states, update_coeffs = self.get_single_sample(state, logphi, masks[i], rands[i], update_states, update_coeffs)
                state_sample_per_thread[i] = state
                logphi_sample_per_thread[i] = logphi
                us_sample_per_thread[i] = update_states
                uc_sample_per_thread[i] = update_coeffs
                i += 1

        return (state_sample_per_thread, logphi_sample_per_thread,
                us_sample_per_thread, uc_sample_per_thread)

    def get_new_samples(self):
        """
        Returns:
            Sample states: state_list
            logphis of the sample state: logphi_list
            thetas of the sample state: theta_list
        """
        if self._warmup:
            self._state0 = self.warmup_sample(n_sample = 5000)
        else:
            self._state0 = self.warmup_sample(n_sample = self.basis_warmup_sample)
        #self._state0 = self.single_state0

        pool = multiprocessing.Pool(self._threads)
        n_sample_per_thread = self._n_sample // self._threads
        self._n_sample = int(n_sample_per_thread*self._threads)

        results = []
        seed_list = np.random.choice(10000, size=self._threads)
        for i in range(self._threads):
            results.append(pool.apply_async(self._mh_sampler, 
                    (n_sample_per_thread, self._state0[i], seed_list[i], )))
        pool.close()
        pool.join()


        state_list = np.zeros([self._threads, n_sample_per_thread] + self._single_state_shape)
        logphi_list = np.zeros([self._threads, n_sample_per_thread])
        us_list = np.zeros([self._threads, n_sample_per_thread, self._update_size] + self._single_state_shape)
        uc_list = np.zeros([self._threads, n_sample_per_thread, self._update_size])

        cnt = 0
        for res in results: 
            state_list[cnt], logphi_list[cnt], us_list[cnt], uc_list[cnt] = res.get()
            cnt += 1

        # update the initial sampling state
        self.single_state0 = state_list[-1,-1,:]
        # np.random.shuffle(self._state0)
        
        return (state_list.reshape([self._n_sample] + self._single_state_shape), 
                logphi_list.reshape(self._n_sample),
                us_list.reshape([self._n_sample, self._update_size] + self._single_state_shape),
                uc_list.reshape([self._n_sample, self._update_size]))

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    from updators import state_flip_updator
    #from core import mlp_cnn
    #from tfim_spin1d import get_init_state
    #from state_flip_updator import updator

    # state_size = [10,2]

    # logphi_model = mlp_cnn(state_size, output_size=2, K=2, F=2)
    # state0, _ = get_init_state(state_size,kind='rand')

    # phi = logphi_model(torch.from_numpy(state0).float())
    # logphi_i = phi[:,0].detach().numpy()
    # theta_i = phi[:,1]

    # Op = updator(state_size)
    # masks = Op.generate_mask(100)

    # sampler = MCsampler(state_size=state_size, model=logphi_model, state0=state0, updator=updator)

    # # state, logphi, theta = sampler.get_single_sample(np.squeeze(state0), logphi_i, theta_i, masks[10], np.random.rand())

    # state, log, theta = sampler._mh_sampler(10, state0, 1234)
    # print(state.shape)