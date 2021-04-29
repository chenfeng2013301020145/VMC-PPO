# encoding: utf-8
import sys
sys.path.append('..')

import torch
import numpy as np
import multiprocessing
import os
from utils import decimalToAny

os.environ["OMP_NUM_THREADS"] = "1"
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

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

class FBsampler():
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

        self._Dp = self._state_size[-1]
        if self._dimension == 1:
            N = self._state_size[0]
            self._single_state_shape = [self._Dp, N]
            self._totalsite = N
        else:
            length = self._state_size[0]
            width = self._state_size[1]
            self._single_state_shape = [self._Dp, length, width]
            self._totalsite = length*width
        
        self._updator = self._update_operator(self._state_size)
        self.single_state0, self._state0_v = self._get_init_state(self._state_size, 
                                                                  kind=self._init_type, n_size=1)
        self._basis_index, self._basis_state = self.create_basis()
        self._n_sample = len(self._basis_index)
        self._us_list, self._uc_list = self.get_updates()
    

    def value2onehot(self, state):
        if self._dimension == 1:
            N = state.shape[0]
            state_onehot = np.zeros([self._Dp, N])
            state_onehot[state.astype(dtype=np.int8), range(N)] = 1
        else:
            L = state.shape[0]
            W = state.shape[1]
            X, Y = np.meshgrid(range(W), range(L))
            state_onehot = np.zeros([self._Dp, L, W])
            state_onehot[state.astype(dtype=np.int8), Y, X] = 1
        return state_onehot

    def create_basis(self):
        basis_index = []
        basis_state = []
        for i in range(self._Dp**self._totalsite):
            num_list = decimalToAny(i, self._Dp)
            state_v = np.array([0]*(self._totalsite - len(num_list)) + num_list)
            if self._state0_v is not None:
                if (state_v - (self._Dp-1)/2).sum() == self._state0_v:
                    basis_index.append(i)
                    basis_state.append(self.value2onehot(state_v.reshape(self._single_state_shape[1:])))
            else:
                state_v = np.array([0]*(self._totalsite - len(num_list)) + num_list)
                basis_index.append(i)
                basis_state.append(self.value2onehot(state_v.reshape(self._single_state_shape[1:])))
        
        return basis_index, np.array(basis_state).astype(dtype=np.float32)
    
    def get_updates(self):
        pool = multiprocessing.Pool(self._threads)
        
        results = []
        for i in range(len(self._basis_index)):
            results.append(pool.apply_async(_generate_updates, (self._basis_state[i], self._op)))
        pool.close()
        pool.join()
        
        us_list = np.zeros([self._n_sample, self._update_size] + self._single_state_shape)
        uc_list = np.zeros([self._n_sample, self._update_size])
        
        cnt = 0
        for res in results:
            us_list[cnt], uc_list[cnt] = res.get()
            cnt += 1
        return us_list, uc_list
    
    def get_new_samples(self):
        psi = self._model(torch.from_numpy(self._basis_state))
        logphi = psi[:,1].detach().numpy()
        return (self._basis_state, logphi, self._us_list, self._uc_list)
    
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    
    from updators import state_swap_updator
    from algos.core import mlp_cnn
    from ops.HS_spin2d import Heisenberg2DSquare, get_init_state
    from updators.state_swap_updator import updator
    
    state_size = [3,3,2]
    ham = Heisenberg2DSquare(state_size=state_size)
    logphi_model, _ = mlp_cnn(state_size, 2, [3,2],complex_nn=True,
                           output_size=2, relu_type='softplus2', bias=True)
    
    Test = FBsampler(state_size=state_size, model=logphi_model, operator=ham,
                     updator=updator, get_init_state=get_init_state)
    print(Test._basis_state.shape)
    bs, log, us, uc = Test.get_new_samples()
    print(bs.shape, log.shape, us.shape, uc.shape)