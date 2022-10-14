# encoding: utf-8
# update version 2.0: compatible with 2d system
import sys
sys.path.append('..')

import numpy as np
import torch
from sampler.mcmc_sampler_complex_ppo import MCsampler
from utils_ppo import SampleBuffer, _generate_updates

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
#------------------------------------------------------------------------
class cal_op():
    def __init__(self, **kwargs):
        self._state_size = kwargs.get('state_size')
        if len(self._state_size) - 1 == 1:
            # (Dp, N)
            self._single_state_shape = [self._state_size[-1], self._state_size[0]]
        else:
            # (Dp, L, W)
            self._single_state_shape = [self._state_size[-1], self._state_size[0], self._state_size[1]]
        self._Dp = self._state_size[-1]

        self.psi_model = kwargs.get('psi_model')
        self._n_sample = kwargs.get('n_sample')
        self._updator = kwargs.get('updator')
        self._get_init_state = kwargs.get('get_init_state')
        self._init_type = kwargs.get('init_type', 'rand')
        self._threads = kwargs.get('threads', 4)
        self._state0 = kwargs.get('state0')
        self._preload_size = kwargs.get('preload_size', 1000)
        self._batch_size = kwargs.get('batch_size', 1000)
        self._run = 0
        self._buffer = SampleBuffer(gpu,self._state_size)
        
    def get_sample(self, op):
        self._sampler = MCsampler(state_size=self._state_size, model=self.psi_model, 
                get_init_state=self._get_init_state, init_type=self._init_type, n_sample=self._n_sample, 
                threads=self._threads, updator=self._updator, operator=op)
        self._sampler._state0 = self._state0
        self._sampler.cal_ops = False
        self._sampler._n_sample = 1000
        states, sym_states, logphis, thetas, counts, uss, upsis, ucs, efflens\
                        = self._sampler.get_new_samples()
        self._states = states
        self.total_n_sample = np.sum(counts)
        self._buffer.update(states, sym_states, logphis, thetas, counts, uss,
                      upsis, ucs, efflens)
        self._buffer.cut_samples(preload_size=self._preload_size, batch_size=self._batch_size, batch_type='equal')
        return 

    #################################
    def regen_updates(self, op):
        self._buffer.update_states, self._buffer.update_coeffs, self._buffer.efflens \
            = _generate_updates(self._states, self.psi_model, op, self._single_state_shape, op._update_size, self._threads)
        return
    #################################
    
    def compute_ops(self, sample_division, batch_size, logphis, thetas):
        data = self._buffer.get(batch_size=batch_size, batch_type='equal', sample_division=sample_division)
        op_coeffs, op_states_unique, inverse_indices \
            = data['update_coeffs'], data['update_states_unique'], data['uss_inverse_indices']
        batch_label = data['batch_label']
        logphi = logphis[batch_label]
        theta = thetas[batch_label]
            
        with torch.no_grad(): 
            n_sample = op_coeffs.shape[0]
            n_updates = op_coeffs.shape[1]

            psi_ops = self.psi_model(op_states_unique)
            logphi_ops = psi_ops[inverse_indices, 0].reshape(n_sample, n_updates)
            theta_ops = psi_ops[inverse_indices, 1].reshape(n_sample, n_updates)

            delta_logphi_os = logphi_ops - logphi*torch.ones(logphi_ops.shape, device=gpu)
            delta_theta_os = theta_ops - theta*torch.ones(theta_ops.shape, device=gpu)
            ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)
            ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)

        padding = torch.zeros(batch_size-n_sample, device=gpu)
        return (torch.cat((ops_real.reshape(-1), padding), dim=0), 
                torch.cat((ops_imag.reshape(-1), padding), dim=0))

    def _ops(self):
        #data = self._buff.get(batch_type='equal', sample_division=sample_division)
        states, counts, _, _ = self._buffer.get_states()
        IntCount = len(states)
        sd = 1 if IntCount < self._batch_size else int(np.ceil(IntCount/self._batch_size))

        psi = self.psi_model(states)
        logphi = psi[:, 0].reshape(IntCount, -1)
        theta = psi[:, 1].reshape(IntCount, -1)
        
        ops_real = torch.zeros(sd, self._batch_size, device=gpu)
        ops_imag = torch.zeros_like(ops_real)
        
        with torch.no_grad():
            for i in range(sd):
                ops_real[i], ops_imag[i] = self.compute_ops(sd, self._batch_size, logphi, theta)
                
            ops_real = ops_real.reshape(sd*self._batch_size, -1)[:IntCount]
            ops_imag = ops_imag.reshape(sd*self._batch_size, -1)[:IntCount]

        return ((ops_real*counts[...,None]).sum().to(cpu), ((ops_real**2)*counts[...,None]).sum().to(cpu), 
                (ops_imag*counts[...,None]).sum().to(cpu), ((ops_imag**2)*counts[...,None]).sum().to(cpu))

    def get_value(self, operator, op_args=dict()):
        if self._run == 0:
            op = operator(**op_args)
            self.get_sample(op)
            self._run += 1
        else:
            op = operator(**op_args)
            self.regen_updates(op)
        
        self.psi_model = self.psi_model.to(gpu)
        avgOp_real, avgOp2_real, avgOp_imag, _ = self._ops()
        self.psi_model = self.psi_model.to(cpu)

        # average over all samples
        AvgOp_real = avgOp_real.numpy()/self.total_n_sample
        AvgOp_imag = avgOp_imag.numpy()/self.total_n_sample

        # std only for real part
        AvgOp2 = avgOp2_real.numpy()/self.total_n_sample
        StdOp = np.sqrt(AvgOp2 - AvgOp_real**2)
        return AvgOp_real + 1j*AvgOp_imag, StdOp, len(self._states)

def onehot2value(state, Dp, dimensions): 
    state = np.squeeze(state,0) if state.shape[0] == 1 else state
    if dimensions == 1:
        state_v = np.arange(0,Dp).reshape(Dp,1)*state
    else:
        state_v = np.arange(0,Dp).reshape(Dp,1,1)*state
    return np.sum(state_v,0).astype(dtype=np.int8)

class Sz():
    def __init__(self, state_size):
        """
        S_z = sum_i{sigma^z_i}
        """
        self._update_size = 1
        self._dimensions = len(state_size) - 1
        self._Dp = state_size[-1]

    def find_states(self, state: np.ndarray):
        if self._dimensions == 1:
            state_v = onehot2value(state, self._Dp, 1) - 1/2
            N = state.shape[-1]
            return (state.reshape(self._update_size,self._Dp,N), 
                    np.sum(state_v).reshape(self._update_size))
        else:
            state_v = onehot2value(state, self._Dp, 2) - 1/2
            L = state.shape[-2]
            W = state.shape[-1]
            return (state.reshape(self._update_size,self._Dp,L,W), 
                    np.sum(state_v).reshape(self._update_size), 1)

class Sx():
    def __init__(self, state_size):
        """
        S_x = sum_i{sigma^x_i}
        """
        self._dimensions = len(state_size) - 1
        self._Dp = state_size[-1]
        if self._dimensions == 1:
            self._update_size = state_size[0]
        else:
            self._update_size = state_size[0]*state_size[1]

    def find_states(self, state: np.ndarray):
        if self._dimensions == 1:
            N = state.shape[-1]
            states = np.zeros([self._update_size, self._Dp, N])
            coeffs = np.zeros(self._update_size)

            for i in range(N):
                temp = state.copy()
                temp[0,i], temp[1,i] = state[1,i], state[0,i]
                states[i,:,:] = temp
                coeffs[i] = 1/2
        
        else:
            W = state.shape[-1]
            L = state.shape[-2]
            states = np.zeros([L*W, self._Dp, L, W])
            coeffs = np.zeros(L*W)

            # off-diagnal
            cnt = 0
            for y in range(L):
                for x in range(W):
                    temp = state.copy()
                    temp[0,x,y], temp[1,x,y] = state[1,x,y], state[0,x,y]
                    states[cnt] = temp
                    coeffs[cnt] = 1/2
                    cnt += 1

        return states, coeffs, cnt

class SzSz():
    def __init__(self, state_size, pbc):
        """
        S_zS_z = sum_i{sigma^z_i + sigma^z_i+1}
        """
        self._pbc = pbc
        self._update_size = 1
        self._dimensions = len(state_size) - 1
        self._Dp = state_size[-1]

    def find_states(self, state:np.ndarray):
        if self._dimensions == 1:
            state_v = onehot2value(state, self._Dp,1) - 1/2
            state_l = np.concatenate((state_v, state_v[0].reshape(1,)), axis=0)
            state_r = np.concatenate((state_v[-1].reshape(1,), state_v), axis=0)
            if self._pbc:   
                diag = - np.sum(state_l[1:]*state_r[1:])
            else:
                diag = - np.sum(state_l[1:-1]*state_r[1:-1])

            return (state.reshape(self._update_size,self._Dp,state.shape[-1]), 
                    diag.reshape(self._update_size))
        else:
            W = state.shape[-1]
            L = state.shape[-2]
            state_v = onehot2value(state, self._Dp, 2) - 1/2
            state_l = np.concatenate((state_v, state_v[:,0].reshape(L,1)), axis=1)
            state_r = np.concatenate((state_v[:,-1].reshape(L,1), state_v), axis=1)
            state_u = np.concatenate((state_v, state_v[0].reshape(1,W)), axis=0)
            state_d = np.concatenate((state_v[-1].reshape(1,W), state_v), axis=0)
            if self._pbc:   
                diag = - np.sum(state_l[:,1:]*state_r[:,1:]) - np.sum(state_u[1:]*state_d[1:])
            else:
                diag = - np.sum(state_l[:,1:-1]*state_r[:,1:-1]) - np.sum(state_u[1:-1]*state_d[1:-1])
            return (state.reshape(self._update_size,self._Dp,L,W), 
                    diag.reshape(self._update_size), 1)
            
# class neel_order():
#     def __init__(self, state_size):
#         """
#         M = \sum_{i}((-1)**i*S_i)
#         """
#         self._dimensions = len(state_size) - 1
#         self._Dp = state_size[-1]
#         if self._dimensions == 1:
#             self._update_size = state_size[0]
#         else:
#             self._update_size = state_size[0]*state_size[1]
        
        


if __name__ == "__main__":
    import sys
    print(sys.path[0])