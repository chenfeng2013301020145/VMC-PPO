# encoding: utf-8

import torch
import numpy as np
import multiprocessing
from multiprocessing import cpu_count
import logging
import os
from torch import nn, Tensor
from typing import List, Tuple, Dict, Union, Callable

class SampleBuffer:
    def __init__(self, device, state_size):
        """
        A buffer for storing samples from Markov chain sampler, keeping the most
        probable sample for the next policy update.
        """
        self._device = device
        if len(state_size) == 2:
            self.single_state_shape = [state_size[0]]
        else:
            self.single_state_shape = [state_size[0], state_size[1]]
        self.Dp = state_size[-1]  # number of physical spins

    def update(self, states, logphis, thetas, counts, 
               update_states, update_psis, update_coeffs, efflens):
        self.states = states
        self.logphis = logphis
        self.thetas = thetas
        self.counts = counts
        self.update_states = update_states
        self.update_psis = update_psis
        self.update_coeffs = update_coeffs
        self.efflens = efflens
        self._call_time = 0
        self.get_energy_ops()
        return
    
    def get_energy_ops(self):
        logphi = torch.from_numpy(self.logphis)
        theta = torch.from_numpy(self.thetas)
        logphi_ops = torch.from_numpy(self.update_psis[:,:,0])
        theta_ops = torch.from_numpy(self.update_psis[:,:,1])

        delta_logphi_os = logphi_ops - logphi[...,None]*torch.ones_like(logphi_ops)
        delta_theta_os = theta_ops - theta[...,None]*torch.ones_like(theta_ops)
        op_coeffs = torch.from_numpy(self.update_coeffs)
        self.ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1).detach()
        self.ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1).detach()
        return 

    def get(self, batch_size=100, batch_type='rand', sample_division=1, get_eops=False):
        n_sample = len(self.states)
        devision_len = n_sample // sample_division 
        
        if n_sample <= batch_size:
            batch_label = range(len(self.states))
        elif batch_type == 'rand':
            batch_label = np.random.choice(n_sample, batch_size, replace=False)
        elif batch_type == 'equal':
            if self._call_time < sample_division - 1:
                batch_label = range(self._call_time*devision_len, (self._call_time+1)*devision_len)
                self._call_time += 1
            else:
                batch_label = range(self._call_time*devision_len, n_sample)
                self._call_time = 0
        
        if not get_eops:
            states = self.states[batch_label]
            logphis = self.logphis[batch_label]
            thetas = self.thetas[batch_label]
            counts = self.counts[batch_label]
            efflen = np.max(self.efflens[batch_label])
            
            update_states = self.update_states[batch_label, :efflen, :]
            update_coeffs = self.update_coeffs[batch_label, :efflen]

            gpu_states = torch.from_numpy(states).float().to(self._device)
            gpu_counts = torch.from_numpy(counts).float().to(self._device)
            gpu_update_coeffs = torch.from_numpy(update_coeffs).float().to(self._device)
            gpu_logphi0 = torch.from_numpy(logphis).float().to(self._device)
            gpu_theta0 = torch.from_numpy(thetas).float().to(self._device)
            
            # save GPU memory with unique array
            update_states = torch.from_numpy(update_states).float().to(self._device).reshape([-1, self.Dp]+self.single_state_shape)
            gpu_update_states_unique, gpu_inverse_indices = torch.unique(update_states,return_inverse=True,dim=0)
            
            return dict(state=gpu_states, count=gpu_counts, update_coeffs=gpu_update_coeffs, 
                        logphi0=gpu_logphi0, theta0=gpu_theta0, update_states_unique=gpu_update_states_unique, 
                        inverse_indices=gpu_inverse_indices)
        else:    
            states = self.states[batch_label]
            logphis = self.logphis[batch_label]
            thetas = self.thetas[batch_label]
            counts = self.counts[batch_label]

            gpu_states = torch.from_numpy(states).float().to(self._device)
            gpu_counts = torch.from_numpy(counts).float().to(self._device)
            gpu_logphi0 = torch.from_numpy(logphis).float().to(self._device)
            gpu_theta0 = torch.from_numpy(thetas).float().to(self._device)
            gpu_ops_real = self.ops_real[batch_label].float().to(self._device)
            gpu_ops_imag = self.ops_imag[batch_label].float().to(self._device)

            return dict(state=gpu_states, count=gpu_counts, logphi0=gpu_logphi0, 
                        theta0=gpu_theta0, ops_real=gpu_ops_real, ops_imag=gpu_ops_imag)
                    
def _get_unique_states(states, logphis, thetas, ustates, upsis, ucoeffs, efflens):
    """
    Returns the unique states, their coefficients and the counts.
    """
    states, indices, counts = np.unique(states, return_index=True, return_counts=True, axis=0)
    logphis = logphis[indices]
    thetas = thetas[indices]
    ustates = ustates[indices]
    upsis = upsis[indices]
    ucoeffs = ucoeffs[indices]
    efflens = efflens[indices]
    return states, logphis, thetas, counts, ustates, upsis, ucoeffs, efflens

def _generate_updates(states, operator, single_state_shape, update_size, threads):
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
    n_sample = states.shape[0]
    ustates = np.zeros([n_sample, update_size] + single_state_shape)
    ucoeffs = np.zeros([n_sample, update_size])

    pool = multiprocessing.Pool(threads)
    results = []
    cnt = 0
    
    for state in states:
        results.append(pool.apply_async(operator.find_states, (state,)))
    pool.close()
    pool.join()

    for cnt, res in enumerate(results):
        ustates[cnt], ucoeffs[cnt] = res.get()

    return ustates, ucoeffs

# logger definitions
def get_logger(filename, verbosity=1, name=None):

    path = filename[0:filename.rfind("/")]
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isfile(filename):
        fd = open(filename, mode="w", encoding="utf-8")
        fd.close()

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter) 
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def rot60(A, num=1, dims=[0,1], center=[0]):
    num = num%6
    input_shape = A.shape
    L = A.shape[dims[0]]
    W = A.shape[dims[1]]
    A = A.reshape(-1,L,W)
    
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(L))
    B = A.clone()   
    Xrot, Yrot = X, Y
    for _ in range(num):
        Xrot, Yrot = (Xrot - Yrot + center[0])%L, Xrot
    B[:, Xrot, Yrot] = A[:, X, Y]
    return B.reshape(input_shape)

def decimalToAny(n,x):
    # a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
    b=[]
    while True:
        s=n//x 
        y=n%x 
        b=b+[y]
        if s==0:
            break
        n=s
    b.reverse()

    return b

def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)
        
        
def np_rot60(A, num=1, axes=[0,1],center=[0]):
    input_shape = A.shape
    L = A.shape[axes[0]]
    W = A.shape[axes[1]]
    A = A.reshape(-1,L,W)
    
    X, Y = np.meshgrid(np.arange(W), np.arange(L))
    B = A.copy()   
    Xrot, Yrot = X, Y
    for _ in range(num):
        Xrot, Yrot = (Xrot - Yrot + center[0])%L, Xrot
    B[:, Xrot, Yrot] = A[:, X, Y]
    return B.reshape(input_shape)


