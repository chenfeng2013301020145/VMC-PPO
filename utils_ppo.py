# encoding: utf-8

import torch
import time
import numpy as np
import multiprocessing
from multiprocessing import cpu_count
import logging
import os
from torch import nn, Tensor
from typing import Hashable, List, Tuple, Dict, Union, Callable
from torch.utils.data import Dataset

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
    unique, inverse_indices = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype,
                        device=inverse_indices.device)
    inverse, perm = inverse_indices.flip([0]), perm.flip([0])
    indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, indices, inverse_indices

def unique_row_view(data, unique_args=dict()):
    b = np.ascontiguousarray(data).view(
        np.dtype((np.void, data.dtype.itemsize * data.shape[1])))
    return np.unique(b, **unique_args)

class SampleBuffer(Dataset):
    def __init__(self, device, state_size, precision=torch.float32):
        """
        A buffer for storing samples from Markov chain sampler, keeping the most
        probable sample for the next policy update.
        """
        self._device = device
        if len(state_size) == 2:
            self.single_state_shape = [state_size[0]]
            self.N = state_size[0]
        else:
            self.single_state_shape = [state_size[0], state_size[1]]
            self.N = state_size[0]*state_size[1]
        self.Dp = state_size[-1]  # number of physical spins
        self.pow_list = np.arange(self.N-1, -1, -1)
        self._precision = precision

    def update(self, states, logphis, thetas, counts, 
               update_states, update_psis, update_coeffs, efflens, preload_size, batch_size):
        self.states = states
        # self.sym_states = sym_states
        self.logphis = logphis
        self.thetas = thetas
        self.counts = counts
        self.update_states = update_states
        self.update_psis = update_psis
        self.update_coeffs = update_coeffs
        self.efflens = efflens
        self.get_uniques()
        
        # self._preload_size = preload_size
        # self._batch_size = batch_size

        if preload_size >= self.uss_len:
            self._preload_size = self.uss_len
            self._batch_size = 0
            self._sd = 0
        else:
            self._preload_size = preload_size
            self._batch_size = batch_size
            n_sample = self.uss_len - self._preload_size
            self._sd = 1 if n_sample < self._batch_size else int(np.ceil(n_sample/self._batch_size))

        if self._sd > 1:
            self._preload_size += n_sample - (self._sd-1)*self._batch_size


        batch_label = np.arange(self._preload_size)
        self.preload_uss = self.unique_uss[batch_label,:]
        self.rest_unique_uss = self.unique_uss[self._preload_size:,:]

        # print(preload_size, self._preload_size, self.uss_len - self._preload_size)
        return
    
    def get_energy_ops(self):
        logphi = torch.from_numpy(self.logphis)
        theta = torch.from_numpy(self.thetas)
        logphi_ops = torch.from_numpy(self.update_psis[:,:,0])
        theta_ops = torch.from_numpy(self.update_psis[:,:,1])

        with torch.no_grad():
            delta_logphi_os = logphi_ops - logphi[...,None]
            delta_theta_os = theta_ops - theta[...,None]
            op_coeffs = torch.from_numpy(self.update_coeffs)
            self.ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)
            self.ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)
        return 
    
    def get_uniques(self):
        # calculate unique symmetry states
        # sym_ss_v0 = self.sym_states[:,0,0,:].reshape(-1, self.N).astype(np.int8)
        # self.symss_len = len(unique_row_view(sym_ss_v0))

        # sym_ss = self.sym_states.reshape([-1, self.Dp]+self.single_state_shape)
        # sym_ss_vs = sym_ss[:,0,:].reshape(-1, self.N).astype(np.int8)
        # _, sym_indices, self.sym_inverse_indices = unique_row_view(sym_ss_vs, 
        #                         unique_args=dict(return_index=True, return_inverse=True))
        # self.unique_symss = sym_ss[sym_indices]

        uss = self.update_states.reshape([-1, self.Dp]+self.single_state_shape)
        ussv = uss[:,0,:].reshape(-1, self.N)
        _, indices, self.uss_inverse_indices = unique_row_view(ussv, 
                                unique_args=dict(return_index=True, return_inverse=True))
        self.unique_uss = uss[indices]
        self.uss_len = len(self.unique_uss)
        return 

    def __len__(self):
        return self.uss_len - self._preload_size
        
    def cut_samples(self, preload_size=100, batch_size=100, batch_type='equal'):
        # n_sample = len(self.states) - preload_size
        n_sample = self.uss_len - preload_size
        devision_len = batch_size

        # if n_sample + preload_size <= batch_size:
        #     self.batch_label = np.arange(n_sample)[None,...]
        # elif batch_type == 'rand':
        #     self.batch_label = np.random.choice(n_sample, batch_size, replace=False)[None,...]
        # elif batch_type == 'equal':
        self.batch_label = []
        for i in range(self._sd):
            if i < self._sd - 1:
                self.batch_label.append(np.arange(i*devision_len+preload_size, (i+1)*devision_len+preload_size))
            elif i*devision_len+preload_size == n_sample+preload_size:
                self._sd -= 1
                break
            else:
                self.batch_label.append(np.arange(i*devision_len+preload_size, n_sample+preload_size))
        return

    def get_states(self):
        gpu_states = torch.from_numpy(self.states).to(self._precision).to(self._device)
        #gpu_sym_states = torch.from_numpy(self.unique_symss).float().to(self._device)
        #gpu_sym_ii = torch.from_numpy(self.sym_inverse_indices).to(self._device)

        gpu_counts = torch.from_numpy(self.counts).to(self._precision).to(self._device)
        gpu_logphi0 = torch.from_numpy(self.logphis).to(self._precision).to(self._device)
        gpu_theta0 = torch.from_numpy(self.thetas).to(self._precision).to(self._device)

        gpu_update_coeffs = torch.from_numpy(self.update_coeffs).to(self._precision).to(self._device)
        gpu_uss_inverse_indices = torch.from_numpy(self.uss_inverse_indices).to(self._device)
        
        pre_gpu_update_states_unique = torch.from_numpy(self.preload_uss).to(self._precision).to(self._device)

        return gpu_states, gpu_counts, gpu_logphi0, gpu_theta0, \
               gpu_update_coeffs, gpu_uss_inverse_indices, pre_gpu_update_states_unique

    def get(self, idx=1, batch_size=100, batch_type='all'):
           
        if batch_type == 'all':
            batch_label = self.batch_label[idx]
            selected_uss = self.unique_uss[batch_label,:]
            gpu_update_states_unique = torch.from_numpy(selected_uss).to(self._precision).to(self._device)
            return dict(update_states_unique=gpu_update_states_unique)
        else:   
            # random batch 
            batch_label = np.random.choice(len(self.states), batch_size, replace=False)[None,...] 

            batch_states = torch.from_numpy(self.states[batch_label,:]).to(self._precision).to(self._device)
            batch_counts = torch.from_numpy(self.counts[batch_label,:]).to(self._precision).to(self._device)
            batch_logphi0 = torch.from_numpy(self.logphis[batch_label,:]).to(self._precision).to(self._device)
            batch_theta0 = torch.from_numpy(self.thetas[batch_label,:]).to(self._precision).to(self._device)
            batch_ucs = torch.from_numpy(self.update_coeffs[batch_label,:]).to(self._precision).to(self._device)
            batch_uss = torch.from_numpy(self.update_states[batch_label,:]).to(self._precision).to(self._device)
            return dict(states=batch_states, counts=batch_counts, logphi=batch_logphi0,
                        theta=batch_theta0, ucs=batch_ucs, uss=batch_uss)

    def __getitem__(self, idx):
        #batch_label = self.batch_label[idx]
        selected_uss = self.rest_unique_uss[idx,:]
        gpu_update_states_unique = torch.from_numpy(selected_uss).to(self._precision).to(self._device)
        return gpu_update_states_unique
                
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

def find_states_and_ops(model, operator, states, single_state_shape ,cal_ops=False):
    with torch.no_grad():
        n_sample = states.shape[0]
        update_states = np.zeros([n_sample, operator._update_size] + single_state_shape)
        update_psis = np.zeros([n_sample, operator._update_size, 2])
        update_coeffs = np.zeros([n_sample, operator._update_size])
        efflens = np.zeros([n_sample], dtype=np.int64)
        
        if cal_ops:
            for i,state in enumerate(states):
                update_states[i], update_coeffs[i], efflen = operator.find_states(state)
                efflens[i] = efflen
                ustates = update_states[i,:efflen,:].reshape([-1]+single_state_shape)
                upsis = model(torch.from_numpy(ustates).float())
                update_psis[i,:efflen,:] = upsis.numpy().reshape([1, efflen, 2])
        else:
            for i,state in enumerate(states):
                update_states[i], update_coeffs[i], efflen = operator.find_states(state)
                efflens[i] = efflen
                ustates = update_states[i,:efflen,:].reshape([-1]+single_state_shape)
                ustates = model.pick_sym_config(torch.from_numpy(ustates)).numpy()
                update_states[i,:efflen,:] = ustates.reshape(update_states[i,:efflen,:].shape)
    return update_states, update_psis, update_coeffs, efflens

def _generate_updates(states, model, operator, single_state_shape, update_size, threads):
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
    efflens = np.zeros([n_sample], dtype=np.int64)

    pool = multiprocessing.Pool(threads)
    results = []
    cnt = 0
    
    for state in states:
        results.append(pool.apply_async(find_states_and_ops, 
                                       (model, operator, state, single_state_shape, )))
    pool.close()
    pool.join()

    for cnt, res in enumerate(results):
        ustates[cnt], ucoeffs[cnt], efflens[cnt] = res.get()

    return ustates, ucoeffs, efflens

# logger definitions
def get_logger(filename, verbosity=0, name=None):

    path = filename[0:filename.rfind("/")]
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isfile(filename):
        fd = open(filename, mode="w", encoding="utf-8")
        fd.close()

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(filename, "w")
    fh.setLevel(level_dict[verbosity+1])
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


