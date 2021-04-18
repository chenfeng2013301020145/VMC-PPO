# encoding: utf-8

import torch
import numpy as np
import multiprocessing
import logging
import os
from torch import nn, Tensor
from typing import List, Tuple, Dict, Union, Callable

class SampleBuffer:
    def __init__(self, device):
        """
        A buffer for storing samples from Markov chain sampler, keeping the most
        probable sample for the next policy update.
        """
        self._device = device

    def update(self,states, logphis, thetas, counts, update_states, update_coeffs):
        self.states = states
        self.logphis = logphis
        self.thetas = thetas
        self.counts = counts
        self.update_states = update_states
        self.update_coeffs = update_coeffs
        self._call_time = 0
        return

    def get(self, batch_size=100, batch_type='rand', sample_division=1):
        n_sample = len(self.states)
        devision_len = n_sample // sample_division 
        
        if n_sample <= batch_size:
            gpu_states = torch.from_numpy(self.states).float().to(self._device)
            gpu_counts = torch.from_numpy(self.counts).float().to(self._device)
            gpu_update_states = torch.from_numpy(self.update_states).float().to(self._device)
            gpu_update_coeffs = torch.from_numpy(self.update_coeffs).float().to(self._device)
            gpu_logphi0 = torch.from_numpy(self.logphis).float().to(self._device)
            gpu_theta0 = torch.from_numpy(self.thetas).float().to(self._device)
        elif batch_type == 'rand':
            batch_label = np.random.choice(n_sample, batch_size, replace=False)
            states = self.states[batch_label]
            logphis = self.logphis[batch_label]
            thetas = self.thetas[batch_label]
            counts = self.counts[batch_label]
            update_states = self.update_states[batch_label]
            update_coeffs = self.update_coeffs[batch_label]

            gpu_states = torch.from_numpy(states).float().to(self._device)
            gpu_counts = torch.from_numpy(counts).float().to(self._device)
            gpu_update_states = torch.from_numpy(update_states).float().to(self._device)
            gpu_update_coeffs = torch.from_numpy(update_coeffs).float().to(self._device)
            gpu_logphi0 = torch.from_numpy(logphis).float().to(self._device)
            gpu_theta0 = torch.from_numpy(thetas).float().to(self._device)
        elif batch_type == 'equal':
            if self._call_time < sample_division - 1:
                batch_label = range(self._call_time*devision_len, (self._call_time+1)*devision_len)
                self._call_time += 1
            else:
                batch_label = range(self._call_time*devision_len, n_sample)
                self._call_time = 0
            
            states = self.states[batch_label]
            logphis = self.logphis[batch_label]
            thetas = self.thetas[batch_label]
            counts = self.counts[batch_label]
            update_states = self.update_states[batch_label]
            update_coeffs = self.update_coeffs[batch_label]

            gpu_states = torch.from_numpy(states).float().to(self._device)
            gpu_counts = torch.from_numpy(counts).float().to(self._device)
            gpu_update_states = torch.from_numpy(update_states).float().to(self._device)
            gpu_update_coeffs = torch.from_numpy(update_coeffs).float().to(self._device)
            gpu_logphi0 = torch.from_numpy(logphis).float().to(self._device)
            gpu_theta0 = torch.from_numpy(thetas).float().to(self._device)

        return dict(state=gpu_states, count=gpu_counts, update_states=gpu_update_states,
                    update_coeffs=gpu_update_coeffs, logphi0=gpu_logphi0, theta0=gpu_theta0)
                    
def _get_unique_states(states, logphis, ustates, ucoeffs):
    """
    Returns the unique states, their coefficients and the counts.
    """
    states, indices, counts = np.unique(states, return_index=True, return_counts=True, axis=0)
    logphis = logphis[indices]
    ustates = ustates[indices]
    ucoeffs = ucoeffs[indices]
    return states, logphis, counts, ustates, ucoeffs

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

# conjugate gradient from openai/baseline
def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = (r*r).sum()

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, x.norm()))
        z = f_Ax(p)
        v = rdotr / (p*z).sum()
        x += v*p
        r -= v*z
        newrdotr = (r*r).sum()
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, x.norm()))  # pylint: disable=W0631
    return x