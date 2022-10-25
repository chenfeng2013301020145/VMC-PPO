# encoding: utf-8
# version 1.0 updates: parallel sampling, logger, saves
# version 2.0 updates: compatible with 2d system, replay buffer
# version 3.0 updates: double neural networks (real and imag)
# ppo-clip: early stop with Fubini-Study distance
import sys
import os
pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
sys.path.append('father_path')

import numpy as np
import torch
import torch.nn as nn
from sampler.mcmc_sampler_complex_ppo import MCsampler
from algos.core_gcnn2 import mlp_cnn_sym, get_paras_number, gradient
from utils_ppo import SampleBuffer, get_logger, _get_unique_states, extract_weights, load_weights
from torch.autograd.functional import jacobian
import scipy.io as sio
import time
from torch.utils.data import DataLoader

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
# ------------------------------------------------------------------------
class train_Ops:
    def __init__(self, **kwargs):
        self._ham = kwargs.get('hamiltonian')
        self._get_init_state = kwargs.get('get_init_state')
        self._updator = kwargs.get('updator')

# ------------------------------------------------------------------------
# main training function
def train(epochs=100, Ops_args=dict(), Ham_args=dict(), net_args=dict(), n_sample=80, 
          init_type='rand', n_optimize=10,
          learning_rate=1E-4, state_size=[10, 2], preload_size=2000, batch_size=2000, clip_ratio=0.1, 
          target_dfs=10, save_freq=10, threads=4, warm_up_sample_length=10, warmup_length=0,
          seed=0, input_fn=0, load_state0=True, output_fn='test', TolSite=1, max_beta=10.0,  
          min_beta=0.5, verbose=True, phase_constriction=True, precision=torch.float32):
    """
    main training process
    wavefunction: psi = phi*exp(1j*theta)
    output of the CNN network: logphi, theta

    Args:
        epochs (int): Number of epochs of interaction.
        
        Ops_args (dict): setup the names of operators.
        
        Ham_args (dict): setup the Hamiltonian.
        
        init_type (str): set the type of generated initial states

        n_sample (int): Number of sampling in each epoch.

        n_optimize (int): Number of update in each epoch.

        learning_rate: learning rate for Adam.

        state_size: size of a single state, (N, Dp) or (L, W, Dp).

        save_freq: frequency of saving.

        Dp: physical index.

        N or L, W: length of 1d lattice or length, with of 2d lattice
    """
    # seed += 1000*np.sum(np.arange(threads))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = os.path.join('../results', output_fn)
    save_dir = os.path.join(output_dir, 'save_model')
    logger = get_logger(os.path.join(output_dir, 'exp_log.txt'))

    dimensions = '1d' if len(state_size) == 2 else '2d'
    equal_probability=False

    if dimensions == '1d':
        TolSite = state_size[0] if TolSite == 1 else TolSite
        #single_state_shape = [state_size[0]]
    else:
        TolSite = state_size[0]*state_size[1] if TolSite == 1 else TolSite
        #single_state_shape = [state_size[0], state_size[1]]
    #Dp = state_size[-1]  # number of physical spins

    train_ops = train_Ops(**Ops_args)
    _ham = train_ops._ham(**Ham_args)
    get_init_state = train_ops._get_init_state
    updator = train_ops._updator
    buffer = SampleBuffer(gpu, state_size, precision)

    psi_model = mlp_cnn_sym(state_size=state_size, device=gpu, precision=precision, **net_args)
    # model for sampling
    mh_model = mlp_cnn_sym(state_size=state_size, device=cpu, precision=precision, **net_args)


    logger.info(psi_model)
    logger.info('Seed: {}'.format(seed))
    logger.info('Num_of_params-phi: {}'.format(get_paras_number(psi_model.model_phi, psi_model.filter_num)))
    logger.info('Num_of_params-theta: {}'.format(get_paras_number(psi_model.model_theta)))
    logger.info('Output_dir: {}'.format(output_dir))

    MHsampler = MCsampler(state_size=state_size, model=mh_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, threads=threads,
                          updator=updator, operator=_ham, precision=precision)

    first_warmup = False
    if input_fn != 0:
        load_model = torch.load(os.path.join('../results', input_fn))
        psi_model.load_state_dict(load_model)
        if load_state0:
            fn_name = os.path.split(os.path.split(input_fn)[0])
            mat_content = sio.loadmat(os.path.join('../results',fn_name[0],'state0.mat'))
            MHsampler._state0 = mat_content['state0']
        else:
            first_warmup = True
    
    # # define the loss function according to the energy functional in GPU
    def compute_psi_ops(op_states_unique, batch_size=0, batch_type='pre_load'):
        # data = buffer.get(idx)
        # op_states_unique = data['update_states_unique']            
        with torch.no_grad(): 
            psi_ops = psi_model(op_states_unique)
            logphi_ops = psi_ops[:, 0]
            theta_ops = psi_ops[:, 1]

        if batch_type=='batch':
            padding = torch.empty(batch_size-len(op_states_unique), device=gpu)
            return (torch.cat((logphi_ops.reshape(-1), padding), dim=0), 
                    torch.cat((theta_ops.reshape(-1), padding), dim=0))
        else:
            return (logphi_ops[...,None], theta_ops[...,None])

    def compute_psi_ops_single(op_states_unique, batch_size=0, batch_type='pre_load', mode='phi'):
        # data = buffer.get(idx)
        # op_states_unique = data['update_states_unique']            
        with torch.no_grad(): 
            if mode == 'phi':
                logphi_ops = psi_model.model_phi(op_states_unique)
                if equal_probability:
                    logphi_ops = torch.zeros_like(logphi_ops)
            else:
                logphi_ops = psi_model.model_theta(op_states_unique)
            # logphi_ops = psi_ops[:, 0]
            # theta_ops = psi_ops[:, 1]

        if batch_type=='batch':
            padding = torch.empty(batch_size-len(op_states_unique), device=gpu, dtype=precision)
            return torch.cat((logphi_ops.reshape(-1), padding), dim=0)
        else:
            return logphi_ops

    def compute_ops_ppo(batch_size):
        data = buffer.get(batch_size=batch_size,get_eops=True)
        ops_real, ops_imag = data['ops_real'], data['ops_imag']
        return ops_real[...,None], ops_imag[...,None]
        
    # mean energy from importance sampling in GPU
    def _energy_ops(sd, preload_size, batch_size, states, 
                counts, op_coeffs, op_ii, pre_op_states):
        
        with torch.no_grad():
            # psi = psi_model(sym_states)
            # logphis = psi[sym_ii, 0].reshape(len(sym_ii), -1)
            # thetas = psi[sym_ii, 1].reshape(len(sym_ii), -1)

            # psi = psi_model.model(sym_states, _only_theta=only_theta)/np.sqrt(psi_model.sym_N)
            # logMa = psi_model.get_logMa(sym_states).to(sym_states.dtype)
            # psi += logMa/psi_model.sym_N
            # psi = psi[sym_ii,:].reshape(-1, psi_model.sym_N, 2).sum(dim=1)

            # logphis = psi[:, 0].reshape(len(psi),-1)
            # thetas  = psi[:, 1].reshape(len(psi),-1)
            psi = psi_model(states)
            logphis = psi[:, 0].reshape(len(states), -1)
            thetas = psi[:, 1].reshape(len(states), -1)

            IntCount_uss = buffer.uss_len - preload_size
            n_sample, n_updates = op_coeffs.shape[0], op_coeffs.shape[1]
            pre_logphi_ops, pre_theta_ops = compute_psi_ops(pre_op_states)

            logphi_ops = torch.empty(sd, batch_size, device=gpu)
            theta_ops = torch.empty_like(logphi_ops)

            for i in range(sd):
                data = buffer.get(i)
                batch_op_states = data['update_states_unique']    
                logphi_ops[i], theta_ops[i] = compute_psi_ops(batch_op_states, batch_size, batch_type='batch')
            
            logphi_ops = logphi_ops.reshape(sd*batch_size, -1)[:IntCount_uss]
            theta_ops = theta_ops.reshape(sd*batch_size, -1)[:IntCount_uss]

            logphi_ops = torch.cat((pre_logphi_ops, logphi_ops), dim=0)[op_ii,:].reshape(n_sample, n_updates)
            theta_ops = torch.cat((pre_theta_ops, theta_ops), dim=0)[op_ii,:].reshape(n_sample, n_updates)

            delta_logphi_os = logphi_ops - logphis
            delta_theta_os = theta_ops - thetas
            ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)[...,None]
            ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)[...,None]

        return (ops_real*counts[...,None]).sum().to(cpu),\
               ((ops_real**2 + ops_imag**2)*counts[...,None]).sum().to(cpu)
               
    def compute_psi(states, count, logphi0, theta0):
        # psi = psi_model.model(sym_states, _only_theta=only_theta)/np.sqrt(psi_model.sym_N)
        # logMa = psi_model.get_logMa(sym_states).to(sym_states.dtype)
        # psi += logMa/psi_model.sym_N
        # psi = psi[sym_ii,:].reshape(-1, psi_model.sym_N, 2).sum(dim=1)
        
        # logphis = psi[:, 0].reshape(len(psi),-1)
        # thetas  = psi[:, 1].reshape(len(psi),-1)

        psi = psi_model(states)
        logphis = psi[:, 0].reshape(len(states), -1)
        thetas = psi[:, 1].reshape(len(states), -1)
        
        with torch.no_grad():
            count = count[...,None]
            delta_logphi = logphis - logphi0[..., None]
            delta_theta = thetas - theta0[...,None]
            deltalogphi = delta_logphi - delta_logphi.mean()
            
            ratio = torch.exp(deltalogphi*2)
            weights = count*ratio
            clip_ws = count*torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
            # norm = weights.sum()
            weights = (weights/weights.sum())
            clip_ws = (clip_ws/clip_ws.sum())
            
            #clip_ws = count*torch.clamp(weights, 1-clip_ratio, 1+clip_ratio)
            
            ratio_cut = torch.exp(deltalogphi*2)
            mincut = ratio_cut.min().item()
            maxcut = ratio_cut.max().item()
            clipped = ratio_cut.gt(1+clip_ratio) | ratio_cut.lt(1-clip_ratio)
            clipfrac = (torch.as_tensor(clipped, dtype=precision)).mean().item()
            
            # calculate the Fubini-Study distance
            phiold_phinew = (count*torch.exp(deltalogphi)*torch.exp(1j*delta_theta)).sum()
            phinew_phiold = phiold_phinew.conj()
            phiold_phiold = count.sum()
            phinew_phinew = (count*torch.exp(2*deltalogphi)).sum()
            
            dfs = torch.acos(torch.sqrt(phiold_phinew*phinew_phiold/phiold_phiold/phinew_phinew))
        
        return logphis, thetas, weights, clip_ws, dfs.abs(), clipfrac, mincut, maxcut
        
    def compute_ops_reim(logphis, thetas, op_coeffs, op_ii, pre_op_states,
                    sd, preload_size, batch_size): 
        with torch.no_grad():   
            IntCount_uss = buffer.uss_len - preload_size
            n_sample, n_updates = op_coeffs.shape[0], op_coeffs.shape[1]
            pre_logphi_ops, pre_theta_ops = compute_psi_ops(pre_op_states)

            if sd > 0:
                logphi_ops = torch.empty(sd, batch_size, device=gpu)
                theta_ops = torch.empty_like(logphi_ops)
                for i in range(sd):
                    data = buffer.get(i)
                    batch_op_states = data['update_states_unique']    
                    logphi_ops[i], theta_ops[i] = compute_psi_ops(batch_op_states, batch_size, batch_type='batch')
                
                logphi_ops = logphi_ops.reshape(sd*batch_size, -1)[:IntCount_uss]
                theta_ops = theta_ops.reshape(sd*batch_size, -1)[:IntCount_uss]

                logphi_ops = torch.cat((pre_logphi_ops, logphi_ops), dim=0)[op_ii,:].reshape(n_sample, n_updates)
                theta_ops = torch.cat((pre_theta_ops, theta_ops), dim=0)[op_ii,:].reshape(n_sample, n_updates)
            else:
                logphi_ops = pre_logphi_ops[op_ii,:].reshape(n_sample, n_updates)
                theta_ops = pre_theta_ops[op_ii,:].reshape(n_sample, n_updates)

            delta_logphi_os = logphi_ops - logphis
            delta_theta_os = theta_ops - thetas
            ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)[...,None]
            ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)[...,None]

        return ops_real, ops_imag

    def compute_ops_delta(logphis, op_coeffs, op_ii, pre_op_states, sd, preload_size, batch_size, mode='phi'): 
        with torch.no_grad():   
            IntCount_uss = buffer.uss_len - preload_size
            n_sample, n_updates = op_coeffs.shape[0], op_coeffs.shape[1]
            pre_logphi_ops = compute_psi_ops_single(pre_op_states, mode=mode)

            if sd > 0:
                logphi_ops = torch.empty(sd, batch_size, device=gpu)
                #theta_ops = torch.empty_like(logphi_ops)
                for i in range(sd):
                    data = buffer.get(i)
                    batch_op_states = data['update_states_unique']    
                    logphi_ops[i] = compute_psi_ops_single(batch_op_states, batch_size, batch_type='batch', mode=mode)
                
                logphi_ops = logphi_ops.reshape(sd*batch_size)[:IntCount_uss]

                logphi_ops = torch.cat((pre_logphi_ops, logphi_ops), dim=0)[op_ii].reshape(n_sample, n_updates)
            else:
                logphi_ops = pre_logphi_ops[op_ii].reshape(n_sample, n_updates)

            #print(logphi_ops[0], logphi_ops[-1])
            delta_logphi_os = logphi_ops - logphis
            #delta_theta_os = theta_ops - thetas
            #ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)[...,None]
            #ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)[...,None]

        return delta_logphi_os
    def compute_loss(count, logphis, thetas, logphis0, thetas0,
                    weights, clip_ws, delta_logphi_os, delta_theta_os, op_coeffs, beta, gamma, output_mode='phi'):   

        with torch.no_grad(): 
            ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)[...,None]
            ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)[...,None]
            # calculate the mean energy
            count = count[...,None]
            me_real = (weights*ops_real).sum()
            cme_real = (clip_ws*ops_real).sum()
        
        delta_theta = thetas - thetas0[...,None]
        delta_theta = delta_theta - (delta_theta.mean()).detach()
        delta_logphi = logphis - logphis0[...,None]
        delta_logphi = delta_logphi - (delta_logphi.mean()).detach()

        # phiold_phinew = (count*torch.exp(1j*delta_theta)).sum()
        # phinew_phiold = phiold_phinew.conj()
        # phiold_phiold = count.sum()
        # phinew_phinew = count.sum()
        
        # dfs = (phiold_phinew*phinew_phiold/phiold_phiold/phinew_phinew).real

        if output_mode == 'phi':
            E_re = ops_real*logphis - me_real*logphis
            cE_re = ops_real*logphis - cme_real*logphis
            #loss = [weights*E_re, clip_ws*E_re, weights*cE_re, clip_ws*cE_re]
            loss = [weights*E_re, clip_ws*cE_re]
            loss_re, _ = torch.max(torch.cat(loss, dim=1), dim=1) 
            loss_re = (loss_re + gamma*(count/count.sum())*delta_logphi**2).sum()
            #loss_re = loss_re + 0.001*torch.pow(2*delta_logphi, 4)
        else:
            if phase_constriction:
                loss_re = weights*ops_imag*thetas - beta*weights*torch.cos(delta_theta)
                loss_re = loss_re.sum()
                #loss_re = (weights*ops_imag*thetas).sum() - beta*dfs
            else:
                loss_re = (weights*ops_imag*thetas).sum()
            #loss_re = loss_re.sum()
            #loss_re = (weights*ops_imag*thetas).sum() - beta*dfs
            
        #return loss_re, 1 - dfs, me_real, cme_real
        return loss_re, (weights*(1-torch.cos(delta_theta))).sum().item(), me_real, cme_real

    # # setting optimizer
    optimizer_theta = torch.optim.Adam(psi_model.model_theta.parameters(), lr=learning_rate)
    optimizer_phi = torch.optim.Adam(psi_model.model_phi.parameters(), lr=learning_rate)
    scheduler_theta = torch.optim.lr_scheduler.MultiStepLR(optimizer_theta, [warm_up_sample_length+100], gamma=1)
    scheduler_phi = torch.optim.lr_scheduler.MultiStepLR(optimizer_phi, [warm_up_sample_length+100], gamma=1)

    def update_theta(n_optimize, preload_size, batch_size, sd):
        # off-policy update
        states, counts, logphi0, theta0, op_coeffs, op_ii, pre_op_states = buffer.get_states()
        delta_logphi_ops = compute_ops_delta(logphi0[...,None], op_coeffs, op_ii, pre_op_states,
                                            sd, preload_size, batch_size, mode='phi')
        # data = buffer.get(batch_size=batch_size, batch_type='cutoff', get_eops=not(cal_ops))
        cme_old = 0
        beta = 1.0
        # n_optimize = n_optimize // 2 if stop else n_optimize
        for i in range(n_optimize):
            optimizer_theta.zero_grad()
            logphis, thetas, weights, clip_ws, dfs, clipfrac, mincut, maxcut \
                = compute_psi(states, counts, logphi0, theta0)

            delta_theta_ops = compute_ops_delta(thetas, op_coeffs, op_ii, pre_op_states,
                                            sd, preload_size, batch_size, mode='theta')
            loss_e, angtol, me, cme_real = compute_loss(counts, logphis, thetas, logphi0, theta0,
                                            weights, clip_ws, delta_logphi_ops, delta_theta_ops, op_coeffs, 
                                            beta, gamma=0, output_mode='theta')

            # adaptive penalty
            if angtol > 0.2:
                beta *= 1.5
            elif angtol < 0.05:
                beta /= 1.5
            
            beta = np.clip(beta, min_beta, max_beta)

            if i == 0:
                er = me

            if abs(me - cme_old) < 1e-6:
                logger.debug(
                'early stop at step={} as reaching converged energy in updating thetas'.format(i))
                break
            else:
                cme_old = me

            if verbose and i%(n_optimize//5) == 0:
                print('me: {:.4f}, dfs: {:.4f}, logmincut: {:.4f}, logmaxcut: {:.4f}, angtol: {:.4f}, beta: {:.2f}'.format(me.item()/TolSite, dfs.item(), np.log(mincut), np.log(maxcut), angtol, beta))

            loss_e.backward()
            optimizer_theta.step()   

        # if cal_energy or early_stop:
        #     AvgE, AvgE2 = _energy_ops(sd, preload_size, batch_size, 
        #                 states, counts, op_coeffs, op_ii, pre_op_states)
        return dfs, angtol, me, cme_real, er, clipfrac, i

    def update_logphi(n_optimize, preload_size, batch_size, sd, target):
        # off-policy update
        states, counts, logphi0, theta0, op_coeffs, op_ii, pre_op_states \
            = buffer.get_states()
        delta_theta_ops = compute_ops_delta(theta0[...,None], op_coeffs, op_ii, pre_op_states,
                            sd, preload_size, batch_size, mode='theta')
        # delta_logphi_ops = compute_ops_delta(logphi0[...,None], op_coeffs, op_ii, pre_op_states,
        #                     sd, preload_size, batch_size, mode='phi')
        # ops_real, ops_imag = compute_ops_reim(logphi0[..., None], theta0[..., None], 
        #                                     op_coeffs, op_ii, pre_op_states,
        #                                     sd, preload_size, batch_size)
        # data = buffer.get(batch_size=batch_size, batch_type='cutoff', get_eops=not(cal_ops))
        
        cme_old = 0
        gamma = 1.0
        for i in range(n_optimize):
            optimizer_phi.zero_grad()
            logphis, thetas, weights, clip_ws, dfs, clipfrac, mincut, maxcut \
                = compute_psi(states, counts, logphi0, theta0)

            # loss_e, me, cme_real =compute_loss(counts, logphis, thetas, op_coeffs, op_ii, pre_op_states,
            #                     weights, clip_ws, sd, preload_size, batch_size, get_eops)
            # update E local
            # if clipfrac > 0.1:
            # adaptive penalty
            if angtol > 0.2:
                gamma *= 1.5
            elif angtol < 0.05:
                gamma /= 1.5
            
            gamma = np.clip(gamma, min_beta, max_beta)
        
            delta_logphi_ops = compute_ops_delta(logphis, op_coeffs, op_ii, pre_op_states,
                                sd, preload_size, batch_size, mode='phi')
            loss_e, angtol, me, cme_real = compute_loss(counts, logphis, thetas, logphi0, theta0,
                                weights, clip_ws, delta_logphi_ops, delta_theta_ops, op_coeffs, 
                                beta=0, gamma=gamma, output_mode='phi')
            if i == 0:
                er = me
                
            #if dfs > 1.5*target:
            if np.log(mincut) < -1.5*target or np.log(maxcut) > target:
                logger.debug(
                'early stop at step={} as reaching maximal FS distance in updating logphis'.format(i))
                # early_stop = True
                break

            if abs(me - cme_old) < 1e-6:
                logger.debug(
                'early stop at step={} as reaching converged energy in updating logphis'.format(i))
                break
            else:
                cme_old = me
            
            if verbose and i%(n_optimize//5) == 0:
                print('me: {:.4f}, dfs: {:.4f}, logmincut: {:.4f}, logmaxcut: {:.4f}, angtol: {:.4f}, gamma: {:.2f}'.format(me.item()/TolSite, dfs.item(), np.log(mincut), np.log(maxcut), angtol, gamma))

            loss_e.backward()
            optimizer_phi.step()
        
        return dfs, 0, me, cme_real, er, clipfrac, i

    # ----------------------------------------------------------------
    tic = time.time()
    logger.info('mean_spin: {}'.format(MHsampler._state0_v))
    logger.info('Start training:')
    DFS = 0
    num_early_stop = 0
    # StdE = 1
    # TDFS = target_dfs
    if first_warmup:
        MHsampler._model.load_state_dict(psi_model.state_dict())
        MHsampler.first_warmup(1000)
    warmup_n_sample = n_sample
    #MHsampler.warmup_length = warmup_length
    warmup_n_sample = n_sample // 4
    # MHsampler._n_sample = warmup_n_sample
    # train_theta = True
    #psi_model._only_phi = True

    for epoch in range(epochs):
        sample_tic = time.time()
        
        if epoch >= warm_up_sample_length - 1:
            warmup_n_sample = n_sample

        if epoch < warm_up_sample_length:
            MHsampler.warmup_length = 0*warmup_length
            MHsampler.acceptance = True
            psi_model._only_theta = True
            MHsampler._model._only_theta = True
            equal_probability=True
        else:
            MHsampler.warmup_length = warmup_length
            MHsampler.acceptance = False
            psi_model._only_theta = False
            MHsampler._model._only_theta = False
            equal_probability=False

        # sync parameters and update the mh_model
        MHsampler._n_sample = warmup_n_sample
        MHsampler._model.model_phi.load_state_dict(psi_model.model_phi.state_dict())
        states, counts, update_states, update_psis, update_coeffs, efflens\
                        = MHsampler.get_new_samples()

        state_gpu = torch.from_numpy(states).to(precision).to(gpu)
        psi_gpu = psi_model(state_gpu)
        logphis = psi_gpu[:,0].cpu().detach().numpy()
        thetas = psi_gpu[:,1].cpu().detach().numpy()

        buffer.update(states, logphis, thetas, counts, update_states,
                      update_psis, update_coeffs, efflens, preload_size, batch_size)
    
        if MHsampler.cal_ops:
            buffer.get_energy_ops()

        IntCount = len(states)
        # SymIntCount = buffer.symss_len

        preload = buffer._preload_size
        batch = buffer._batch_size

        buffer.cut_samples(preload_size=preload, batch_size=batch, batch_type='equal')
        
        sample_toc = time.time()
        # ------------------------------------------GPU------------------------------------------
        # sd = 1 if (buffer.uss_len - preload) < batch else int(np.ceil((buffer.uss_len - preload)/batch))
        op_tic = time.time()

        if epoch < warm_up_sample_length:
            DFS, AngTol, ME, CME, ER, clipfrac, idx = update_theta(n_optimize, preload, batch, buffer._sd)

        else:
            DFS, AngTol, ME, CME, ER, clipfrac, idx = update_theta(n_optimize, preload, batch, buffer._sd)

            if AngTol < 0.1 or not phase_constriction:
                # psi_gpu = psi_model(state_gpu) 
                # logphis = psi_gpu[:,0].cpu().detach().numpy()
                thetas = psi_model.model_theta(state_gpu).cpu().detach().numpy()
                buffer.thetas = thetas

                DFS, _, ME, CME, _, clipfrac, idx = update_logphi(n_optimize, preload, batch, buffer._sd, target_dfs)
            else:
                num_early_stop += 1

        op_toc = time.time()

        # AvgE, AvgE2 = _energy_ops(sd, batch_size)
        # ---------------------------------------------------------------------------------------
        # average over all samples
        # AvgE = AvgE/n_real_sample
        # AvgE2 = AvgE2/n_real_sample
        # StdE = np.sqrt(AvgE2 - AvgE**2)/TolSite

        # # adjust the learning rate
        scheduler_theta.step()
        scheduler_phi.step()
        # # scheduler.step()
        # # lr = scheduler.get_last_lr()[-1]
        # # print training informaition
        # # for name, param in psi_model.named_parameters():
        # #     if 'act' in name:
        # #         alpha = param.cpu()
        alpha = []
        for name, param in psi_model.model_theta.named_parameters():
            if 'act' in name:
                #alpha = np.around(param.cpu().detach().item(),5)
                alpha.append(np.around(param.cpu().detach().item(),5))
        if verbose:
            print(alpha)

        # logger.info('Epoch: {}, AvgE: {:.6f}, ME: {:.5f}, CME: {:.5f}, StdE: {:.5f}, DFS: {:.5f}, ClipFrac: {:.3f}, StopIter: {}, IntCount: {}, A: {}, num_batch: {}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
        #     format(epoch, AvgE/TolSite, ME/TolSite, CME/TolSite, StdE, DFS, clipfrac, idx, IntCount, alpha, sd, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))
        logger.info('Epoch: {}, AvgE: {:.6f}, ME: {:.5f}, CME: {:.5f}, DFS: {:.5f}, ClipFrac: {:.3f}, StopIter: {}, IntCount: {}, A: {:.5f}, num_batch {}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
        format(epoch, ER/TolSite, ME/TolSite, CME/TolSite, DFS, clipfrac, idx, IntCount, AngTol, buffer._sd, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))

        # save the trained NN parameters
        torch.cuda.empty_cache()
        if epoch % save_freq == 0 or epoch == epochs - 1:

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(psi_model.state_dict(), os.path.join(save_dir, 'model_'+str(epoch)+'.pkl'))
            sio.savemat(os.path.join(output_dir, 'state0.mat'), {'state0': MHsampler._state0})

        # if epoch < warm_up_sample_length:
        #     # first 10 epochs are used to warm up due to the limitations of memory
        #     warmup_n_sample += n_sample//warm_up_sample_length

        # if IntCount < (n_sample*threads)//10:
        #     warmup_n_sample = 2*n_sample

        # if epoch >= warm_up_sample_length + num_early_stop + 200:
        #     n_optimize = 10
        #     learning_rate = 1e-4

    logger.info('Finish training.')

    return psi_model.to(cpu), MHsampler._state0
