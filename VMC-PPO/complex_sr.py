# encoding: utf-8
# version 1.0 updates: parallel sampling, logger, saves
# version 2.0 updates: compatible with 2d system, replay buffer
# version 3.0: double CNN (real and imag), imaginary time propagation (Stochastic Reconfiguration, ISGO)
# evaluate Oks, Skk_matrix with torch.autograd.functional.jacobian
# update with Adam
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sampler.mcmc_sampler_complex_ppo import MCsampler
from algos.core import mlp_cnn_sym, get_paras_number
from utils_ppo import SampleBuffer, get_logger, _get_unique_states, extract_weights, load_weights
from torch.autograd.functional import jacobian
import scipy.io as sio
import copy
import time
import os

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
# ------------------------------------------------------------------------
class train_Ops:
    def __init__(self, **kwargs):
        self._ham = kwargs.get('hamiltonian')
        self._get_init_state = kwargs.get('get_init_state')
        self._updator = kwargs.get('updator')

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    '''
    Solving S_kk \delta w = -\gamma*F_k with conjugate gradients
    input:
        A: S_kk
        b: -\gamma*F_k
    return:
        x: \delta w
    '''
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

# ------------------------------------------------------------------------
# main training function
def train(epochs=100, Ops_args=dict(), Ham_args=dict(), n_sample=80, init_type='rand', n_optimize=10,
          learning_rate=1E-4, state_size=[10, 2], preload_size=2000, batch_size=2000, clip_ratio=0.01, 
          target_dfs=1, save_freq=10, net_args=dict(), threads=4, warmup_length=500, warm_up_sample_length=10,
          seed=0, input_fn=0, load_state0=True, output_fn='test', TolSite=1):
    """
    main training process
    wavefunction: psi = phi*exp(1j*theta)
    output of the CNN network: logphi, theta

    Args:
        epochs (int): Number of epochs of interaction.

        n_sample (int): Number of sampling in each epoch.

        n_optimize (int): Number of update in each epoch.

        lr: learning rate for Adam.

        state_size: size of a single state, [n_sites, Dp].

        save_freq: frequency of saving.

        Dp: physical index.

        N or L, W: length of 1d lattice or length and with of 2d lattice
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = os.path.join('../results', output_fn)
    save_dir = os.path.join(output_dir, 'save_model')
    logger = get_logger(os.path.join(output_dir, 'exp_log.txt'))

    dimensions = '1d' if len(state_size) == 2 else '2d'
    if dimensions == '1d':
        TolSite = state_size[0] if TolSite == 1 else TolSite
        single_state_shape = [state_size[0]]
    else:
        TolSite = state_size[0]*state_size[1] if TolSite == 1 else TolSite
        single_state_shape = [state_size[0], state_size[1]]
    Dp = state_size[-1]  # number of physical spins

    train_ops = train_Ops(**Ops_args)
    _ham = train_ops._ham(**Ham_args)
    get_init_state = train_ops._get_init_state
    updator = train_ops._updator
    buffer = SampleBuffer(gpu, state_size) 

    psi_model = mlp_cnn_sym(state_size=state_size, device=gpu, **net_args)
    # model for sampling
    mh_model = mlp_cnn_sym(state_size=state_size, device=cpu, **net_args)

    logger.info(psi_model)
    logger.info('Seed: {}'.format(seed))
    logger.info('Num_of_params: {}'.format(get_paras_number(psi_model)))
    logger.info('Output_dir: {}'.format(output_dir))

    MHsampler = MCsampler(state_size=state_size, model=mh_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, threads=threads,
                          updator=updator, operator=_ham)

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

    # define the loss function according to the energy functional in GPU
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

    def compute_ops_ppo(batch_size):
        data = buffer.get(batch_size=batch_size,get_eops=True)
        ops_real, ops_imag = data['ops_real'], data['ops_imag']
        return ops_real[...,None], ops_imag[...,None]

    # mean energy from importance sampling in GPU
    def _energy_ops(sd, preload_size, batch_size, sym_states, sym_ii, 
                counts, op_coeffs, op_ii, pre_op_states, only_theta):
        
        with torch.no_grad():
            # psi = psi_model(sym_states)
            # logphis = psi[sym_ii, 0].reshape(len(sym_ii), -1)
            # thetas = psi[sym_ii, 1].reshape(len(sym_ii), -1)

            psi = psi_model.model(sym_states)/np.sqrt(psi_model.sym_N)
            logMa = psi_model.get_logMa(sym_states).to(sym_states.dtype)
            psi += logMa/psi_model.sym_N
            psi = psi[sym_ii,:].reshape(-1, psi_model.sym_N, 2).sum(dim=1)

            logphis = psi[:, 0].reshape(len(psi),-1)
            if only_theta:
                logphis = torch.ones_like(logphis)
            thetas  = psi[:, 1].reshape(len(psi),-1)

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

    def compute_psi(sym_states, sym_ii, count, logphi0, theta0, only_theta):
        psi = psi_model.model(sym_states)/np.sqrt(psi_model.sym_N)
        logMa = psi_model.get_logMa(sym_states).to(sym_states.dtype)
        psi += logMa/psi_model.sym_N
        psi = psi[sym_ii,:].reshape(-1, psi_model.sym_N, 2).sum(dim=1)
        
        logphis = psi[:, 0].reshape(len(psi),-1)
        if only_theta:
            logphis = torch.ones_like(logphis)

        thetas  = psi[:, 1].reshape(len(psi),-1)
      
        count = count[...,None]
        delta_logphi = logphis - logphi0[..., None]
        delta_theta = thetas - theta0[...,None]
        deltalogphi = delta_logphi - delta_logphi.mean()
        
        with torch.no_grad():
            ratio = torch.exp(deltalogphi*2)
            weights = count*ratio
            weights = (weights/weights.sum())
            
        # calculate the Fubini-Study distance
        phiold_phinew = (count*torch.exp(deltalogphi)*torch.exp(1j*delta_theta)).sum()
        phinew_phiold = phiold_phinew.conj()
        phiold_phiold = count.sum()
        phinew_phinew = (count*torch.exp(2*deltalogphi)).sum()
        
        dfs = torch.acos(torch.sqrt(phiold_phinew*phinew_phiold/phiold_phiold/phinew_phinew))
        
        return logphis, thetas, weights, dfs
        
    def compute_loss(count, logphis, thetas, op_coeffs, op_ii, pre_op_states,
                    weights, sd, preload_size, batch_size, get_eops): 
        with torch.no_grad():   
            if not get_eops:
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

            else:
                ops_real, ops_imag = compute_ops_ppo(batch_size)
            
            # calculate the mean energy
            count = count[...,None]
            me_real = (weights*ops_real).sum()
        
        E_re = ops_real*logphis - me_real*logphis + ops_imag*thetas
        loss_re = (weights*E_re).sum()
        
        return loss_re, me_real
    
    # optimizer = torch.optim.Adam(psi_model.parameters(), lr=learning_rate)

    def update(n_optimize, preload_size, batch_size, sd, target, DFS_fac, get_eops, only_theta):
        # off-policy update
        sym_states, sym_ii, counts, logphi0, theta0, op_coeffs, op_ii, pre_op_states \
            = buffer.get_states(preload_size)
        # data = buffer.get(batch_size=batch_size, batch_type='cutoff', get_eops=not(cal_ops))
        cme_old = 0
        for i in range(n_optimize):
            # random batch for large systems
            # data = buffer.get(batch_size=batch_size, batch_type='rand', get_eops=not(cal_ops))
            # optimizer.zero_grad()
            logphis, thetas, weights, dfs \
                = compute_psi(sym_states, sym_ii, counts, logphi0, theta0, only_theta)

            # if dfs > DFS_fac*target:
            #     logger.debug(
            #     'early stop at step={} as reaching maximal FS distance'.format(i))
            #     break
            loss, me =compute_loss(counts, logphis, thetas, op_coeffs, op_ii, pre_op_states,
                                weights, sd, preload_size, batch_size, get_eops)

            print(loss)

            max_me = me
            if abs(max_me - cme_old) < 1e-8:
                logger.debug(
                'early stop at step={} as reaching converged energy'.format(i))
                break
            else:
                cme_old = max_me
            
            # TRPO step:
            grads = torch.autograd.grad(dfs, psi_model.model.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

            def Fvp(v):
                grads = torch.autograd.grad(dfs, psi_model.parameters(), create_graph=True)
                flat_grad_dfs = torch.cat([grad.view(-1) for grad in grads])

                dfs_v = (flat_grad_dfs * Variable(v)).sum()
                grads = torch.autograd.grad(dfs_v, psi_model.parameters())
                flat_grad_grad_dfs = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

                return flat_grad_grad_dfs
            
            stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
            shs = 0.5*(stepdir*Fvp(stepdir)).sum(0, keepdim=True)
            lm = torch.sqrt(shs / (target*DFS_fac))

            print(lm.size())
            # loss_e.backward()
            # optimizer.step()

        AvgE, AvgE2 = _energy_ops(sd, preload_size, batch_size, 
                sym_states, sym_ii, counts, op_coeffs, op_ii, pre_op_states, only_theta)
        
        return dfs.cpu().item(), me, AvgE, AvgE2, i

    # ----------------------------------------------------------------
    tic = time.time()
    logger.info('mean_spin: {}'.format(MHsampler._state0_v))
    logger.info('Start training:')
    DFS = 0
    StdE = 1
    # TDFS = target_dfs
    if first_warmup:
        MHsampler._model.load_state_dict(psi_model.state_dict())
        MHsampler.first_warmup(20000)
    warmup_n_sample = n_sample//2
    #train_theta = True

    for epoch in range(epochs):
        sample_tic = time.time()
        
        # get new samples from MCMC smapler
        if DFS > target_dfs:
            MHsampler.warmup_length = 2*warmup_length
        elif StdE < 0.01 or DFS < 0.1*target_dfs:
            MHsampler.warmup_length = warmup_length // 2
        elif DFS < 0.01*target_dfs:
            MHsampler.warmup_length = 0
        else:
            MHsampler.warmup_length = warmup_length
            
        if epoch < 0:
            # calculate updata_psis in sampling process
            MHsampler.cal_ops = True
            batch = n_sample*threads
        else:
            MHsampler.cal_ops = False
            batch = batch_size

        if epoch < warm_up_sample_length:
            DFS_fac = 100
        else:
            DFS_fac = 4

        set_op_steps = n_optimize
        MHsampler.acceptance = False
        psi_model._only_theta = False
        MHsampler._model._only_theta = False
        
        # sync parameters and update the mh_model
        MHsampler._n_sample = warmup_n_sample
        MHsampler._model.load_state_dict(psi_model.state_dict())
        states, sym_states, logphis, thetas, counts, update_states, update_psis, update_coeffs, efflens\
                        = MHsampler.get_new_samples()

        # sym_ss, _ = MHsampler._model.symmetry(torch.from_numpy(sym_states))
        # sym_ss = torch.unique(sym_ss, dim=0)
        # print(len(sym_ss))

        n_real_sample = sum(counts)
        buffer.update(states, sym_states, logphis, thetas, counts, update_states,
                      update_psis, update_coeffs, efflens)

        # print(np.mean(logphis), np.mean(thetas))
    
        if MHsampler.cal_ops:
            buffer.get_energy_ops()

        IntCount = len(states)
        SymIntCount = buffer.symss_len

        if preload_size > buffer.uss_len:
            preload = buffer.uss_len - 1
            batch = 1
        else:
            preload = preload_size
            batch = batch_size

        buffer.cut_samples(preload_size=preload, batch_size=batch, batch_type='equal')

        sample_toc = time.time()

        # ------------------------------------------GPU------------------------------------------
        sd = 1 if (buffer.uss_len - preload) < batch else int(np.ceil((buffer.uss_len - preload)/batch))
        op_tic = time.time()

        # if train_theta and epoch < warm_up_sample_length + 20:
        #     DFS, ME, CME, clipfrac, AvgE, AvgE2 = update(preload, batch, sd, IntCount - preload, 
        #                         target_dfs, get_eops=MHsampler.cal_ops, clear_grad=zero_grad_logphi)
        #     if DFS < 0.1*target_dfs:
        #         train_theta = False
        # else:
        #train_theta = False

        DFS, ME, AvgE, AvgE2, idx = update(set_op_steps, preload, batch, sd, 
                target_dfs, DFS_fac, get_eops=MHsampler.cal_ops, only_theta=psi_model._only_theta)

        op_toc = time.time()

        # AvgE, AvgE2 = _energy_ops(sd, batch_size)
        # ---------------------------------------------------------------------------------------
        # average over all samples
        AvgE = AvgE.numpy()/n_real_sample
        AvgE2 = AvgE2.numpy()/n_real_sample
        StdE = np.sqrt(AvgE2 - AvgE**2)/TolSite

        # adjust the learning rate
        # scheduler.step()
        # lr = scheduler.get_last_lr()[-1]
        #EGE = AvgE/TolSite - 0 if np.isnan(StdE) else AvgE/TolSite - StdE
        #TDFS = target_fubini_study_distance(EGE, AvgE, AvgE2, lr*n_optimize)
        # print training informaition
        for name, param in psi_model.named_parameters():
            if 'act' in name:
                alpha = param.cpu().item()

        logger.info('Epoch: {}, AvgE: {:.5f}, ME: {:.5f}, StdE: {:.5f}, DFS: {:.5f}, StopIter: {}, IntCount: {}, SymIntCount: {}, A: {:.3f}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
            format(epoch, AvgE/TolSite, ME/TolSite, StdE, DFS, idx, IntCount, SymIntCount, alpha, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))
        
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

        if epoch > warm_up_sample_length:
        # #     # first 10 epochs are used to warm up due to the limitations of memory
            warmup_n_sample = n_sample
            # print(warmup_n_sample)

    logger.info('Finish training.')

    return psi_model.to(cpu), MHsampler._state0, AvgE/TolSite
