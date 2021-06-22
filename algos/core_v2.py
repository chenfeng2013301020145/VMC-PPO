# encoding: utf-8
import sys
from torch.autograd.grad_mode import F
sys.path.append('..')

import torch.nn as nn
import torch
import numpy as np
from utils_ppo import rot60
# from utils import extract_weights,load_weights

def periodic_padding(x, kernel_size, dimensions):
    if dimensions == '1d':
        # shape of real x: (batch_size, Dp, N) and complex x: (batch_size, 2, Dp, N)
        if len(x.shape) == 3:
            pad = (0, kernel_size-1)
        else:
            pad = (0, kernel_size-1, 0, 0)
    else:
        # shape of real x: (batch_size, Dp, Length, Width) 
        #      and complex x: (batch_size, 2, Dp, Length, Width)
        if len(x.shape) == 4:
            pad = (0, kernel_size[1]-1, 0, kernel_size[0] - 1)
        else:
            pad = (0, kernel_size[1]-1, 0, kernel_size[0] - 1, 0, 0)
    return nn.functional.pad(x, pad, mode='circular')
    
def complex_periodic_padding(x, kernel_size, stride, dimensions):
    if dimensions == '1d':
        N = x.shape[-1]
        pN = (stride - 1)*N + (kernel_size - 1) - stride + 1
        # shape of complex x: (batch_size, 2, Dp, N)
        x_old = x.clone()
        for _ in range(pN // N):
            x = torch.cat((x, x_old), -1)
        return torch.cat((x, x[:,:,:,0:(pN%N)]), -1)
    else:
        # shape of complex x: (batch_size, 2, Dp, Length, Width) 
        L = x.shape[-2]
        W = x.shape[-1]
        pL = (stride[0] - 1)*L + (kernel_size[0] - 1) - stride[0] + 1
        x_old = x.clone()
        for _ in range(pL // L):
            x = torch.cat((x, x_old), -2)
        x = torch.cat((x, x[:,:,:,0:(pL%L),:]), -2)
        
        pW = (stride[1] - 1)*W + (kernel_size[1] - 1) - stride[1] + 1
        x_old = x.clone()
        for _ in range(pW // W):
            x = torch.cat((x, x_old), -1)
        x = torch.cat((x, x[:,:,:,:,0:(pW%W)]), -1)
        return x
    
def get_paras_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def gradient(y, x, grad_outputs=None):
    """compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def complex_init(model):
    cnt = -1
    for name, p in model.named_parameters():
        if int(name.split(".")[0]) - cnt == 1:
            rho = torch.from_numpy(np.random.rayleigh(1, size=p.data.shape)).float()
            theta = np.random.uniform(-np.pi, +np.pi, size=p.data.shape)
            w = np.sqrt(np.prod(p.data.shape[:-2])*(p.data.shape[-1] + p.data.shape[-2]))
            cnt += 1
        if name.split(".")[-1] == 'weight':
            if name.split(".")[3] == 'conv_re' or name.split(".")[2] == 'linear_re':
                p.data = torch.from_numpy(np.cos(theta)).float()*rho/w
            elif name.split(".")[3] == 'conv_im' or name.split(".")[2] == 'linear_im':
                p.data = torch.from_numpy(np.sin(theta)).float()*rho/w
    return

# COMPLEX NEURAL NETWORK
# ----------------------------------------------------------------
class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, 
                    padding=0, dilation=1, groups=1, bias=True, dimensions='1d'):
        super(ComplexConv,self).__init__()

        ## Model components
        if dimensions == '1d':
            self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real, imaginary),dim=1)
        return output

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear,self).__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,in_features]
        real = self.linear_re(x[:,0]) - self.linear_im(x[:,1])
        imaginary = self.linear_re(x[:,1]) + self.linear_im(x[:,0])
        output = torch.stack((real, imaginary),dim=1)
        return output

class ComplexLnCosh(nn.Module):
    def __init__(self):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexLnCosh, self).__init__()

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        z = torch.log(2*torch.cosh(z))
        return torch.stack((z.real, z.imag), dim=1)
    
class ComplexTanh(nn.Module):
    def __init__(self):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexTanh, self).__init__()

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        z = torch.tanh(z)
        return torch.stack((z.real, z.imag), dim=1)

class ComplexReLU(nn.Module):
    def __init__(self, relu_type='zReLU'):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexReLU, self).__init__()
        self.type = relu_type
        self._threshold = -0.1

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        if self.type == 'zReLU':
            mask = ((0 < z.angle()) * (z.angle() < np.pi/2)).float()
            return torch.stack((real*mask, imag*mask), dim=1)
        elif self.type == 'sReLU':
            mask = ((-np.pi/2 < z.angle()) * (z.angle() < np.pi/2)).float()
            return torch.stack((real*mask, imag*mask), dim=1)
        elif self.type == 'modReLU':
            modulus = torch.clamp(z.abs(), min=1e-5)
            z *= torch.relu(1. - self._threshold / modulus)
            return torch.stack((z.real, z.imag), dim=1)
        elif self.type == 'softplus':
            # z = torch.log(1. + torch.exp(z))
            real = torch.log(1 + torch.exp(real))
            imag = torch.log(1 + torch.exp(imag))
            return torch.stack((real, imag), dim=1)
        elif self.type == 'softplus2':
            # z = torch.log(1./2. + torch.exp(z)/2.)
            real = torch.log(1./2. + torch.exp(real)/2.)
            imag = torch.log(1./2. + torch.exp(imag)/2.)
            return torch.stack((real, imag), dim=1)
        elif self.type == 'selu':
            return torch.stack((nn.functional.selu(real), nn.functional.selu(imag)), dim=1)
        else:
            return torch.stack((torch.relu(real), torch.relu(imag)), dim=1)

# COMPLEX CNN
#--------------------------------------------------------------------
def translation_phase(z, k, dimensions):
    xdevice = z.device
    n_sample = z.shape[0]
    F = z.shape[1]
    real = z.real
    imag = z.imag
    if dimensions == '1d':
        N = z.shape[-1]
        vec = torch.exp(1j*2*np.pi*k[0]*torch.arange(N, device=xdevice)/N)
        vec_real = vec.real.repeat(n_sample, F, 1)
        vec_imag = vec.imag.repeat(n_sample, F, 1)
        real_part = real*vec_real - imag*vec_imag
        imag_part = real*vec_imag + imag*vec_real
    else:
        L = z.shape[-2]
        W = z.shape[-1]
        mat = (torch.exp(1j*2*np.pi*k[1]*torch.arange(W, device=xdevice)/W)
            *torch.exp(1j*2*np.pi*k[0]*torch.arange(L, device=xdevice)/L)[...,None])
        mat_real = mat.real.repeat(n_sample, F, 1, 1)
        mat_imag = mat.imag.repeat(n_sample, F, 1, 1)
        real_part = real*mat_real - imag*mat_imag
        imag_part = real*mat_imag + imag*mat_real
    return real_part + 1j*imag_part

class CNN_complex_layer(nn.Module):
    def __init__(self,K,F_in,F_out,stride,layer_name='mid',relu_type='sReLU',
                 pbc=True, bias=True, dimensions='1d'):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN_complex_layer,self).__init__()
        self.K = [K,K] if type(K) is int and dimensions=='2d' else K
        self._pbc = pbc
        self.layer_name = layer_name
        self.dimensions = dimensions
        self.stride = [stride, stride] if type(stride) is int and dimensions=='2d' else stride
        
        complex_relu = ComplexReLU(relu_type)
        complex_lncosh = ComplexLnCosh()
        if pbc:
            complex_conv = ComplexConv(F_in,F_out,self.K,stride,0, dimensions=dimensions, bias=bias)
        else:
            complex_conv = ComplexConv(F_in,F_out,self.K,stride,1, dimensions=dimensions, bias=bias)
        self.conv = nn.Sequential(*[complex_conv, complex_relu])

    def forward(self, x):
        if self.layer_name == '1st':
            x = torch.stack((x, torch.zeros_like(x)), dim=1)
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions=self.dimensions)
        x = self.conv(x)
        return x

class OutPut_complex_layer(nn.Module):
    def __init__(self,K,F,pbc=True,dimensions='1d',momentum=[0,0]):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut_complex_layer,self).__init__()
        self.K = [K,K] if type(K) is int and dimensions=='2d' else K
        self.F = F
        self._pbc=pbc
        self.dimensions = dimensions
        self.momentum = momentum
        self.linear = ComplexLinear(F, 1, bias=False)
    
    def forward(self,x):
        # shape of complex x: (batch_size, 2, F, N) or (batch_size, 2, F, L, W)
        # variance scaling 
        norm = np.sqrt(np.prod(x.shape[2:]))
        # x = x.sum(2)/norm
        z = torch.exp((x[:,0] + 1j*x[:,1])/norm)
        if self._pbc:
            z = translation_phase(z, k=self.momentum, dimensions=self.dimensions)
        z = z.sum(dim=[2]) if self.dimensions=='1d' else z.sum(dim=[2,3])
        x = self.linear(torch.stack((z.real, z.imag), dim=1)).squeeze(-1)
        z = x[:,0] + 1j*x[:,1]
        z = torch.log(z)
        return torch.stack((z.real, z.imag), dim=1) 
        
    
#--------------------------------------------------------------------
def mlp_cnn(state_size, K, F=[4,3,2], stride=[1],relu_type='selu', pbc=True, bias=True, momentum=[0,0]):
    '''Input: State (batch_size, Dp, N) for 1d lattice,
                    (batch_size, Dp, L, W) for 2d lattice.
       Output: [logphis, thetas].
    '''
    K = K[0] if type(K) is list and len(K) == 1 else K
    stride = stride[0] if type(stride) is list and len(stride) == 1 else stride
    dim = len(state_size) - 1
    dimensions = '1d' if dim == 1 else '2d'
    layers = len(F)
    Dp = state_size[-1]

    input_layer = CNN_complex_layer(K=K, F_in=Dp, F_out=F[0], stride=stride, layer_name='1st', dimensions=dimensions,
                                    relu_type=relu_type, pbc=pbc, bias=bias)
    output_layer = OutPut_complex_layer(K,F[-1], pbc=pbc, dimensions=dimensions, momentum=momentum)

    # input layer
    cnn_layers = [input_layer]
    cnn_layers += [CNN_complex_layer(K=K, F_in=F[i-1], F_out=F[i], stride=stride, dimensions=dimensions,
                        relu_type=relu_type, pbc=pbc, bias=bias) for i in range(1,layers)]
    cnn_layers += [output_layer]

    def weight_init(m):
        if isinstance(m, ComplexConv):
            nn.init.xavier_uniform_(m.conv_re.weight, gain=1/np.sqrt(2))
            nn.init.xavier_uniform_(m.conv_im.weight, gain=1/np.sqrt(2))
            if m.conv_re.bias is not None:
                nn.init.zeros_(m.conv_re.bias)
                nn.init.zeros_(m.conv_im.bias)
        elif isinstance(m, ComplexLinear):
            nn.init.xavier_uniform_(m.linear_re.weight, gain=1/np.sqrt(2))
            nn.init.xavier_uniform_(m.linear_im.weight, gain=1/np.sqrt(2))

    model = nn.Sequential(*cnn_layers)
    model.apply(weight_init)
    complex_init(model)
    return model

# physical symmetry 
# -------------------------------------------------------------------------------------------------
# symmetry functions
def get_max_label1d(full_x, N):
    # input shape of full_x: (batch_size, sym_N, Dp, 2*N-1)
    full_xv = full_x[:,:,0,:]
    batch_size, pad_N = full_xv.shape[0], full_xv.shape[-1]
    pow_list = torch.arange(N-1, -1, -1, device=full_x.device)
    filters = torch.pow(1.5, pow_list).reshape(1,1,N)
    full_xv = nn.functional.conv1d(full_xv.reshape(-1, 1, pad_N), filters)
    max_label = full_xv.reshape(batch_size, -1).argmax(dim=1)
    return [max_label//N, max_label%N]  

def get_max_label2d(full_x, L, W):
    # input shape of full_x: (batch_size, sym_N, Dp, 2*L-1, 2*W-1)
    """
    Future: act the symmetry function on the filters
    """
    full_xv = full_x[:,:,0,:,:]
    batch_size, pad_L, pad_W = full_xv.shape[0],full_xv.shape[-2], full_xv.shape[-1]
    pow_list = torch.arange(L*W-1, -1, -1, device=full_x.device)
    filters = torch.pow(1.5, pow_list).reshape(1,1,L,W)
    full_xv = nn.functional.conv2d(full_xv.reshape(-1, 1, pad_L, pad_W), filters)
    max_label = full_xv.reshape(batch_size, -1).argmax(dim=1)
    return [max_label//(L*W), (max_label%(L*W))//W, max_label%W]

def translation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    # output shape of pad_x: (batch_size, Dp, 2*N-1) or (batch_size, Dp, 2*L-1, 2*W-1)
    dimension = len(x.shape) - 2
    if dimension == 1:
        N = x.shape[-1]
        pad_x = periodic_padding(x, N, dimensions='1d')
        return pad_x, 1
    else:
        L, W = x.shape[-2], x.shape[-1]
        pad_x = periodic_padding(x, [L, W], dimensions='2d')
        return pad_x, 1

def inverse(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    x_inverse = torch.flip(x, dims=[1])
    full_x = torch.stack((x, x_inverse), dim=1)
    return full_x.reshape([-1] + list(x.shape[1:])), 2

def identity(x):
    return x, 1

def transpose(x):
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        xT = torch.transpose(x, 2, 3)
        full_x = torch.stack((x, xT), dim=1)
        return full_x.reshape([-1] + list(x.shape[1:])), 2

def c6rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        x60 = rot60(x, num=1, dims=[2,3], center=[0])
        x120 = rot60(x, num=2, dims=[2,3], center=[0])
        x180 = rot60(x, num=3, dims=[2,3], center=[0])
        x240 = rot60(x, num=4, dims=[2,3], center=[0])
        x300 = rot60(x, num=5, dims=[2,3], center=[0])
        full_x = torch.stack((x,x60,x120,x180,x240,x300), dim=1)
        return full_x.reshape([-1] + list(x.shape[1:])), 6
    
def c4rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        return c2rotation(x)
    else:
        x90 = torch.rot90(x, 1, dims=[2,3])
        x180 = torch.rot90(x, 2, dims=[2,3])
        x270 = torch.rot90(x, 3, dims=[2,3])
        full_x =  torch.stack((x, x90, x180, x270), dim=1)
        return full_x.reshape([-1] + list(x.shape[1:])), 4
    
def c3rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        x120 = rot60(x, num=2, dims=[2,3], center=[1])
        x240 = rot60(x, num=4, dims=[2,3], center=[1])
        full_x =  torch.stack((x, x120, x240), dim=1)
        return full_x.reshape([-1] + list(x.shape[1:])), 3

def c2rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1:
        x180 = torch.rot90(x, 2, dims=[2])
    else:
        x180 = torch.rot90(x, 2, dims=[2, 3])
    full_x = torch.stack((x, x180), dim=1)
    return full_x.reshape([-1] + list(x.shape[1:])), 2

# -------------------------------------------------------------------------------------------------
# symmetry network

class sym_model(nn.Module):
    def __init__(self, state_size, K, F=[4,3,2], stride=[1], relu_type='selu', 
                pbc=False, bias=True, momentum=[0,0], sym_funcs=[identity]):
        super(sym_model,self).__init__()
        self.model = mlp_cnn(state_size=state_size, K=K, F=F, stride=stride, relu_type=relu_type, 
                    pbc=pbc, bias=bias, momentum=momentum)
        self.sym_funcs = sym_funcs
    
    def symmetry(self, x):
        xdevice = x.device
        dimensions = len(x.shape) - 2
        old_shape = x.shape
        batch_size = x.shape[0]
        sym_N = 1

        sym_x = torch.zeros_like(x)
        for func in self.sym_funcs:
            x, n = func(x)
            sym_N *= n
        single_shape = list(x.shape[1:])
        full_x = x.reshape([batch_size, sym_N] + single_shape)
        
        if dimensions == 1:
            # shape of output x: (batch_size, sym_N, Dp, 2*N-1)
            N = old_shape[-1]
            max_label = get_max_label1d(full_x, N)
            XX, YY = torch.meshgrid(torch.arange(N, device=xdevice), torch.arange(batch_size, device=xdevice))
            full_x = full_x[range(batch_size), max_label[0],:,:]
            sym_x[YY,:,XX] = full_x[YY,:,XX+max_label[1]]
        else:
            # shape of output x: (batch_size, sym_N, Dp, 2*L-1, 2*W-1)
            L, W = old_shape[-2], old_shape[-1]
            max_label = get_max_label2d(full_x, L, W)
            XX, YY, ZZ = torch.meshgrid(torch.arange(W, device=xdevice), 
                        torch.arange(L, device=xdevice), torch.arange(batch_size, device=xdevice))
            full_x = full_x[range(batch_size), max_label[0],:,:,:]
            sym_x[ZZ,:,YY,XX] = full_x[ZZ,:,YY+max_label[1],XX+max_label[2]]
        return sym_x
    
    # apply symmetry
    def forward(self, x, return_state=False):
        sym_x = self.symmetry(x)
        # save GPU memory with unique array
        sym_x_unique, inverse_indices = torch.unique(sym_x,return_inverse=True,dim=0)
        sym_x_unique = self.model(sym_x_unique)
        if return_state:
            return (torch.stack((sym_x_unique[inverse_indices,0], sym_x_unique[inverse_indices,1]), dim=1), 
                    sym_x.squeeze().detach().numpy())
        else:
            return torch.stack((sym_x_unique[inverse_indices,0], sym_x_unique[inverse_indices,1]), dim=1)

def mlp_cnn_sym(state_size, K, F=[4,3,2], stride=[1], relu_type='selu', 
                pbc=True, bias=True, momentum=[0,0], sym_funcs=[identity]):
    model = sym_model(state_size=state_size, K=K, F=F, stride=stride, relu_type=relu_type, 
                    pbc=pbc, bias=bias, momentum=momentum, sym_funcs=sym_funcs)
    return model

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    seed = 286
    torch.manual_seed(seed)
    np.random.seed(seed)
    logphi_model = mlp_cnn_sym([4,4,2], 4, [2,2], stride=[1], pbc=False, relu_type='selu', 
                                  bias=True, momentum=[0,0], sym_funcs=[c6rotation, transpose, translation])
    #op_model = mlp_cnn([10,10,2], 2, [2],complex_nn=True, output_size=2, relu_type='sReLU', bias=True)
    # print(logphi_model)
    print(get_paras_number(logphi_model))
    import sys
    sys.path.append('..')
    from ops.HS_spin2d import get_init_state
    state0,_ = get_init_state([4,4,2], kind='rand', n_size=100)
    state_zero = torch.from_numpy(state0[0][None,...])
    state_zero = torch.stack((state_zero, torch.zeros_like(state_zero)), dim=1)
    state_t0 = rot60(torch.from_numpy(state0[0][None,...]).float(), num=4, dims=[2,3], center=[1])
    t1 = rot60(torch.from_numpy(state0[0][None,...]).float(), num=3, dims=[2,3], center=[1])
    t2 = rot60(t1, num=1, dims=[2,3], center=[1])
    print(state0[0])
    print(state_t0)
    print(t2 - state_t0)
    # print(complex_periodic_padding(state_zero, [3,3], [1,1], dimensions='2d'))
    #print(state0.shape)
    import time
    tic = time.time()
    state0 = torch.from_numpy(state0).float()
    #state0r = torch.roll(state0, shifts=1, dims=2)
    # state0r = torch.rot90(state0, 1, dims=[2,3])
    state0r = rot60(state0, 2, dims=[2,3], center=[0])
    #state0r = torch.transpose(state0, 2, 3)
    tt1, sym_state = logphi_model(state0, return_state=True)
    tt2 = logphi_model(torch.from_numpy(sym_state).float())
    print(tt1)
    print(time.time() - tic)
    #for name,p in logphi_model.named_parameters():
    #    print(name)
    #import scipy.io as sio
    #sio.savemat('test2.mat', {'tt2':tt})
    # print(logphi_model(torch.from_numpy(state0).float())[:3])
    #state_t = torch.roll(state_t0, shifts=2, dims=2)
    # state_t = torch.rot90(torch.from_numpy(state0[0][None,...]).float(),2, dims=[2,3])
    #print(state_t)
    #print(logphi_model(state_t))
    #logphi_model_sym, _ = mlp_cnn_sym([4,4,2], 2, [2,2], stride=[1], complex_nn=True,
    #                       output_size=2, relu_type='selu', bias=True, momentum=[0,0], sym_funcs=[identity])
    #logphi_model_sym.load_state_dict(logphi_model.state_dict())
    # logphi_model_sym.eval()
    #print(logphi_model_sym(torch.from_numpy(state0).float())[1])
    # print(list(logphi_model(torch.from_numpy(state0).float()).size()))
    # x, M = reflection(torch.from_numpy(state0))
    
    #pad = (0,0,0,2)
    #x = torch.nn.functional.pad(torch.from_numpy(state0[0][None,...]).float(), pad, mode='circular', value=0)
    #print(x)
    #print(x.shape)
    #print(x.shape)
    #print(complex_periodic_padding(torch.from_numpy(state0[0]).reshape(1,2,1,4,4),[2,2],'2d'))

    # phi = logphi_model(torch.from_numpy(state0).float())
    # # logphi = phi[:,0].reshape(1,-1)
    # # theta = phi[:,1].reshape(1,-1)
    # print(phi[:,0].std()/phi[:,0].mean())
    # print(phi[:,1].std()/phi[:,1].mean(),phi[:,1].max(), phi[:,1].min())
    
    # logphi_model_sym, _ = mlp_cnn_sym([4,4,2], 2, [3,2],complex_nn=True,
    #                        output_size=2, relu_type='softplus2', bias=True)
    # phi = logphi_model_sym(torch.from_numpy(state0).float())
    # print(get_paras_number(logphi_model_sym))
