# encoding: utf-8
import sys
sys.path.append('..')

import torch.nn as nn
import torch
import numpy as np
from utils_ppo import rot60
import copy
# from snake.activations import Snake
# from utils import extract_weights,load_weights

def periodic_padding(x, kernel_size, dimensions, mode='circular'):
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
    return nn.functional.pad(x, pad, mode)

def get_paras_number(net, filter_num=0):
    # total_num = sum(p.numel() for p in net.parameters())
    # trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_num = 0
    trainable_num = 0
    for name,p in net.named_parameters():
        if 'conv' in name and 'weight' in name and filter_num>0:
            total_num += p.shape[0]*p.shape[1]*(filter_num)
        else:
            total_num += p.numel()

        if p.requires_grad:
            if 'conv' in name and 'weight' in name and filter_num>0:
                trainable_num += p.shape[0]*p.shape[1]*(filter_num)
            else:
                trainable_num += p.numel()
    if filter_num == 21:
        return {'Total': total_num, 'Trainable': trainable_num, 'filter_shape':'custom_sq'}
    elif filter_num == 19:
        return {'Total': total_num, 'Trainable': trainable_num, 'filter_shape':'custom_tri'}
    else:
        return {'Total': total_num, 'Trainable': trainable_num, 'filter_shape':'default'}

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
            if 'conv_re' in name: #or name.split(".")[2] == 'linear_re':
                # p.data = torch.from_numpy(np.cos(theta)).float()*rho/w
                p.data /= w
            elif 'conv_im' in name: #or name.split(".")[2] == 'linear_im':
                # p.data = torch.from_numpy(np.sin(theta)).float()*rho/w
                p.data /= w
    return

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
    #unique, inverse_indices = torch.unique_consecutive(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype,
                        device=inverse_indices.device)
    inverse, perm = inverse_indices.flip([0]), perm.flip([0])
    indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, indices, inverse_indices

# COMPLEX NEURAL NETWORK
# ----------------------------------------------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear,self).__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x): # shpae of x : [batch,2,in_features]
        real = self.linear_re(x[:,0]) - self.linear_im(x[:,1])
        imag = self.linear_re(x[:,1]) + self.linear_im(x[:,0])
        return torch.stack((real, imag),dim=1)

class RealLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(RealLinear,self).__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x): # shpae of x : [batch,2,in_features]
        real = self.linear_re(x[:,0])
        imag = self.linear_im(x[:,1])
        output = torch.stack((real, imag),dim=1)
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
    def __init__(self, act_size, relu_type='zReLU', alpha=0.35):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexReLU, self).__init__()
        self.type = relu_type
        self.b = -0.05
        #self.act = nn.Parameter(torch.randint(1, 10, (2,)).float())
        self.act = nn.Parameter(torch.tensor(alpha))
        #self.act = nn.Parameter(torch.ones(act_size) * alpha)
        #m = torch.distributions.exponential.Exponential(torch.tensor([0.1]))
        #self.act = nn.Parameter((m.rsample(act_size)).squeeze()) # random init = mix of frequencies
        # print(self.act.shape)
        self.act.requires_grad = True
        # self.act_im = nn.Parameter(torch.tensor(2.))
        # self.act_im.requires_grad = False

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
            z = torch.relu(torch.abs(z) + self.b) * torch.exp(1.j * torch.angle(z)) 
            return torch.stack((z.real, z.imag), dim=1)
        elif self.type == 'softplus':
            z = torch.log(1. + torch.exp(z))
            return torch.stack((z.real, z.imag), dim=1)
        elif self.type == 'softplus2':
            real = torch.log(1. + torch.exp(real))
            imag = torch.log(1. + torch.exp(imag))
            return torch.stack((real, imag), dim=1)
        elif self.type == 'selu':
            return torch.stack((nn.functional.selu(real), nn.functional.selu(imag)), dim=1)
        elif self.type == 'serelu':
            return torch.stack((torch.selu(real), nn.functional.relu(imag)), dim=1)
        elif self.type == 'snake':
            real = nn.functional.selu(real)
            imag = nn.functional.relu(imag + torch.sin(self.act*imag)*torch.sin(self.act*imag)/self.act)
            # imag = imag + (1 - torch.cos(2*self.act*imag))/(2*self.act)
            return torch.stack((real, imag), dim=1)
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

class CNN_real_layer(nn.Module):
    def __init__(self,Kre,Kim,F_in,F_out,stride,complex_relu,layer_name='mid',
                 pbc=True, bias=True, dimensions='1d', groups=1, 
                 custom_filter=False, lattice_shape=None, device='cpu'):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN_real_layer,self).__init__()
        self.Kre = [Kre,Kre] if type(Kre) is int and dimensions=='2d' else Kre
        self.Kim = [Kim,Kim] if type(Kim) is int and dimensions=='2d' else Kim
        self._pbc = pbc
        self.layer_name = layer_name
        self.dimensions = dimensions
        self.stride = [stride, stride] if type(stride) is int and dimensions=='2d' else stride

        if dimensions == '1d':
            self.conv_re = nn.Conv1d(F_in,F_out,self.Kre,stride,0, groups=groups,bias=bias)
            self.conv_im = nn.Conv1d(F_in,F_out,self.Kim,stride,0, groups=groups,bias=bias)
        else:
            self.conv_re = nn.Conv2d(F_in,F_out,self.Kre,stride,0, groups=groups,bias=bias)
            self.conv_im = nn.Conv2d(F_in,F_out,self.Kim,stride,0, groups=groups,bias=bias)

        self.complex_relu = complex_relu
        self.custom_filter = custom_filter

        if lattice_shape == 'sq':
            """
            tensor([
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    ])
            """
            self.filter = torch.tensor([
                                    [0, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    ], device=device).reshape(1,1,5,5)
            # print(self.filter)

        else:
            """
            tensor([
                    [0, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0],
                    ])

            NNNN:
            tensor([
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    ])
            """
            self.filter = torch.tensor([
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    ], device=device).reshape(1,1,5,5)
            # print(self.filter)

    def forward(self, x, _only_theta=False, _only_phi=False):
        if self.layer_name == '1st':
            xre = x.clone()
            xim = x.clone()
        else:
            xre = x[:,0]
            xim = x[:,1]

        if self._pbc:
            xre = periodic_padding(xre, self.Kre, dimensions=self.dimensions, mode='circular')
            xim = periodic_padding(xim, self.Kim, dimensions=self.dimensions, mode='circular')
        else:
            xre = periodic_padding(xre, self.Kre, dimensions=self.dimensions, mode='constant')
            xim = periodic_padding(xim, self.Kim, dimensions=self.dimensions, mode='constant')

        if self.custom_filter:
            self.conv_re.weight.data = self.conv_re.weight.data*self.filter
            self.conv_im.weight.data = self.conv_im.weight.data*self.filter
        
        if _only_phi:
            xre = self.conv_re(xre)
            xim = torch.zeros_like(xre)
        else:
            xim = self.conv_im(xim)
            if _only_theta:
                xre = torch.zeros_like(xim)
            else:
                xre = self.conv_re(xre)

        x = torch.stack((xre, xim), dim=1)
        return self.complex_relu(x)

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
        #self.linear = RealLinear(F, 1, bias=False)
        #self.complex_relu = complex_relu

    def forward(self,x,_only_theta=False,_only_phi=False):
        # shape of complex x: (batch_size, 2, F, N) or (batch_size, 2, F, L, W)
        # variance scaling
        norm = np.sqrt(np.prod(x.shape[2:]))
        z = (x[:,0] + 1j*x[:,1])/norm

        if self._pbc and np.sum(self.momentum) != 0:
            z = translation_phase(z, k=self.momentum, dimensions=self.dimensions)

        z = z.sum(dim=[1,2]) if self.dimensions=='1d' else z.sum(dim=[1,2,3])
        return torch.stack((z.real, z.imag), dim=1)
        #x = torch.stack((z.real, z.imag), dim=1)
        #return self.linear(x).squeeze(dim=-1)
    
#--------------------------------------------------------------------
class mySequential(nn.Sequential):
    def forward(self, input, _only_theta=False, _only_phi=False):
        for module in self._modules.values():
            input = module(input, _only_theta, _only_phi)
        return input

def mlp_cnn(state_size, K, F=[4,3,2], stride0=[1], stride=[1], 
    relu_type='selu', pbc=True, bias=True, momentum=[1,0], groups=1, alpha=0.35,
    custom_filter=False, lattice_shape=None, device='cpu'):
    '''Input: State (batch_size, Dp, N) for 1d lattice,
                    (batch_size, Dp, L, W) for 2d lattice.
       Output: [logphis, thetas].
    '''
    K = K[0] if type(K) is list and len(K) == 1 else K
    stride0 = stride0[0] if type(stride0) is list and len(stride0) == 1 else stride0
    stride = stride[0] if type(stride) is list and len(stride) == 1 else stride
    dim = len(state_size) - 1
    dimensions = '1d' if dim == 1 else '2d'
    act_size = [state_size[0]] if dim == 1 else [state_size[0], state_size[1]]
    layers = len(F)

    Dp = state_size[-1]
    complex_relu = ComplexReLU(act_size, relu_type, alpha=alpha)

    # input_layer = CNN_complex_layer(K=K, F_in=Dp, F_out=F[0], stride=stride0, layer_name='1st', dimensions=dimensions,
    #                                 complex_relu=complex_relu, pbc=pbc, bias=bias, groups=groups,
    #                                 custom_filter=custom_filter, lattice_shape=lattice_shape, device=device)
    input_layer = CNN_real_layer(Kre=K, Kim=K, F_in=Dp, F_out=F[0], stride=stride0, layer_name='1st', dimensions=dimensions,
                                    complex_relu=complex_relu, pbc=pbc, bias=bias, groups=groups,
                                    custom_filter=custom_filter, lattice_shape=lattice_shape, 
                                    device=device)
    output_layer = OutPut_complex_layer(K,F[-1], pbc=pbc, dimensions=dimensions, momentum=momentum)

    # input layer
    cnn_layers = [input_layer]
    # cnn_layers += [CNN_complex_layer(K=K, F_in=F[i-1], F_out=F[i], stride=stride, dimensions=dimensions,
    #                     complex_relu=complex_relu, pbc=pbc, bias=bias, groups=groups,
    #                     custom_filter=custom_filter, lattice_shape=lattice_shape, device=device) for i in range(1,layers)]
    cnn_layers += [CNN_real_layer(Kre=K, Kim=K, F_in=F[i-1], F_out=F[i], stride=stride, dimensions=dimensions,
                        complex_relu=complex_relu, pbc=pbc, bias=bias, groups=groups,
                        custom_filter=custom_filter, lattice_shape=lattice_shape, 
                        device=device) for i in range(1,layers)]
    cnn_layers += [output_layer]

    return mySequential(*cnn_layers)

# physical symmetry
# -------------------------------------------------------------------------------------------------
# symmetry functions
def translation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    # output shape of pad_x: (batch_size, Dp, 2*N-1) or (batch_size, Dp, 2*L-1, 2*W-1)
    dimension = len(x.shape) - 2
    if dimension == 1:
        N = x.shape[-1]
        pad_x = periodic_padding(x, N, dimensions='1d', mode='circular')
        return pad_x, 1
    else:
        L, W = x.shape[-2], x.shape[-1]
        pad_x = periodic_padding(x, [L, W], dimensions='2d', mode='circular')
        return pad_x, 1

def inverse(x, dim=1):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    N = 2
    x_inverse = torch.flip(x, dims=[dim])
    full_x = torch.stack((x, x_inverse), dim=1).reshape([-1] + list(x.shape[1:]))
    return full_x, N

def identity(x):
    return x, 1

def transpose(x):
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 2
        xT = torch.transpose(x, 2, 3)
        full_x = torch.stack((x, xT), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def reflection(x):
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 2
        xR = torch.flip(x, dims=[2])
        full_x = torch.stack((x, xR), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c6rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 6
        x60 = rot60(x, num=1, dims=[2,3], center=[0])
        x120 = rot60(x, num=2, dims=[2,3], center=[0])
        x180 = rot60(x, num=3, dims=[2,3], center=[0])
        x240 = rot60(x, num=4, dims=[2,3], center=[0])
        x300 = rot60(x, num=5, dims=[2,3], center=[0])
        full_x = torch.stack((x,x60,x120,x180,x240,x300), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c4rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        return c2rotation(x)
    else:
        N = 4
        x90 = torch.rot90(x, 1, dims=[2,3])
        x180 = torch.rot90(x, 2, dims=[2,3])
        x270 = torch.rot90(x, 3, dims=[2,3])
        full_x = torch.stack((x, x90, x180, x270), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c3rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 3
        x120 = rot60(x, num=2, dims=[2,3], center=[1])
        x240 = rot60(x, num=4, dims=[2,3], center=[1])
        full_x = torch.stack((x, x120, x240), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

def c2rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    N = 2
    if dimension == 1:
        x180 = torch.rot90(x, 2, dims=[2])
    else:
        x180 = torch.rot90(x, 2, dims=[2, 3])
        full_x = torch.stack((x, x180), dim=1).reshape([-1] + list(x.shape[1:]))
        return full_x, N

"""
need to update for larger system
"""
def get_max_label1d(full_x, N):
    # input shape of full_x: (batch_size, sym_N, Dp, 2*N-1)
    full_xv = full_x[:,:,0,:]
    batch_size, pad_N = full_xv.shape[0], full_xv.shape[-1]
    pow_list = torch.arange(N-1, -1, -1, device=full_x.device)
    filters = torch.pow(torch.tensor(1.5, dtype=torch.float64), pow_list).reshape(1,1,N)
    full_xv = nn.functional.conv1d(full_xv.reshape(-1, 1, pad_N).double(), filters)
    # max_label = full_xv.reshape(batch_size, -1).argmax(dim=1)
    max_num, max_label = torch.max(full_xv.reshape(batch_size, -1), dim=1)
    x = torch.div(max_label, N, rounding_mode='floor')
    y = max_label%N
    return [x, y], max_num

def get_max_label2d(full_x, L, W):
    # input shape of full_x: (batch_size, sym_N, Dp, 2*L-1, 2*W-1)
    """
    Future: act the symmetry function on the filters
    """
    full_xv = full_x[:,:,0,:,:]
    batch_size, pad_L, pad_W = full_xv.shape[0],full_xv.shape[-2], full_xv.shape[-1]
    pow_list = torch.arange(L*W-1, -1, -1, device=full_x.device)
    filters = torch.pow(torch.tensor(1.5, dtype=torch.float64), pow_list).reshape(1,1,L,W)
    full_xv = nn.functional.conv2d(full_xv.reshape(-1, 1, pad_L, pad_W).double(), filters)
    max_num, max_label = torch.min(full_xv.reshape(batch_size, -1), dim=1)
    x = torch.div(max_label, (L*W), rounding_mode='floor')
    y = torch.div(max_label%(L*W), W, rounding_mode='floor')
    z = max_label%W
    return [x, y, z], max_num

# -------------------------------------------------------------------------------------------------
# symmetry network
# Group CNN 
class sym_model(nn.Module):
    def __init__(self, state_size, K, F=[4,3,2], stride0=[1], stride=[1], relu_type='selu', device="cpu",
                pbc=True, bias=True, momentum=[0,0], sym_funcs=[identity], 
                apply_unique=False, MPphase=False, MPtype='NN', alpha=0.35, 
                custom_filter=False, lattice_shape=None):
        super(sym_model,self).__init__()

        self.sym_funcs = sym_funcs
        # calculate number of groups 
        net_state_size = state_size[:]
        x0 = torch.zeros(list(np.roll(net_state_size, shift=1)))[None, ...]
        dimensions = len(state_size) - 1
        _, self.sym_N = self.symmetry(x0)
        self.apply_unique = apply_unique
        self.MPphase = MPphase
        self._device = device
        self._pbc = pbc
        self._only_theta = False
        self._only_phi = False

        if lattice_shape == 'sq':
            """
            tensor([
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    ])
            """
            self.filter = torch.tensor([
                                    [0, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 0],
                                    ], device=device).reshape(1,1,5,5)
            # print(self.filter)
        else:
            """
            tensor([
                    [0, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0],
                    ])

            NNNN:
            tensor([
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    ])
            """
            self.filter = torch.tensor([
                                    [1, 1, 1, 0, 0],
                                    [1, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    ], device=device).reshape(1,1,5,5)
            # print(self.filter)
        
        self.filter_num = self.filter.sum().cpu().item()
        self.filter_num *= 1 if custom_filter else 0

        self.model = mlp_cnn(state_size=state_size, K=K, F=F, stride0=stride0, 
                stride=stride, relu_type=relu_type, 
                pbc=pbc, bias=bias, momentum=momentum, groups=1, alpha=alpha,
                custom_filter=custom_filter, lattice_shape=lattice_shape, 
                device=device)

        def weight_init(m):
            # if isinstance(m, ComplexConv) or isinstance(m, RealConv):
            #     nn.init.xavier_uniform_(m.conv_re.weight, gain=1/np.sqrt(2))
            #     nn.init.xavier_uniform_(m.conv_im.weight, gain=1/np.sqrt(2))
            #     #nn.init.kaiming_uniform_(m.conv_im.weight)
            #     if m.conv_re.bias is not None:
            #         nn.init.zeros_(m.conv_re.bias)
            #         nn.init.zeros_(m.conv_im.bias)
            if isinstance(m, RealLinear):
                nn.init.ones_(m.linear_re.weight)
                nn.init.ones_(m.linear_im.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1/np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.model.apply(weight_init)
        # complex_init(self.model)
        self.model = self.model.to(device)

        if dimensions == 1:
            sl_row = torch.zeros(state_size[0], device=self._device)
            sl_row[torch.arange(0, state_size[0], 2, device=self._device)] = 1

            sublattice = torch.zeros(state_size[0], device=self._device)
            sublattice[:] = sl_row
            self.sublattice = sublattice.reshape(1,1,state_size[0])

        else:
            # Marshall-Peierls rule
            if MPtype == 'NN':
                sl_row_even = torch.zeros(state_size[0], device=self._device)
                sl_row_even[torch.arange(0, state_size[0], 2, device=self._device)] = 1

                sl_row_odd = torch.zeros(state_size[0], device=self._device)
                sl_row_odd[torch.arange(1, state_size[0], 2, device=self._device)] = 1

            elif MPtype == 'stripe':
                sl_row_even = torch.zeros(state_size[0], device=self._device)
                sl_row_even[torch.arange(0, state_size[0], 2, device=self._device)] = 1

                sl_row_odd = sl_row_even

            elif MPtype == 'single':
                sl_row_even = torch.zeros(state_size[0], device=self._device)
                sl_row_even[torch.arange(0, state_size[0], 2, device=self._device)] = 1

                sl_row_odd = sl_row_even*0
            
            else:
                sl_row_even0 = torch.zeros(state_size[0]//2, device=self._device)
                sl_row_even0[torch.arange(0, state_size[0]//2, 2, device=self._device)] = 1

                sl_row_odd0 = torch.zeros(state_size[0]//2, device=self._device)
                sl_row_odd0[torch.arange(1, state_size[0]//2, 2, device=self._device)] = 1

                sl_row_even = torch.cat((sl_row_even0, sl_row_even0), 0)
                sl_row_odd  = torch.cat((sl_row_odd0 , sl_row_even0), 0)

            sublattice = torch.zeros(state_size[0], state_size[1], device=self._device)
            sublattice[torch.arange(0, state_size[0], 2),:] = sl_row_even
            sublattice[torch.arange(1, state_size[0], 2),:] = sl_row_odd
            self.sublattice = sublattice.reshape(1,1,state_size[0], state_size[1])

    def symmetry(self, x):
        sym_N = 1
        for func in self.sym_funcs:
            x, n = func(x)
            sym_N *= n
        return x, sym_N
    
    def pick_sym_config(self, x):
        xdevice = x.device
        dimensions = len(x.shape) - 2
        old_shape = x.shape
        batch_size = x.shape[0]
        sym_N = 1

        # shape of x: (1, Dp, L, W)
        symfuncs = copy.deepcopy(self.sym_funcs)
        sym_x = torch.zeros_like(x)
        if self._pbc:
            symfuncs.append(translation)
        for func in symfuncs:
            x, n = func(x)
            sym_N *= n

        single_shape = list(x.shape[1:])
        full_x = x.reshape([batch_size, sym_N] + single_shape)

        if dimensions == 1:
            # shape of output x: (batch_size, sym_N, Dp, 2*N-1)
            N = old_shape[-1]
            max_label, max_num = get_max_label1d(full_x, N)
            XX, YY = torch.meshgrid(torch.arange(N, device=xdevice), torch.arange(batch_size, device=xdevice))
            full_x = full_x[range(batch_size), max_label[0],:,:]
            sym_x[YY,:,XX] = full_x[YY,:,XX+max_label[1]]
        else:
            # shape of output x: (batch_size, sym_N, Dp, 2*L-1, 2*W-1)
            L, W = old_shape[-2], old_shape[-1]
            max_label, _ = get_max_label2d(full_x, L, W)
            XX, YY, ZZ = torch.meshgrid(torch.arange(W, device=xdevice), 
                        torch.arange(L, device=xdevice), torch.arange(batch_size, device=xdevice))
            full_x = full_x[range(batch_size), max_label[0],:,:,:]
            sym_x[ZZ,:,YY,XX] = full_x[ZZ,:,YY+max_label[1],XX+max_label[2]]

        return sym_x

        # pi phase from Marshall-Peierls rule
    def get_logMa(self, x):
        if self.MPphase:
            Ma = nn.functional.conv2d(x[:,0].reshape(x.shape[0],1,x.shape[2],x.shape[3]), 
                                    self.sublattice.to(x.dtype)).squeeze().reshape(-1)
            Ma = torch.round(Ma).long()
            Ma = (-1)**Ma
        else: 
            Ma = torch.tensor(1. + 0*1j, device=x.device).reshape(1)
        logMa = torch.log(Ma.to(torch.complex64))
        logMa = torch.stack((logMa.real, logMa.imag), dim=1)
        return logMa

    # apply symmetry
    def forward(self, x, apply_sym=True):
        if apply_sym:
            xshape = x.shape
            x, sym_N = self.symmetry(x)
            logMa = self.get_logMa(x).to(x.dtype)

            if self.apply_unique:
                _, indices, inverse_indices = unique(x[:,0], dim=0)
                x = self.model(x[indices], self._only_theta, self._only_phi)[inverse_indices,:]/np.sqrt(sym_N)
                x += logMa/sym_N
                x = x.reshape(xshape[0], sym_N, 2).sum(dim=1)
            else:
                x = self.model(x, self._only_theta, self._only_phi)/np.sqrt(sym_N)
                x += logMa/sym_N
                x = x.reshape(xshape[0], sym_N, 2).sum(dim=1)
            return x
                
        else:
            logMa = self.get_logMa(x).to(x.dtype)
            x = self.model(x, self._only_theta, self._only_phi) + logMa
            return x

def mlp_cnn_sym(state_size, K, F=[4,3,2], stride0=[1], stride=[1], relu_type='selu', device="cpu",
                pbc=True, bias=True, momentum=[0,0], sym_funcs=[identity], 
                apply_unique=False, MPphase=False, MPtype='NN', alpha=0.35,
                custom_filter=False, lattice_shape=None):
    return sym_model(state_size=state_size, K=K, F=F, stride0=stride0, 
                stride=stride, relu_type=relu_type, device=device, pbc=pbc, bias=bias, 
                momentum=momentum, sym_funcs=sym_funcs, apply_unique=apply_unique, 
                MPphase=MPphase, MPtype=MPtype, alpha=alpha, 
                custom_filter=custom_filter, lattice_shape=lattice_shape)

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    seed = 286
    torch.manual_seed(seed)
    np.random.seed(seed)
    logphi_model = mlp_cnn_sym([6,6,2], 5, [8,6,4], stride0=[1], stride=[1], pbc=True, relu_type='snake', 
    bias=True, momentum=[0,0], sym_funcs=[c4rotation, transpose], MPphase=False, MPtype='NN', 
    custom_filter=True, lattice_shape='sq')
    #op_model = mlp_cnn([10,10,2], 2, [2],complex_nn=True, output_size=2, relu_type='sReLU', bias=True)
    print(logphi_model)
    print(get_paras_number(logphi_model))
    import sys
    sys.path.append('..')
    from ops.HS_spin2d import get_init_state
    state0,_ = get_init_state([6,6,2], kind='rand', n_size=10000)
    state_zero = torch.from_numpy(state0[0][None,...])
    state_zero = torch.stack((state_zero, torch.zeros_like(state_zero)), dim=1)
    state_t0 = rot60(torch.from_numpy(state0[0][None,...]).float(), num=4, dims=[2,3], center=[1])
    t1 = rot60(torch.from_numpy(state0[0][None,...]).float(), num=3, dims=[2,3], center=[1])
    t2 = rot60(t1, num=1, dims=[2,3], center=[1])
    #print(state0[0].shape)
    #print(state_t0)
    #print(t2 - state_t0)
    # print(complex_periodic_padding(state_zero, [3,3], [1,1], dimensions='2d'))
    #print(state0.shape)
    import time
    tic = time.time()
    #state0 = torch.from_numpy(state0).float()
    #state0r = torch.transpose(state0, 2, 3)
    #tt2 = logphi_model(state0r).detach().numpy()
    #print(np.max(tt - tt2))
    #state_unique = np.unique(state0, return_inverse=True, return_counts=True, axis=0)
    #state0 = torch.from_numpy(state0).float()
    #tt = logphi_model(state0).detach().numpy()[:,0]
    #state_unique, inverse_indices, counts = torch.unique(state0, return_inverse=True, return_counts=True, dim=0)
    #print(time.time() - tic)
    #print(tt[inverse_indices].shape, state_unique.shape)
    logphi_model._only_theta = True
    tic = time.time()
    state0 = torch.from_numpy(state0).float()
    #print(state0[0])
    #print(torch.flip(state0[0], dims=[2]))
    # state0r = torch.flip(state0, dims=[1])
    state0r = torch.flip(state0, dims=[3])
    #state0r = torch.rot90(state0, 1, dims=[2,3])
    #state0r = rot60(state0, 1, dims=[2,3], center=[0])
    #state0r = torch.transpose(state0, 2, 3)
    #logphi_model = logphi_model.double()
    tt1 = logphi_model(state0)
    #print(logphi_model.model.forward.__dir__())
    tt2 = logphi_model(state0r)
    # tt1.sum().backward()
    # for name, p in logphi_model.named_parameters():
    #     print(name, p.shape)

    # state0x = logphi_model.pick_sym_config_new(state0[0][None,...])
    # print(state0[0])
    # print(state0x)
    # print(0==torch.max(tt1 - tt2))
    #print(tt1)
    print(tt1)
    #print(logphi_model(state0[0][None,...]))
    print(time.time() - tic)

        
    # for name,p in logphi_model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    # for name, p in logphi_model.named_parameters():
    #    print(name)
    # param_grp = []
    # for idx, m in enumerate(logphi_model.modules()):
    #     if isinstance(m, ComplexConv):
    #         param_grp.append(m.conv_re.weight)

    # print(dir(logphi_model.model))

    # x, _ = logphi_model.symmetry(state0[0][None,...])
    # print(x)
    # # print(sort_by_label2d(x[None,...], x.shape[-2], x.shape[-1]))

    # x, _ = logphi_model.symmetry(state0r[0][None,...])
    # print(x)
    # print(sort_by_label2d(x[None,...], x.shape[-2], x.shape[-1]))

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
