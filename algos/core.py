# encoding: utf-8

import torch.nn as nn
import torch
import numpy as np
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

# Real CNN 
#--------------------------------------------------------------------
class CNN_layer(nn.Module):
    def __init__(self,K,F_in,F_out,act=nn.ReLU,pbc=True,dimensions='1d'):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN_layer,self).__init__()
        self._pbc = pbc
        self.dimensions = dimensions
        self.K = [K,K] if type(K) is int else K
        if dimensions == '1d':
            self.conv = nn.Sequential(nn.Conv1d(F_in,F_out,self.K,1,0),act())
        else:
            self.conv = nn.Sequential(nn.Conv2d(F_in,F_out,self.K,[1,1],0),act())
            
    def forward(self,x):
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions=self.dimensions)
        x = self.conv(x)
        return x

class OutPut_layer(nn.Module):
    def __init__(self,K,F,output_size, pbc=True, dimensions='1d'):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut_layer,self).__init__()
        self._pbc = pbc
        self.output_size = output_size
        self.dimensions = dimensions
        self.K = [K,K] if type(K) is int else K
        self.linear = nn.Linear(F,output_size, bias=False)
            
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions=self.dimensions)
        x = x.sum(2) if self.dimensions == '1d' else x.sum(dim=[2,3])
        return self.linear(x)

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
            z = torch.log(1. + torch.exp(z))
            return torch.stack((z.real, z.imag), dim=1)
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
def translation_phase(x, k, dimensions):
    xdevice = x.device
    n_sample = x.shape[0]
    z = x[:,0] + 1j*x[:,1]
    z = torch.exp(z)
    real = z.real
    imag = z.imag
    if dimensions == '1d':
        N = x.shape[-1]
        vec = torch.exp(1j*2*np.pi*k[0]*torch.arange(N, device=xdevice)/N)
        vec_real = vec.real.repeat(n_sample, 1)
        vec_imag = vec.imag.repeat(n_sample, 1)
        real_part = real*vec_real - imag*vec_imag
        imag_part = real*vec_imag + imag*vec_real
    else:
        L = x.shape[-2]
        W = x.shape[-1]
        mat = (torch.exp(1j*2*np.pi*k[1]*torch.arange(W, device=xdevice)/W)
            *torch.exp(1j*2*np.pi*k[0]*torch.arange(L, device=xdevice)/L)[...,None])
        mat_real = mat.real.repeat(n_sample, 1, 1)
        mat_imag = mat.imag.repeat(n_sample, 1, 1)
        real_part = real*mat_real - imag*mat_imag
        imag_part = real*mat_imag + imag*mat_real
    return torch.stack((real_part, imag_part), dim=1)

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
        # complex_lncosh = ComplexLnCosh()
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
        # self.linear = ComplexLinear(F,1, bias=False)
    
    def forward(self,x):
        # shape of complex x: (batch_size, 2, F, N) or (batch_size, 2, F, L, W)
        # variance scaling 
        norm = np.sqrt(np.prod(x.shape[2:]))
        x = x.sum(dim=2)/norm
        x = translation_phase(x, k=self.momentum, dimensions=self.dimensions)
        x = x.sum(2) if self.dimensions=='1d' else x.sum(dim=[2,3])
        z = x[:,0] + 1j*x[:,1]
        z = torch.log(z)
        return torch.stack((z.real, z.imag), dim=1)
    
#--------------------------------------------------------------------
def mlp_cnn(state_size, K, F=[4,3,2], stride=[1], output_size=1, output_activation=False, act=nn.ReLU,
        complex_nn=False, inverse_sym=False, relu_type='selu', pbc=True, bias=True, momentum=[1,0]):
    K = K[0] if type(K) is list and len(K) == 1 else K
    stride = stride[0] if type(stride) is list and len(stride) == 1 else stride
    dim = len(state_size) - 1
    dimensions = '1d' if dim == 1 else '2d'
    layers = len(F)
    name_index = 0
    if complex_nn:
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
    else:
        Dp = state_size[-1]

        input_layer = CNN_layer(K=K, F_in=Dp, F_out=F[0], act=act, pbc=pbc, dimensions=dimensions)
        output_layer = OutPut_layer(K,F[-1],output_size,pbc=pbc, dimensions=dimensions)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN_layer(K=K, F_in=F[i-1], F_out=F[i], dimensions=dimensions,
                                act=act, pbc=pbc) for i in range(1,layers)]
        cnn_layers += [output_layer]

        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.zeros_(m.bias)

        model = nn.Sequential(*cnn_layers)
        model.apply(weight_init)
    return model, name_index

# physical symmetry 
# -------------------------------------------------------------------------------------------------
def translation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1:
        N = x.shape[-1]
        Dp = x.shape[1]
        x = x.reshape(x.shape[0],1, Dp, N)
        x_roll_total = x.clone()
        for i in range(N - 1):
            x_roll_single = torch.roll(x, shifts=i+1, dims=3)
            x_roll_total = torch.cat((x_roll_total, x_roll_single), dim=1)
        # shape of x_roll: (batch_size, N, Dp, N)
        x = x_roll_total.reshape(-1, Dp, N)
        return x, N 
    else:
        L = x.shape[-2]
        W = x.shape[-1]
        Dp = x.shape[1]
        x = x.reshape(x.shape[0],1, Dp, L, W)
        l, w = np.meshgrid(range(L), range(W))
        l = l.reshape(-1)[1:]
        w = w.reshape(-1)[1:]
        x_roll_total = x.clone()
        for dx, dy in zip(l, w):
            x_roll_single = torch.roll(x, shifts=[dx, dy], dims=[3, 4])
            x_roll_total = torch.cat((x_roll_total, x_roll_single), dim=1)
        # shape of x_roll: (batch_size, L*W, Dp, L, W)
        x = x_roll_total.reshape(-1, Dp, L, W)
        return x, L*W 
    
def inverse(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    N = 2
    x_inverse = torch.flip(x, dims=[1])
    return torch.stack((x, x_inverse), dim=1).reshape([-1] + list(x.shape[1:])), N

def identity(x):
    return x, 1

def reflection(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    if dimension == 1:
        N = 2
        x_flr = torch.flip(x, dims=[2])
        return torch.stack((x, x_flr), dim=1).reshape([-1] + list(x.shape[1:])), N
    else:
        N = 4
        x_flr = torch.flip(x, dims=[2])
        x_fud = torch.flip(x, dims=[3])
        x_flrud = torch.flip(x, dims=[2,3])
        return torch.stack((x, x_flr, x_fud, x_flrud), dim=1).reshape([-1] + list(x.shape[1:])), N

def c2rotation(x):
    # input shape of x: (batch_size, Dp, N) or (batch_size, Dp, L, W)
    dimension = len(x.shape) - 2
    N = 2
    if dimension == 1:
        x180 = torch.rot90(x, 2, dims=[2])
    else:
        x180 = torch.rot90(x, 2, dims=[2, 3])
    return torch.stack((x, x180), dim=1).reshape([-1] + list(x.shape[1:])), N

def transpose(x):
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 2
        xT = torch.transpose(x, 2, 3)
        return torch.stack((x, xT), dim=1).reshape([-1] + list(x.shape[1:])), N
    
def transpose2(x):
    dimension = len(x.shape) - 2
    if dimension == 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError('Only 2D square-shape lattice can be transposed.')
    else:
        N = 3
        xT = torch.transpose(x, 2, 3)
        xT2 = torch.rot90(torch.transpose(torch.rot90(x, 1, dims=[2,3]), 2, 3), -1, dims=[2,3])
        return torch.stack((x, xT, xT2), dim=1).reshape([-1] + list(x.shape[1:])), N
    
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
        return torch.stack((x, x90, x180, x270), dim=1).reshape([-1] + list(x.shape[1:])), N
    
class sym_model(nn.Module):
    def __init__(self, state_size, K, F=[4,3,2], stride=[1], output_size=1, 
                output_activation=False, act=nn.ReLU, complex_nn=False, relu_type='selu', 
                pbc=True, bias=True, momentum=[0,0], sym_funcs=[identity]):
        super(sym_model,self).__init__()
        self.model, _ = mlp_cnn(state_size=state_size, K=K, F=F, stride=stride, output_size=output_size, 
                    output_activation=output_activation, act=act, complex_nn=complex_nn, relu_type=relu_type, 
                    pbc=pbc, bias=bias, momentum=momentum)
        self.sym_funcs = sym_funcs
    
    def symmetry(self, x):
        N = 1
        for func in self.sym_funcs:
            x, n = func(x)
            N *= n
        return x, N       
    
    # apply symmetry
    def forward(self, x):
        sym_x, N = self.symmetry(x)
        sym_x = self.model(sym_x)
        sym_x = sym_x.reshape(x.shape[0], N, 2)
        z = sym_x[:,:,0] + 1j*sym_x[:,:,1]
        z = torch.exp(z).sum(dim=1)
        z = torch.log(z)
        return torch.stack((z.real, z.imag),dim=1)

def mlp_cnn_sym(state_size, K, F=[4,3,2], stride=[1], output_size=1, output_activation=False, act=nn.ReLU,
        complex_nn=False, relu_type='selu', pbc=True, bias=True, momentum=[0,0], sym_funcs=[identity]):
    model = sym_model(state_size=state_size, K=K, F=F, stride=stride, output_size=output_size, 
                    output_activation=output_activation, act=act, complex_nn=complex_nn, relu_type=relu_type, 
                    pbc=pbc, bias=bias, momentum=momentum, sym_funcs=sym_funcs)
    name_index = 1
    return model, name_index

    
# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    seed = 286
    torch.manual_seed(seed)
    np.random.seed(seed)
    logphi_model, _ = mlp_cnn_sym([4,4,2], 2, [2,2], stride=[1], complex_nn=True,
                           output_size=2, relu_type='selu', bias=True, momentum=[0,0], sym_funcs=[c4rotation, identity])
    #op_model = mlp_cnn([10,10,2], 2, [2],complex_nn=True, output_size=2, relu_type='sReLU', bias=True)
    # print(logphi_model)
    print(get_paras_number(logphi_model))
    import sys
    sys.path.append('..')
    from ops.HS_spin2d import get_init_state
    state0,_ = get_init_state([3,3,2], kind='rand', n_size=500)
    state_zero = torch.from_numpy(state0[0][None,...])
    state_zero = torch.stack((state_zero, torch.zeros_like(state_zero)), dim=1)
    state_t0 = torch.rot90(torch.from_numpy(state0[0][None,...]).float(),3, dims=[2,3])
    print(state_t0)
    # print(complex_periodic_padding(state_zero, [3,3], [1,1], dimensions='2d'))
    #print(state0.shape)
    print(logphi_model(state_t0))
    # print(logphi_model(torch.from_numpy(state0).float())[:3])
    state_t = torch.roll(state_t0, shifts=1, dims=2)
    # state_t = torch.rot90(torch.from_numpy(state0[0][None,...]).float(),2, dims=[2,3])
    print(state_t)
    print(logphi_model(state_t))
    logphi_model_sym, _ = mlp_cnn_sym([4,4,2], 2, [2,2], stride=[1], complex_nn=True,
                           output_size=2, relu_type='selu', bias=True, momentum=[0,0], sym_funcs=[identity])
    logphi_model_sym.load_state_dict(logphi_model.state_dict())
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
