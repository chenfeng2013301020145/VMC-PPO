# encoding: utf-8

import numpy as np 

def get_init_state(state_size,kind='rand',n_size=1):
    N = state_size[0]
    Dp = state_size[1]
    state = np.zeros([n_size, Dp, N])
    if kind == 'ferro':
        for i in range(n_size):
            pos = np.random.randint(0, Dp)
            state[i,pos,:] = 1

    if kind == 'rand':
        state_v = np.random.randint(0, Dp, size=(n_size, N))
        for i in range(n_size):
            state[i,state_v[i],range(N)] = 1
    
    if kind == 'half_filling':
        for i in range(N//2):
            state[:,0, i] = 1
            state[:,-1, i + N//2] = 1

    return state, None

def onehot2value(state, Dp): 
    state_v = np.arange(0,Dp).reshape(Dp,1)*np.squeeze(state)
    return np.sum(state_v,0).astype(dtype=np.int8)
    
class TFIMSpin1D():

    def __init__(self, g: float, state_size, pbc=True):
        """
        Tranverse Ising model in 1 dimension:
        Hamiltonian: 
            H = - sum_i{sigma^z_i * sigma^z_{i + 1}} + g*sum_i{sigma^x_i}

        Args:
            g: Strength of the transverse field.
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._g = g
        self._update_size = state_size[0] + 1

    def find_states(self, state: np.ndarray):
        N = state.shape[-1]
        Dp = state.shape[-2]
        states = np.zeros([N+1, Dp, N])
        coeffs = np.zeros(N+1)

        # off-diagnal
        for i in range(N):
            temp = state.copy()
            temp[0,i], temp[1,i] = state[1,i], state[0,i]
            states[i,:,:] = temp
            coeffs[i] = 1/2*self._g

        # diagnal 
        state_v = onehot2value(state, Dp) - 1/2
        state_l = np.concatenate((state_v, state_v[0].reshape(1,)), axis=0)
        state_r = np.concatenate((state_v[-1].reshape(1,), state_v), axis=0)
        if self._pbc:   
            diag = - np.sum(state_l[1:]*state_r[1:])
        else:
            diag = - np.sum(state_l[1:-1]*state_r[1:-1])
        states[-1,:,:] = state
        coeffs[-1] = diag

        return states, coeffs

if __name__=='__main__':
    from tfim_spin1d import get_init_state

    state_size = [10,2]
    state0 = get_init_state(state_size, kind='ferro',n_size=10)
    print(state0)

    ham = TFIMSpin1D(1, pbc=True)
    states, coeffs = ham.find_states(np.squeeze(state0[2]))
    print(states)
    print(coeffs)