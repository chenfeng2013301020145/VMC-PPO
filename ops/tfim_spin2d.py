# encoding: utf-8

import numpy as np 

def get_init_state(state_size, kind='rand', n_size=1):
    L = state_size[0]
    W = state_size[1]
    Dp = state_size[-1]
    state = np.zeros([n_size, Dp, L, W])

    if kind == 'ferro':
        for i in range(n_size):
            pos = np.random.randint(0, Dp)
            state[i,pos,:,:] = 1

    if kind == 'rand':
        state_v = np.random.randint(0, Dp, size=(n_size, L, W))
        X, Y = np.meshgrid(range(W), range(L))
        for i in range(n_size):
            state[i,state_v[i],Y,X] = 1

    return state, None

def onehot2value(state, Dp): 
    state_v = np.arange(0,Dp).reshape(Dp,1,1)*np.squeeze(state)
    return np.sum(state_v,0).astype(dtype=np.int8)

def value2onehot(state, Dp):
    L = state.shape[0]
    W = state.shape[1]
    X, Y = np.meshgrid(range(W), range(L))
    state_onehot = np.zeros([Dp, L, W])
    state_onehot[state.astype(dtype=np.int8), Y, X] = 1
    return state_onehot
    
class TFIMSpin2D():

    def __init__(self, g: float, state_size, pbc=True):
        """
        Tranverse Ising model in 2 dimension:
        Hamiltonian: 
            H = - sum_ij{sigma^z_ij * sigma^z_{i + 1}j + sigma^z_ij * sigma^z_i{j + 1}} 
                + g*sum_ij{sigma^x_ij}

        Args:
            g: Strength of the transverse field.
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._g = g
        self._nearest_neighbors = ((0, 1), (1, 0))
        self._update_size = state_size[0]*state_size[1] + 1

    def find_states(self, state: np.ndarray):
        L = state.shape[-2]
        W = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([self._update_size, Dp, L, W])
        coeffs = np.zeros(self._update_size)

        # off-diagnal
        cnt = 0
        diag = 0
        for r in range(L):
            for c in range(W):
                temp = state.copy()
                temp[0,r,c], temp[1,r,c] = state[1,r,c], state[0,r,c]
                states[cnt] = temp
                coeffs[cnt] = 1/2*self._g
                cnt += 1
                
                for dr, dc in self._nearest_neighbors:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        diag += 0.25
                    else:
                        diag -= 0.25
                                  
        states[cnt] = state
        coeffs[cnt] = diag

        return states, coeffs, cnt+1
    
class TFIMJ1J2_Spin2D():

    def __init__(self, state_size, g=0,j1=0,j2=0, pbc=True):
        """
        Tranverse Ising model in 2 dimension:
        Hamiltonian: 
            H = - j2*sum_<<ij>>(S^z_i*S^z_j) 
                - j1*sum_<ij>(S^z_i*S^z_j)
                + g*sum_ij{sigma^x_ij}

        Args:
            g: Strength of the transverse field.
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._g = g
        self._j1 = j1
        self._j2 = j2
        self._nearest_neighbors_j1 = ((0, 1), (1, 0))
        self._nearest_neighbors_j2 = ((1, 1), (-1,1))
        self._update_size = state_size[0]*state_size[1] + 1

    def find_states(self, state: np.ndarray):
        L = state.shape[-2]
        W = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([self._update_size, Dp, L, W])
        coeffs = np.zeros(self._update_size)

        # off-diagnal
        cnt = 0
        diag = 0.0
        for r in range(L):
            for c in range(W):
                temp = state.copy()
                temp[0,r,c], temp[1,r,c] = state[1,r,c], state[0,r,c]
                states[cnt] = temp
                coeffs[cnt] = 1/2*self._g
                cnt += 1
                
                for dr, dc in self._nearest_neighbors_j1:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        diag -= 0.25*self._j1
                    else:
                        diag += 0.25*self._j1
                        
                for dr, dc in self._nearest_neighbors_j2:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        diag -= 0.25*self._j2
                    else:
                        diag += 0.25*self._j2
                                  
        states[cnt] = state
        coeffs[cnt] = diag

        return states, coeffs, cnt+1
    
class ISINGJ1J2_Spin2D():

    def __init__(self, state_size, g=0,j1=0,j2=0, pbc=True):
        """
        Tranverse Ising model in 2 dimension:
        Hamiltonian: 
            H = - j2*sum_<<ij>>(S^z_i*S^z_j) 
                - j1*sum_<ij>(S^z_i*S^z_j)
                + g*sum_ij{sigma^x_ij}

        Args:
            g: Strength of the transverse field.
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._g = g
        self._j1 = j1
        self._j2 = j2
        self._nearest_neighbors_j1 = ((0, 1), (1, 0))
        self._nearest_neighbors_j2 = ((1, 1), (-1,1))
        self._update_size = 1

    def find_states(self, state: np.ndarray):
        L = state.shape[-2]
        W = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([self._update_size, Dp, L, W])
        coeffs = np.zeros(self._update_size)

        # off-diagnal
        cnt = 0
        diag = 0.0
        for r in range(L):
            for c in range(W):               
                for dr, dc in self._nearest_neighbors_j1:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        diag -= 0.25*self._j1
                    else:
                        diag += 0.25*self._j1
                        
                for dr, dc in self._nearest_neighbors_j2:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        diag -= 0.25*self._j2
                    else:
                        diag += 0.25*self._j2
                                  
        states[cnt] = state
        coeffs[cnt] = diag

        return states, coeffs, cnt+1

if __name__=='__main__':
    from tfim_spin2d import get_init_state

    state_size = [10,10,2]
    state0 = get_init_state(state_size, kind='rand',n_size=10)
    print(state0[0])
    # print(np.sum(state0[0],0))
    '''
    state_v = onehot2value(state0[0],Dp=2)
    print(state_v)
    
    ham = TFIMSpin2D(1, pbc=True)
    states, coeffs = ham.find_states(np.squeeze(state0[2]))
    print(states)
    print(coeffs)
    '''
    