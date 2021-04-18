# encoding: utf-8
"""Two dimensional XY model on triangle lattice."""

import numpy as np

def get_init_state(state_size, kind='rand', n_size=1):
    L = state_size[0]
    W = state_size[1]
    Dp = state_size[-1]
    state = np.zeros([n_size, Dp, L, W])

    if kind == 'half_filling':
        for i in range(n_size):
            state[i,0,np.arange(L//2),:] = 1
            state[i,-1,np.arange(L//2)+L//2, :] = 1

    if kind == 'rand':
        for i in range(n_size):
            state_v = np.zeros([L, W])
            pos = np.random.choice(L*W, L*W//2, replace=False)
            pos_x = pos // L
            pos_y = pos % L
            state_v[pos_x, pos_y] = 1
            state[i] = value2onehot(state_v, Dp)

    return state

def value2onehot(state, Dp):
    L = state.shape[0]
    W = state.shape[1]
    X, Y = np.meshgrid(range(L), range(W))
    state_onehot = np.zeros([Dp, L, W])
    state_onehot[state.astype(dtype=np.int8), Y, X] = 1
    return state_onehot

class XY2DTriangle():

    def __init__(self, state_size, pbc=True):
        """Initializes a 2D Heisenberg AFM Hamiltonian.

        H = \sum_<i,j> S^x_iS^x_j + S^y_iS^y_j
          = \sum_<i,j>1/2(S^+_iS^-_j + h.c)
        Args:
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        # self._nearest_neighbors = ((0, 1), (1, 0), (1, 1))
        self._nearest_neighbors = ((-1, 0), (0, 1), (1, 0))
        self._update_size = 3*state_size[0]*state_size[1]

    def find_states(self, state: np.ndarray):
        # shape: (Dp, L, W)
        L = state.shape[-2]
        W = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([self._update_size, Dp, L, W])
        coeffs = np.zeros(self._update_size)
        cnt = 0
        for r in range(L):
            for c in range(W):
                for dr, dc in self._nearest_neighbors:
                    rr, cc = r + dr, c + dc
                    if rr >= L or cc >= W:
                        if self._pbc:
                            rr %= L
                            cc %= W
                        else:
                            continue
                    if np.sum(state[:, r, c] * state[:, rr, cc]) != 1:
                        temp = state.copy()

                        # This is the correct way of swapping states when
                        # temp.ndim > 2.
                        temp[:, [r, rr], [c, cc]] = temp[:, [rr, r], [cc, c]]
                        states[cnt] = temp
                        coeffs[cnt] = 0.5
                    cnt +=1
        return np.stack(states), np.array(coeffs)

if __name__ == '__main__':
    state0 = get_init_state([4,4,2], kind='rand', n_size=10)
    print(state0[1])

    _ham = Heisenberg2DTriangle([4,4,2], pbc=True)
    ustates, ucoeffs = _ham.find_states(state0[1])
    print(ucoeffs)