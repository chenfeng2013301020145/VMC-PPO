#encoding: utf-8

from typing import Tuple
import numpy as np

def get_init_state(state_size, kind='rand', n_size=1):
    N = state_size[0]
    Dp = state_size[1]
    state = np.zeros([n_size, Dp, N])
    if kind == 'ferro':
        state[:,1,:] = 1

    if kind == 'rand':
        # np.random.seed(1234)
        for i in range(n_size):
            state_v = np.zeros(N, dtype=np.int8)
            index = np.random.choice(N, N//2, replace=False)
            state_v[index] = Dp - 1
            state[i,state_v,range(N)] = 1
    
    if kind == 'half_filling':
        for i in range(N//2):
            state[:,0, i] = 1
            state[:,-1, i + N//2] = 1

    return state, 0

class SUNSpin1D( ):

    def __init__(self, state_size, t: float, pbc: bool) -> None:
        """
        Initializes a SU(N) symmetric 1D spin chain Hamiltonian.

        H = t\sum_i P_{i, i+1}, where P is spin exchange operator.

        Args:
            t: The pre-factor of the Hamiltonian, if t is negative, the
                Hamiltonian is Ferromagnetic, if positive it is
                Anti-Ferromagnetic.
            pbc: True for periodic boundary condition.
        """
        self._pbc = pbc
        self._t = t
        self._update_size = state_size[0] + 1

    def find_states(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = state.shape[-1]
        Dp = state.shape[0]
        states = np.zeros([self._update_size, Dp, N])
        coeffs = np.zeros(self._update_size)
        n_sites = state.shape[-1]
        diag = 0.0
        for i in range(n_sites - 1):
            if np.sum(state[:,i] * state[:,i + 1]) != 1:
                temp = state.copy()

                # This is the correct way of swapping states when temp.ndim > 1.
                temp[:,[i, i + 1]] = temp[:,[i + 1, i]]
                states[i] = temp
                coeffs[i] = self._t
            else:
                diag += self._t
        if self._pbc:
            # if np.any(state[n_sites - 1] != state[0]):
            if np.sum(state[:,n_sites - 1] * state[:,0]) != 1:
                temp = state.copy()

                # This is the correct way of swapping states when temp.ndim > 1.
                temp[:,[n_sites - 1, 0]] = temp[:,[0, n_sites - 1]]
                states[i] = temp
                coeffs[i] = self._t
            else:
                diag += self._t
        # state[:,-1] = state[:,0]
        states[-1] = state.copy()
        coeffs[-1] = diag
        return states, coeffs


if __name__ == '__main__':
    from nqs_vmc_torch1d import _get_init_nqs
    state0, _ = _get_init_nqs(10,2,kind='rand')
    print(state0)
    Ham = SUNSpin1D(1,pbc=True)
    update_states, update_coeffs = Ham.find_states(np.squeeze(state0))
    print(update_coeffs)
    print(update_states)