# encoding: utf-8
# compatible with 2d system

import numpy as np

class updator():
    def __init__(self, state_size):
        if len(state_size) - 1 == 1:
            self._N = state_size[0]
        else:
            self._N = state_size[0]*state_size[1]
        self._Dp = state_size[-1]

    def generate_mask(self, n_sample):
        rang = range(n_sample)
        swaps = np.random.randint(0, self._N, (2, n_sample))
        masks = np.arange(self._N)[None, :].repeat(n_sample, axis=0)
        masks[rang, swaps[0]], masks[rang, swaps[1]] = (
            masks[rang, swaps[1]], masks[rang, swaps[0]])
        # print(np.random.randint(0,10))
        return masks

    def generate_state_mask(self, state):
        full_index = np.arange(1, self._N+1)
        state = state.reshape(self._Dp, self._N)

        state_index_list = []
        for d in range(self._Dp):
            state_index_list.append(np.unique(state[d]*full_index)[1:])

        swap_dp = np.random.choice(self._Dp, size=2, replace=False)

        first_index = int(np.random.choice(state_index_list[swap_dp[0]], size=1) - 1)
        second_index = int(np.random.choice(state_index_list[swap_dp[1]], size=1) - 1)

        mask = np.arange(self._N)
        mask[first_index], mask[second_index] = mask[second_index], mask[first_index]
        return mask

    def _get_update(self, state, mask):
        temp = state.reshape(self._Dp, self._N).T
        state_f = temp[mask]
        # self._state = np.concatenate((self._state, self._state[0,:].reshape(1,self._Dp)),0)
        return state_f.T.reshape(state.shape)

