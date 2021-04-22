# Variational Monte Carlo with Proximal Policy Optimization
[toc]

## Requirement
    pyTorch >= 1.8.0 + cu111 

## Neural-network quantum state (NQS)
    algos.core
Two types of convolution neural network are implemented as NQS.
**Inputs**: spin configurations.
**Outputs**: $\log|\Psi|$ and $\theta$

* pesudocomplex CNN with only real parameters.
![avatar](pesudo-complex-CNN.png)
* complex CNN contains two sublayers in each layer, one stands for real part and the other denotes the imag part of the complex layer.
![avatar](complex-CNN.png)

## Optimization: Natural gradient descent
Optimization object,
$$
{\rm minimize}_{w} E_{w}(\Psi(s;w))~~~{\rm s.t.}~ D^2_{FS}(\Psi_{\rm old}(s;w_{\rm old}), \Psi(s;w)) \leq \delta.
$$

where $E_w$ is the energy estimated via importance sampling and $D_{FS}$ is the Fubini-Study distance.

### PPO algorthim
    algos.pesudocomplex_ppo or algos.complex_ppo
NGD can be approximately solved by PPO-clip and PPO-clip updates wavefunctions via,

$w_{k+1} = {\rm argmin}_{w} \mathbb{E} [L(w_{k}, w)]$

with the loss function,
$$
L(w_k,w) = \max\Big(\frac{|\Psi_w|^2}{|\Psi_{w_{k}}|^2}E_{w~{\rm or}~w_{k}}, {\rm clip}\Big( \frac{|\Psi_w|^2}{|\Psi_{w_{k}}|^2}, 1-\epsilon, 1+\epsilon \Big)E_{w~{\rm or}~w_{k}} \Big),
$$

where $E_w$  is estimated by the current wavefunction $\Psi_w$  and $E_{w_k}$  is estimated by the old wavefunction $\Psi_{w_k}$ .


### Stochastic reconfiguration (TDVP)
    algos.pesudocomplex_sr or algos.complex_sr
The optimization object of NGD is also equivalent to,
$$
{\rm minimize}_{\Delta w} \big\{ E_w + \nabla_wE_w\Delta w \big\}~~{\rm s.t.}~~\frac{1}{2}\Delta w^{\dagger}{\bf S}\Delta \omega < \delta,
$$
due to,
$D^2_{FS} \approx \sum_{ij}dw_i^*dw_j[\langle \mathcal{O}_i^*\mathcal{O}_j \rangle - \langle \mathcal{O}_i^* \rangle \langle \mathcal{O}_j \rangle],$

where 
$$
S_{ij} = [\langle \mathcal{O}_i^*\mathcal{O}_j \rangle - \langle \mathcal{O}_i^* \rangle \langle \mathcal{O}_j \rangle]
$$ 
is the stochastic reconfiguration matrix and $\mathcal{O}_i = \partial_w\log\Psi$.
Such a contitional minimal problem is hence equivalent to:
$$
{\rm minimize}_{\Delta w} \{ E_w + \nabla_wE_w + \lambda(\frac{1}{2}\Delta w^{\dagger}{\bf S}\Delta w - \epsilon)\},
$$
with a Lagrange multiplier $\lambda$. Its minimal satisfies,

${\bf S}\Delta w = -\alpha\nabla_wE_w,$
which is the exact TDVP equation.


## Reference
