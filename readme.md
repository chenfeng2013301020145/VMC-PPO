# Variational Monte Carlo with Proximal Policy Optimization
[toc]

## Requirement
    pyTorch >= 1.8.0 + cu111 

## Neural-network quantum state (NQS)
    algos.core
Two types of convolution neural network are implemented as NQS.
**Inputs**: spin configurations.
**Outputs**: <img src="http://latex.codecogs.com/gif.latex?\log{|\Psi|~{\rm and}~\theta}">

* pesudocomplex CNN with only real parameters.
![avatar](pesudo-complex-CNN.png)
* complex CNN contains two sublayers in each layer, one stands for real part and the other denotes the imag part of the complex layer.
![avatar](complex-CNN.png)

## Optimization: Natural gradient descent
Optimization object,

<div align=center><img src="http://latex.codecogs.com/gif.latex?{\rm minimize}_{w} E_{w}(\Psi(s;w))~~~{\rm s.t.} D^2_{FS}(\Psi_{\rm old}(s;w_{\rm old}), \Psi(s;w)) \leq \delta."/></div>

where <img src="http://latex.codecogs.com/gif.latex?E_w"> is the energy estimated via importance sampling and <img src="http://latex.codecogs.com/gif.latex?D_{FS}"> is the Fubini-Study distance.

### PPO algorthim 
    algos.pesudocomplex_ppo or algos.complex_ppo
NGD can be approximately solved by PPO-clip and PPO-clip updates wavefunctions via,

<div align=center><img src="http://latex.codecogs.com/gif.latex?w_{k+1} = {\rm argmin}_{w} \mathbb{E} [L(w_{k}, w)]"></div>

with the loss function,
<div align=center><img src="http://latex.codecogs.com/gif.latex?L(w_k,w) = \max\Big(\frac{|\Psi_w|^2}{|\Psi_{w_{k}}|^2}E_{w~{\rm or}~w_{k}}, {\rm clip}\Big( \frac{|\Psi_w|^2}{|\Psi_{w_{k}}|^2}, 1-\epsilon, 1+\epsilon \Big)E_{w~{\rm or}~w_{k}} \Big),"></div>

where <img src="http://latex.codecogs.com/gif.latex?E_w">  is estimated by the current wavefunction <img src="http://latex.codecogs.com/gif.latex?\Psi_w">  and <img src="http://latex.codecogs.com/gif.latex?E_{w_k}">  is estimated by the old wavefunction <img src="http://latex.codecogs.com/gif.latex?\Psi_{w_k}"> .


### Stochastic reconfiguration (TDVP)
    algos.pesudocomplex_sr or algos.complex_sr
The optimization object of NGD is also equivalent to,
<div align=center><img src="http://latex.codecogs.com/gif.latex?{\rm minimize}_{\Delta w} \big\{ E_w + \nabla_wE_w\Delta w \big\}~~{\rm s.t.}~~\frac{1}{2}\Delta w^{\dagger}{\bf S}\Delta \omega < \delta,"></div>
due to,
<div align=center><img src="http://latex.codecogs.com/gif.latex?D^2_{FS} \approx \sum_{ij}dw_i^*dw_j[\langle \mathcal{O}_i^*\mathcal{O}_j \rangle - \langle \mathcal{O}_i^* \rangle \langle \mathcal{O}_j \rangle],"></div>

where <img src="http://latex.codecogs.com/gif.latex?S_{ij} = [\langle \mathcal{O}_i^*\mathcal{O}_j \rangle - \langle \mathcal{O}_i^* \rangle \langle \mathcal{O}_j \rangle]"> is the stochastic reconfiguration matrix and <img src="http://latex.codecogs.com/gif.latex?\mathcal{O}_i = \partial_w\log\Psi">.
Such a contitional minimal problem is hence equivalent to:
<div align=center><img src="http://latex.codecogs.com/gif.latex?{\rm minimize}_{\Delta w} \{ E_w + \nabla_wE_w + \lambda(\frac{1}{2}\Delta w^{\dagger}{\bf S}\Delta w - \epsilon)\},"></div>
with a Lagrange multiplier <img src="http://latex.codecogs.com/gif.latex?\lambda">. Its minimal satisfies,

<div align=center><img src="http://latex.codecogs.com/gif.latex?{\bf S}\Delta w = -\alpha\nabla_wE_w,"></div>
which is the exact TDVP equation.


