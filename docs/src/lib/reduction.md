```@index
Pages = ["reduction.md"]
```

# Reduction-related functions

## Functions
```@docs
bared
fared
reduceSys
```

## Details on the forward algebraic riccati equation
Given matrices ``A \in \mathbb{R}^{n_x \times n_x}``, ``C \in \mathbb{R}^{n_x \times n_y}``, ``D \in \mathbb{R}^{n_y \times n_v}``, ``G \in \mathbb{R}^{n_x \times n_w}``, ``Q \in \mathbb{R}^{n_x \times n_x}``, ``R \in \mathbb{R}^{n_u \times n_u}``, and ``\gamma > 0`` the discrete-time forward algebraic Riccati equation (FARED) is defined as.

```math
\begin{aligned}
    S & = (AX^{-1}A^\top + \gamma^{-2}GG^\top)^{-1} \\
    X & = S + \gamma^2 C^\top (DD^\top)^{-1}C - Q \\
    L & = \gamma^2 AX^{-1}C(DD^\top)^{-1}.
\end{aligned}
```

The FARED may have multiple solutions. The function `fared` computes the solution associated with the maximal ``S`` (over the positive semidefinite cone).
The function also returns a matrix `H` which is given by
```math
\begin{bmatrix}
    SX^{-1}S -S & 0 & \gamma^2 SX^{-1}C(DD^\top)^{-1} \\
    0 & R & 0 \\
    \gamma^2 C^\top X^{-1}S(DD^\top)^{-1} & 0 & -\left((DD^\top)^{-1} + C(S - Q)^{-1}C^\top\right)^{-1}
    \end{bmatrix},
```
The observer matrices are constructed by ``\hat A = A X^{-1} S`` and ``\hat G = \gamma^2 A X^{-1} C (DD^\top)^{-1}``.

### fared :LMI1
Solves (if possible) the SDP
```math
\begin{aligned}
    \text{maximize} & \quad \text{trace}(S) \\
    \text{subject to} & \quad \begin{bmatrix}
        S & SA & SG \\
        A^\top S & X & 0_{n_x \times n_w} \\
        G^\top S & 0_{n_w \times n_x} & \gamma^2 I_{n_w}
    \end{bmatrix} \succeq 0 \\
    & \quad X = S + \gamma^2 C^\top (DD^\top)^{-1}C - Q,
\end{aligned}
```
where ``S`` is the decision variable. The function returns the optimal ``S`` and the corresponding ``X``, ``L``, and ``H``.

### fared :LMI2
Directly constructs an SDP whose solution gives the optimal ``(S, \hat A, \hat G, H)``. 
Let 
```math
Y = \begin{bmatrix}
        I_{n_x} & 0 \\
        0 & C^\top \\
        0 & 0 \\
        0 & D^\top
    \end{bmatrix}
```

The SDP is given by
```math
\begin{aligned}
    \underset{S, \hat A, \hat B, H}{\text{minimize}} & \quad \text{trace}(H) \\
    \text{subject to} & \quad \begin{bmatrix}
        S & -S & 0 & 0 & -\hat A^\top \\
        -S & S - Q & 0 & 0 & A^\top S - C^\top \hat G \\
        0 & 0 & \gamma^2 I_{n_w} & 0 & G^\top S \\
        0 & 0 & 0 & \gamma^2 I_{n_v} & -\hat G^\top \\
        -\hat A & A^\top S & G & -G & -\hat G D S
    \end{bmatrix} -
    \begin{bmatrix}
        Y H Y^\top & 0 \\
        0 & 0
    \end{bmatrix} \succeq 0\\
    & S \succeq 0, \\
    & H - H^\top = 0.
    \end{aligned}
```


### fared :Iterate
This method solves the FARED by iterating starting with ``S_0 = \gamma^2 I_{n_x}``. The iteration is given by
```math
\begin{aligned}
    S_{t + 1}& = (AX_t^{-1}A^\top + \gamma^{-2}GG^\top)^{-1} \\
    X_t & = S_t + \gamma^2 C^\top (DD^\top)^{-1}C - Q
\end{aligned}
```

### fared :Laub
Uses [MatrixEquations.jl](https://github.com/andreasvarga/MatrixEquations.jl/) function `ared` to solve the FARED.
`ared` implements W. F. Arnold and A. J. Laub, "Generalized eigenproblem algorithms and software for algebraic Riccati equations," in Proceedings of the IEEE, vol. 72, no. 12, pp. 1746-1754, Dec. 1984, doi: 10.1109/PROC.1984.13083.

## Details on the backward algebraic riccati equation
Fix ``\gamma > 0`` and consider the transformation
```math
	\begin{aligned}
		G(\gamma) & = \gamma\hat G\sqrt{-H^{-dd}}, \qquad
		R = H^{uu} - H^{ud}H^{-dd}H^{du}, \\
Q & = H^{dd} - H^{zd}H^{-dd}H^{dz}\\
  & \qquad -(H^{zu} - H^{zd}H^{-dd} H^{du})R^{-1}(*), \\
	A & = \hat A - BR^{-1}(H^{zu} - H^{zd}H^{-dd}H^{du}) \\
		  & \quad - GH^{-dd}\left(H^{dz} - H^{du}R^{-1}\left(H^{uz} - H^{ud}H^{-dd}H^{dz}\right)\right)\\
		B & = \hat B - G(H^{dd})^{-1}H^{du}.
	\end{aligned}
```
As we are interested in the upper value, ``d_t`` is allowed to depend causally on ``z`` and ``u``, but ``u`` is required to depend strictly causally on ``d``.
The following parameterization of ``d_t`` and ``u_t`` respect the causalilty structure of the problem,
```math
\begin{aligned}
	d_t & = -H^{-dd}(H^{dz}z_t + H^{du}u_t) + \gamma\sqrt{-H^{-dd}}\delta_t, \quad t \geq 0 \\
u_t & = -R^{-1}(H^{uz} - H^{ud}H^{-dd}H^{dz})z_t + v_t. \quad t \geq 0, 
\end{aligned}
```
and leads to the follow equivalent dynamic game.
Compute
```math
	\inf_{\nu}\sup_{\delta, N} \sum_{t=0}^{N-1} \left( |z_t|^2_{Q} + |v_t|^2_{R} - \gamma^2|\delta_t|^2\right),
```
subject to the dynamics
```math
\begin{aligned}
	z_{t + 1} & = Az_t + Bv_t + G(\gamma)\delta_t, \quad t \geq 0, \\
	v_t & = \nu_t(z_0, \ldots, z_t), \quad t \geq 0.
\end{aligned}
```
This reformulation puts the problem on standard $\gamma$-suboptimal $\Hinf$ form, and the value function, if bounded, is well known to be a quadratic function of the initial state, and the optimal controller is of the form ``v_t = -K_\dagger z_t``.

The value function, ``V_\star(z_0)``, has the form ``V_\star(z_0) = |z_0|^2_{P_\star}``, where ``P_\star`` is the minimal fixed point of the generalized algebraic riccati equation. 
``P_\star`` and ``K_\dagger`` and can be computed, for example, through value iteration, Arnold and Laub's Schur methods and convex optimization.
