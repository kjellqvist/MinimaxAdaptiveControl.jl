# Abstract Types
#Abstract type for representing a plant in control systems. `T` is the type of the elements in the plant's matrices and vectors.
abstract type AbstractPlant{T} end

#Abstract type for representing a linear model in control systems. `T` is the type of the elements in the model's matrices.
abstract type AbstractLinMod{T} end

#Abstract type for representing a controller in control systems. `T` is the type of the elements in the controller's parameters.
abstract type AbstractController{T} end

#Abstract type for representing a performance metric in control systems. `T` is the type of the elements in the performance metric.
abstract type AbstractPerformanceMetric{T} end

# Struct Types
@doc raw"""
    SSPlant{T}

State-space representation of the plant:

```math
\begin{aligned}
        x_{t + 1} & = A x_t + B u_t + w_t, t \geq 0
\end{aligned}
```
where ``x_t`` is the state vector, ``u_t`` is the control input vector, and ``w_t`` is the disturbance vector.

# Fields
- `A::AbstractMatrix{T}`: State transition matrix.
- `B::AbstractMatrix{T}`: Control input matrix.
- `x::AbstractVector{T}`: State vector.
"""
struct SSPlant{T} <: AbstractPlant{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    x::AbstractVector{T}
end

@doc raw"""
    OFPlant{T}

Output-feedback representation of a plant of the form

```math
\begin{aligned}
		x_{t + 1} & = A x_t + B u_t + Gw_t,\quad  t \geq 0 \\
		y_t & = C x_t + D v_t.
\end{aligned}
```

# Fields
- `A::AbstractMatrix{T}`: State transition matrix.
- `B::AbstractMatrix{T}`: Control input matrix.
- `G::AbstractMatrix{T}`: Disturbance input matrix.
- `C::AbstractMatrix{T}`: Output matrix.
- `D::AbstractMatrix{T}`: Noise Feedforward matrix.
- `x::AbstractVector{T}`: State vector.
"""
struct OFPlant{T} <: AbstractPlant{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    G::AbstractMatrix{T}
    C::AbstractMatrix{T}
    D::AbstractMatrix{T}
    x::AbstractVector{T}
end

@doc raw"""
    SSLinMod{T}

State-space representation of the linear model

```math
        x_{t + 1} = A x_t + B u_t, t \geq 0
```
with cost function

```math
        J = \sum_{t = 0}^{\infty} \left( x_t^T Q x_t + u_t^T R u_t \right).
```

# Fields
- `A::AbstractMatrix{T}`: State transition matrix.
- `B::AbstractMatrix{T}`: Control input matrix.
- `Q::AbstractMatrix{T}`: State cost matrix.
- `R::AbstractMatrix{T}`: Control input cost matrix.
"""
struct SSLinMod{T} <: AbstractLinMod{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    Q::AbstractMatrix{T}
    R::AbstractMatrix{T}
end

@doc raw"""
    OFLinMod{T}

Output-feedback representation of the linear model

```math
\begin{aligned}
		x_{t + 1} & = A x_t + B u_t + Gw_t,\quad  t \geq 0 \\
		y_t & = C x_t + D v_t.
\end{aligned}
```
with cost function

```math
        J = \sum_{t = 0}^{\infty} \left( x_t^T Q x_t + u_t^T R u_t \right).
```

# Fields
- `A::AbstractMatrix{T}`: State transition matrix.
- `B::AbstractMatrix{T}`: Control input matrix.
- `G::AbstractMatrix{T}`: Disturbance input matrix.
- `C::AbstractMatrix{T}`: Output matrix.
- `D::AbstractMatrix{T}`: Feedforward matrix.
- `Q::AbstractMatrix{T}`: State cost matrix.
- `R::AbstractMatrix{T}`: Control input cost matrix.
"""
struct OFLinMod{T} <: AbstractLinMod{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    G::AbstractMatrix{T}
    C::AbstractMatrix{T}
    D::AbstractMatrix{T}
    Q::AbstractMatrix{T}
    R::AbstractMatrix{T}
end

@doc raw"""
    MAController{T}

Minimax Adaptive Switching Controller for the system
```math
\begin{aligned}
        z_{t + 1} & = \hat A z_t + \hat B u_t + \hat G d_t, t \geq 0 \\
        u_t & = K_t z_t,
\end{aligned}
```

with the uncertain objective
```math
J = \min_u \max_{d, N, H} \sum_{t = 0}^\infty \sigma_H(z_t, u_t, d_t)
```
where ``z_t`` is the state vector, ``u_t`` is the control input vector and ``d_t`` is measured disturbance vector.
The function $\sigma_H$ is a quadratic function of the form
```math
\begin{aligned}
        \sigma_H(z_t, u_t, d_t) = \begin{bmatrix} z_t \\ u_t \\ d_t \end{bmatrix}^T H \begin{bmatrix} z_t \\ u_t \\ d_t \end{bmatrix}.
\end{aligned}

`selectionRule` is a (possibly stateful) function that selects the control action based on $rs$ and the current state.

# Fields
- `z::AbstractVector{T}`: State vector of the controller.
- `Ahat::AbstractMatrix{T}`: State transition matrix.
- `Bhat::AbstractMatrix{T}`: Estimated control input matrix.
- `Ghat::AbstractMatrix{T}`: Estimated disturbance input matrix.
- `Ks::AbstractVector{Matrix{T}}`: Feedback gains, one for each mode.
- `Hs::AbstractVector{Matrix{T}}`: Quadratic cost matrices, one for each mode.
- `rs::AbstractVector{T}`: Worst-case historically incurred costs
- `selectionRule::Function`: Function for selecting the control action.
"""
struct MAController{T} <: AbstractController{T}
    z::AbstractVector{T}
    Ahat::AbstractMatrix{T}
    Bhat::AbstractMatrix{T}
    Ghat::AbstractMatrix{T}
    Ks::AbstractVector{Matrix{T}}
    Hs::AbstractVector{Matrix{T}}
    rs::AbstractVector{T}
    selectionRule::Function
end

"""
    SelfTuningRegulatorNMP{T}

Direct self-tuning regulator for non-minimum phase systems.

# Fields
- `d::Int`: Delay parameter.
- `k::Int`: Order of the control input polynomial.
- `l::Int`: Order of the output polynomial.
- `QCoeffs::AbstractMatrix{T}`: Coefficients of the Q polynomial.
- `PCoeffs::AbstractMatrix{T}`: Coefficients of the P polynomial.
- `us::AbstractVector{T}`: Past control inputs.
- `ufs::AbstractVector{T}`: Filtered past control inputs.
- `ys::AbstractVector{T}`: Past outputs.
- `yfs::AbstractVector{T}`: Filtered past outputs.
- `parameterEstimates::AbstractVector{T}`: Current parameter estimates.
- `regressors::AbstractMatrix{T}`: Regressor matrix for parameter estimation.
- `errorCovariance::AbstractMatrix{T}`: Covariance matrix of the parameter estimation error.
- `kalmanGain::AbstractMatrix{T}`: Kalman gain matrix.
"""
struct SelfTuningRegulatorNMP{T} <: AbstractController{T}
    d::Int
    k::Int
    l::Int
    QCoeffs::AbstractMatrix{T}
    PCoeffs::AbstractMatrix{T}
    us::AbstractVector{T} # u_{t} u_{t-1} ... u_{t-max(degQ, k)}
    ufs::AbstractVector{T} # P*/Q* us, ufs = [uf_t ... uf_{t - max(degP, k)}]'
    ys
end


"""
    InducedlpGain{T, N<:Integer}

Performance metric based on induced l_p gain.

# Fields
- `value::T`: Value of the induced l_p gain.
- `sumpOutput::T`: Sum of the p-norm of the output to the p-th power.
- `sumpInput::T`: Sum of the p-norm of the input to the p-th power.
- `p::N`: Order of the norm.
"""
struct InducedlpGain{T, N<:Integer} <: AbstractPerformanceMetric{T}
    value::T
    sumpOutput::T
    sumpInput::T
    p::N
end

"""
    ValueFunction{T}

Value function for an optimal control problem.

# Fields
- `value::T`: Value of the function.
- `Ps::AbstractVector{Matrix{T}}`: Sequence of cost-to-go matrices.
- `weights::AbstractVector{Vector{T}}`: Sequence of weights.
- `controller::MAController{T}`: Associated minimax adaptive controller.
"""
struct ValueFunction{T} <: AbstractPerformanceMetric{T}
    value::T
    Ps::AbstractVector{Matrix{T}}
    weights::AbstractVector{Vector{T}}
    controller::MAController{T}
end

"""
    InternalStates{T}

Performance metric based on internal states of a controller.

# Fields
- `value::AbstractVector{T}`: Vector of internal states.
- `controller::MAController{T}`: Associated model adaptive controller.
"""
struct InternalStates{T} <: AbstractPerformanceMetric{T}
    value::AbstractVector{T}
    controller::MAController{T}
end
