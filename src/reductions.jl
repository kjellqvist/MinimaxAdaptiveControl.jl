@doc raw"""
    reduceSys(sys::Vector{SSLinMod{T}}, γ::T, models::AbstractVector{GenericModel{T}}) where T<: Real

The function computes the aggregated model matrices (`Ahat`, `Bhat`, `Ghat`) and the gain (`Ks`) and auxiliary (`Hs`) matrices for each of the input models.

Reduces state-feedback adaptive control with uncertain parameters
```math
    \begin{aligned}
        x_{t+1} & = Ax_t + Bu_t + w_t \\
        (A, B) & \in \{(A_1, B_1), \ldots, (A_M, B_M)\}
    \end{aligned}
```
with the soft-constrained objective function
```math
    \min_u\max_{w, N, A, B}\sum_{t=0}^N\left( |x_t|^2_Q + |u_t|^2_R - \gamma^2 |w_t|^2 \right),
```
to a certain system
```math
    \begin{aligned}
        z_{t + 1} & = \hat Az_t + \hat Bu_t + \hat G d_t
    \end{aligned}
```
with uncertain objective
```math
\min_u \max_{H, d, N} \sum_{t = 0}^N \sigma_H(z_t, u_t, d_t)
```
# Arguments
- `sys::Vector{SSLinMod{T}}`: Vector of state-space linear models to be reduced.
- `γ::T`: Induced ``\ell_2``-gain.
- `models::AbstractVector{GenericModel{T}}`: Vector of JuMP models, one per state-space model.

# Returns
- `Ahat::Matrix{T}`: Aggregated state transition matrix.
- `Bhat::Matrix{T}`: Aggregated control input matrix.
- `Ghat::Matrix{T}`: Aggregated disturbance input matrix.
- `Ks::Vector{Matrix{T}}`: Vector of ``\gamma``-suboptimal ``\mathcal H_\infty`` gain matrices.
- `Hs::Vector{Matrix{T}}`: Vector of cost weights.


    reduceSys(sys::Vector{OFLinMod{T}}, γ::T, models::AbstractVector{GenericModel{T}}, method::Symbol = :LMI2) where T <: Real

Reduces a set of output-feedback linear models into a single aggregated model using a specified method.
The function computes the aggregated model matrices (`Ahat`, `Bhat`, `Ghat`) and the gain (`Ks`) and auxiliary (`Hs`) matrices for each of the input models.

Takes uncertain systems of the form
```math
\begin{aligned}
		x_{t + 1} & = A x_t + B u_t + Gw_t,\quad  t \geq 0 \\
		y_t & = C x_t + D v_t \\
                (A, B, C, D, G) & \in \mathcal M
\end{aligned}
```
with soft-constrained objective function

```math
    \min_u\max_{w, v, x_0 N, M \in \mathcal M}\sum_{t=0}^N\left( |x_t|^2_Q + |u_t|^2_R - \gamma^2 |(w_t, v_t)|^2 \right) - |x_0 - \hat x_0|^2_{S_{M, 0}},
```
to a certain system
```math
    \begin{aligned}
        z_{t + 1} & = \hat Az_t + \hat Bu_t + \hat G d_t
    \end{aligned}
```
with uncertain objective
```math
\min_u \max_{H, d, N} \sum_{t = 0}^N \sigma_H(z_t, u_t, d_t)
```

# Arguments
- `sys::Vector{OFLinMod{T}}`: Vector of output-feedback linear models to be reduced.
- `γ::T`: Induced ``\ell_2``-gain.
- `models::AbstractVector{GenericModel{T}}`: Vector of JuMP models, one for each ouput-feedback model
- `method::Symbol`: Method to be used for the reduction (default is `:LMI2`). See [`fared`](@ref)

# Returns
- `A::Matrix{T}`: Aggregated observer state transition matrix.
- `B::Matrix{T}`: Aggregated observer control input matrix.
- `G::Matrix{T}`: Aggregated observer measurement input matrix.
- `Ks::Vector{Matrix{T}}`: Vector of ``\gamma``-suboptimal ``\mathcal H_\infty`` observer gain matrices.
- `Hs::Vector{Matrix{T}}`: Vector of cost weights.

"""
function reduceSys(sys::Vector{SSLinMod{T}}, γ::T, models::AbstractVector{GenericModel{T}}) where T<: Real
    N = length(sys)
    nx = size(sys[1].A, 1)
    nu = size(sys[1].B, 2)
    Ahat = zeros(nx, nx)
    Bhat = zeros(nx, nu)
    Ghat = Matrix{T}(I(nx))

    N = length(sys)
    Hs = [ [ss.Q zeros(nx, nu) zeros(nx, nx); zeros(nu, nx) ss.R zeros(nu, nx); zeros(nx, nx) zeros(nx, nu) zeros(nx, nx)] - γ^2 *[-ss.A -ss.B Matrix(I(nx))]' * [-ss.A -ss.B Matrix(I(nx))] for ss in sys]
    Ks = [bared(Ahat, Bhat, Ghat, H, models[i])[2] for (i, H) in enumerate(Hs)]
    return (Ahat, Bhat, Ghat, Ks, Hs)
end

function reduceSys(sys::Vector{OFLinMod{T}}, γ::T, models::AbstractVector{GenericModel{T}}, method::Symbol = :LMI2) where T <: Real
    N = length(sys)
    nxs = [size(sys[i].A, 1) for i in 1:N]  # System orders
    nu = size(sys[1].B, 2)  # Number of inputs
    ny = size(sys[1].C, 1)  # Number of outputs

    Ahat = zeros(sum(nxs), sum(nxs))
    Bhat = zeros(sum(nxs), nu)
    Ghat = zeros(sum(nxs), ny)



    N = length(sys)
    nxs = [size(sys[i].A, 1) for i in 1:N]  # System orders
    nu = size(sys[1].B, 2)  # Number of inputs
    ny = size(sys[1].C, 1)  # Number of outputs

    A = zeros(sum(nxs), sum(nxs))
    B = zeros(sum(nxs), nu)
    G = zeros(sum(nxs), ny)
    Ks = Vector{Matrix{Float64}}(undef, N)
    Hs = Vector{Matrix{Float64}}(undef, N)

    cummulativeNxs = 0
    for i = 1:N
        Ks[i] = zeros(nu, sum(nxs))
        Hs[i] = zeros(sum(nxs) + nu + ny, sum(nxs) + nu + ny)
        (_, Ahat, Ghat, H, _) = fared(sys[i], γ, models[i], method = :LMI2)
        A[cummulativeNxs+1:cummulativeNxs+nxs[i], cummulativeNxs+1:cummulativeNxs+nxs[i]] .= Ahat
        B[cummulativeNxs+1:cummulativeNxs+nxs[i], :] .= sys[i].B
        G[cummulativeNxs+1:cummulativeNxs+nxs[i], :] .= Ghat
        (_, K, _) = bared(Ahat, sys[i].B, Ghat, H, models[i])
        Ks[i][:, cummulativeNxs+1:cummulativeNxs+nxs[i]] .= K

        Hxx = H[1:nxs[i], 1:nxs[i]]
        Hxu = H[1:nxs[i], nxs[i]+1:nxs[i]+nu]
        Hxy = H[1:nxs[i], nxs[i]+nu+1:nxs[i]+nu+ny]
        Huu = H[nxs[i]+1:nxs[i]+nu, nxs[i]+1:nxs[i]+nu]
        Huy = H[nxs[i]+1:nxs[i]+nu, nxs[i]+nu+1:nxs[i]+nu+ny]
        Hyy = H[nxs[i]+nu+1:nxs[i]+nu+ny, nxs[i]+nu+1:nxs[i]+nu+ny]

        Hs[i][cummulativeNxs+1:cummulativeNxs+nxs[i], cummulativeNxs+1:cummulativeNxs+nxs[i]] .= Hxx
        Hs[i][cummulativeNxs+1:cummulativeNxs+nxs[i], sum(nxs)+1:sum(nxs)+nu] .= Hxu
        Hs[i][cummulativeNxs+1:cummulativeNxs+nxs[i], sum(nxs)+nu+1:sum(nxs)+nu+ny] .= Hxy
        Hs[i][sum(nxs)+1:sum(nxs)+nu, cummulativeNxs+1:cummulativeNxs+nxs[i]] .= Hxu'
        Hs[i][sum(nxs)+1:sum(nxs)+nu, sum(nxs)+1:sum(nxs)+nu] .= Huu
        Hs[i][sum(nxs)+1:sum(nxs)+nu, sum(nxs)+nu+1:sum(nxs)+nu+ny] .= Huy

        Hs[i][sum(nxs)+nu+1:sum(nxs)+nu+ny, cummulativeNxs+1:cummulativeNxs+nxs[i]] .= Hxy'
        Hs[i][sum(nxs)+nu+1:sum(nxs)+nu+ny, sum(nxs)+1:sum(nxs)+nu] .= Huy'
        Hs[i][sum(nxs)+nu+1:sum(nxs)+nu+ny, sum(nxs)+nu+1:sum(nxs)+nu+ny] .= Hyy
        cummulativeNxs += nxs[i]
    end
    return (A, B, G, Ks, Hs)
end

