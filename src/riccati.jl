@doc raw"""
    fared(sys::OFLinMod{T}, γ::Real, model::GenericModel{T}; method::Symbol = :LMI2) where T <: Real
    fared(sys::OFLinMod{T}, γ::Real; method::Symbol = :Iterate, iters::Int = 1000) where T <: Real
    fared(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}, G::AbstractMatrix{T}, Q::AbstractMatrix{T}, R::AbstractMatrix{T}, γ::Real, model::GenericModel{T}; method = :LMI2) where T <: Real
    fared(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}, G::AbstractMatrix{T}, Q::AbstractMatrix{T}, R::AbstractMatrix{T}, γ::Real; method = :Iterate, iters = 1000) where T <: Real

Solves the forward algebraic Riccati equation for an output-feedback linear model (or the given system matrices) using the specified method.

# Arguments
- `sys::OFLinMod{T}`: Output-feedback linear model.
- `A::AbstractMatrix{T}`: State transition matrix.
- `B::AbstractMatrix{T}`: Control input matrix.
- `C::AbstractMatrix{T}`: Output matrix.
- `D::AbstractMatrix{T}`: Measurement noise matrix.
- `G::AbstractMatrix{T}`: Disturbance input matrix.
- `Q::AbstractMatrix{T}`: State cost matrix.
- `R::AbstractMatrix{T}`: Control input cost matrix.
- `γ::Real`: Induced $\ell_2$-gain
- `model::GenericModel{T}`: Generic optimization model.
- `method::Symbol`: Method to be used for solving the Riccati equation (default is `:LMI2` or `:Iterate` depending on the function signature). The options are `:LMI1`, `:LMI2`, `:Iterate`, and `:Laub`. See [Details on the forward algebraic riccati equation](@ref) for more information.
- `iters::Int`: Number of iterations for the iterative method (default is 1000).

# Returns
- A tuple containing:
  - `S::Matrix{T}`: Forward riccati solution matrix, positive definite.
  - `Ahat::Matrix{T}`: Observer state transition matrix.
  - `Ghat::Matrix{T}`: Observer output injection matrix.
  - `H::Matrix{T}`: Cost matrix.
  - `termination_status::Symbol`: Status of the optimization.
"""
function fared(sys::OFLinMod{T},
        γ::Real,
        model::GenericModel{T};
        method::Symbol = :LMI2,
    ) where T <: Real
    return fared(sys.A, sys.B, sys.C, sys.D, sys.G, sys.Q, sys.R, γ, model, method = method)
end

function fared(sys::OFLinMod{T},
        γ::Real;
        method::Symbol = :Iterate,
        iters::Int = 1000
    ) where T <: Real
    if method == :Iterate
        return fared(sys.A, sys.B, sys.C, sys.D, sys.G, sys.Q, sys.R, γ, method = method, iters = iters)
    elseif method ==:Laub
        return fared(sys.A, sys.B, sys.C, sys.D, sys.G, sys.Q, sys.R, γ, method = method)
    end
end

function fared(A::AbstractMatrix{T},
        B::AbstractMatrix{T},
      C::AbstractMatrix{T},
      D::AbstractMatrix{T},
      G::AbstractMatrix{T},
      Q::AbstractMatrix{T},
      R::AbstractMatrix{T},
      γ::Real,
      model::GenericModel{T};
      method = :LMI2,
     ) where T <: Real
    
    if method == :LMI1
        return fared_LMI1_(A, B, C, D, G, Q, R, γ, model)
    elseif method ==:LMI2
        return fared_LMI2_(A, B, C, D, G, Q, R, γ, model)
    end
end

function fared(A::AbstractMatrix{T},
        B::AbstractMatrix{T},
      C::AbstractMatrix{T},
      D::AbstractMatrix{T},
      G::AbstractMatrix{T},
      Q::AbstractMatrix{T},
      R::AbstractMatrix{T},
      γ::Real;
      method = :Iterate,
      iters = 1000,
     ) where T <: Real 
    
    if method == :Iterate
        return fared_iterate_(A, B, C, D, G, Q, R, γ, iters)
    elseif method ==:Laub
        return fared_laub_(A, B, C, D, G, Q, R, γ)
    end
end

"""
    bared(Ahat::AbstractMatrix{T}, Bhat::AbstractMatrix{T}, Ghat::AbstractMatrix{T}, H::AbstractMatrix{T}, model::GenericModel{T}) where T <: Real
    bared(Ahat::AbstractMatrix{T}, Bhat::AbstractMatrix{T}, Ghat::AbstractMatrix{T}, H::AbstractMatrix{T}; method::Symbol = :Laub, iters::Int = 1000) where T <: Real

Solves the backward algebraic Riccati equation using the specified method.

# Arguments
- `Ahat::AbstractMatrix{T}`: State transition matrix.
- `Bhat::AbstractMatrix{T}`: Control input matrix.
- `Ghat::AbstractMatrix{T}`: Disturbance input matrix.
- `H::AbstractMatrix{T}`: Cost matrix, symmetric.
- `model::GenericModel{T}`: [JuMP](https://jump.dev/JuMP.jl/stable/) optimization model.
- `method::Symbol`: Method to be used for solving the Riccati equation (default is `:Laub`). The available options are `:Laub` and `:Iterate`, see [Details on the backward algebraic riccati equation](@ref) for more information.
- `iters::Int`: Number of iterations for the iterative method (default is 1000).

# Returns
- A tuple containing:
  - `P::Matrix{T}`: Backward riccati solution matrix, positive definite.
  - `K::Matrix{T}`: Gain matrix.
  - `termination_status::Symbol`: Status of the optimization.
"""
function bared(Ahat::AbstractMatrix{T},
        Bhat::AbstractMatrix{T},
        Ghat::AbstractMatrix{T},
        H::AbstractMatrix{T},
        model::GenericModel{T};
    ) where T <: Real
    return bared_LMI_(Ahat, Bhat, Ghat, H, model)
end

function bared(Ahat::AbstractMatrix{T},
        Bhat::AbstractMatrix{T},
        Ghat::AbstractMatrix{T},
        H::AbstractMatrix{T};
        method::Symbol = :Laub,
        iters::Int = 1000
    ) where T <: Real
    if method == :Laub
        return bared_Laub_(Ahat, Bhat, Ghat, H)
    elseif method == :Iterate
        return bared_iterate_(Ahat, Bhat, Ghat, H, iters)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

# Forward riccati methods

function fared_LMI1_(A::AbstractMatrix{T},
        B::AbstractMatrix{T},
      C::AbstractMatrix{T},
      D::AbstractMatrix{T},
      G::AbstractMatrix{T},
      Q::AbstractMatrix{T},
      R::AbstractMatrix{T},
      γ::Real,
      model::GenericModel{T};
      ) where T <: Real 
    assert_dimensions_(A, B, C, D, G, Q, R)
    (nx, nu, ny, nw, nv) = get_dims_(A, B, C, D, G, Q)
    
    @variable(model, S[1:nx, 1:nx], PSD)
    X = Symmetric(S + γ^2 * C' / (D * D') * C - Q)
    @constraint(model, S - Q in PSDCone())
    M = Symmetric([
        S S * A S * G
        A' * S X zeros(nx, nw)
        G' * S zeros(nw, nx) γ^2 * I
    ])

    @constraint(model, M in PSDCone())
    @objective(model, Max, tr(S))
    optimize!(model)

    S = value.(S)
    X = value.(X)
    Ahat = A / value.(X) * S
    Ghat = γ^2 * A / value.(X) * C' / (D * D')

    
    Hxx = S / X * S - S
    Hxu = zeros(nx, nu)
    Huu = R
    Hxy = γ^2 * S / X * C' / (D * D')
    Huy = zeros(nu, ny)
    Hyy = -inv(D * D' / γ^2 + C /(S - Q) * C')
    H = Symmetric([Hxx Hxu Hxy; Hxu' Huu Huy; Hxy' Huy' Hyy])

    return (S, Ahat, Ghat, H, termination_status(model))

end

function fared_LMI2_(A::AbstractMatrix{T},
        B::AbstractMatrix{T},
      C::AbstractMatrix{T},
      D::AbstractMatrix{T},
      G::AbstractMatrix{T},
      Q::AbstractMatrix{T},
      R::AbstractMatrix{T},
      γ::Real,
      model::GenericModel{T}
      ) where T <: Real
    assert_dimensions_(A, B, C, D, G, Q, R)
    (nx, nu, ny, nw, nv) = get_dims_(A, B, C, D, G, Q)
 
    @variable(model, S[1:nx, 1:nx], PSD)
    @variable(model, H[1:(nx + ny), 1:(nx + ny)], Symmetric)
    @variable(model, Ahat[1:nx, 1:nx])
    @variable(model, Ghat[1:nx, 1:ny])
    M1 = Symmetric([
                S -S zeros(nx, nw) zeros(nx, nv) -Ahat'
                -S S-Q zeros(nx, nw) zeros(nx, nv) A' * S - C' * Ghat'
                zeros(nw, 2 * nx) γ^2 * I(nw) zeros(nw, nv) G' * S
                zeros(nv, 2 * nx + nw) γ^2 * I(nv) (-D' * Ghat')
                -Ahat S * A - Ghat * C G (-Ghat) * D S
               ])
    Y = [
         I(nx) zeros(nx, ny)
         zeros(nx, nx) C'
         zeros(nw, nx + ny)
         zeros(nv, nx) D'
        ]
    M2 = Symmetric([
        Y * H * Y' zeros(2 * nx + nw + nv, nx)
        zeros(nx, 2 * nx + nw + nv) zeros(nx, nx)
       ])

    @constraint(model, M1 + M2  in PSDCone())
    @objective(model, Min, tr(H))
    optimize!(model)

    S = value.(S)
    Ahat = S \ value.(Ahat)
    Ghat = S \ value.(Ghat)
    H = value.(H)
    Hfull = [
             H[1:nx, 1:nx] zeros(nx, nu) H[1:nx, (nx + 1):(nx + ny)]
             zeros(nu, nx) R zeros(nu, ny)
             H[(nx + 1):(nx + ny), 1:nx] zeros(ny, nu) H[(nx + 1):(nx + ny), (nx + 1):(nx + ny)]
            ]
    return (S, Ahat, Ghat, Hfull, termination_status(model))
end

function fared_iterate_(A::AbstractMatrix{T},
        B::AbstractMatrix{T},
      C::AbstractMatrix{T},
      D::AbstractMatrix{T},
      G::AbstractMatrix{T},
      Q::AbstractMatrix{T},
      R::AbstractMatrix{T},
      γ::Real,
      iters::Int
      ) where T <: Real
    assert_dimensions_(A, B, C, D, G, Q, R)
    (nx, nu, ny, nw, nv) = get_dims_(A, B, C, D, G, Q)
    
    S = γ^2 * I
    X = 0
    for i in 1:iters
    X = Symmetric(S + γ^2 * C' / (D * D') * C - Q)
        S = inv(Symmetric(A / X * A' + γ^(-2) * G * G'))
    end
    Ahat = A / X * S
    Ghat = γ^2 * A / X * C' / (D * D')
    Hxx = S / X * S - S
    Hxu = zeros(nx, nu)
    Huu = R
    Hxy = γ^2 * S / X * C' / (D * D')
    Huy = zeros(nu, ny)
    Hyy = -inv(D * D' / γ^2 + C /(S - Q) * C')
    H = Symmetric([Hxx Hxu Hxy; Hxu' Huu Huy; Hxy' Huy' Hyy])
    return (S, Ahat, Ghat, H, nothing)
end

function fared_laub_(A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    C::AbstractMatrix{T},
    D::AbstractMatrix{T},
    G::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    R::AbstractMatrix{T},
    γ::Real
      ) where T <: Real
    assert_dimensions_(A, B, C, D, G, Q, R)
    (nx, nu, ny, nw, nv) = get_dims_(A, B, C, D, G, Q)
    Bb = [C; I]'
    Rr = [D * D' / γ^2 zeros(ny, nx); zeros(nx, ny) -Q]
    Qq =G * G' / γ^2
    Aa = A'

    Xlaub, CLSEIG, Flaub = ared(Aa, Bb, Rr, Qq)
    S = inv(Xlaub)
    X = Symmetric(S + γ^2 * C' / (D * D') * C - Q)
    Ahat = A / X * S
    Ghat = Flaub'[:, 1:ny]

    Hxx = S / X * S - S
    Hxu = zeros(nx, nu)
    Huu = R
    Hxy = γ^2 * S / X * C' / (D * D')
    Huy = zeros(nu, ny)
    Hyy = -inv(D * D' / γ^2 + C /(S - Q) * C')
    H = Symmetric([Hxx Hxu Hxy; Hxu' Huu Huy; Hxy' Huy' Hyy])
    return (S, Ahat, Ghat, H, nothing)
end

# Backward riccati methods

function bared_LMI_(Ahat::AbstractMatrix{T},
        Bhat::AbstractMatrix{T},
        Ghat::AbstractMatrix{T},
        H::AbstractMatrix{T},
        model::GenericModel{T}
        ) where T <: Real

    nx = size(Ahat, 1) 
    nu = size(Bhat, 2)
    ny = size(Ghat, 2)

    (schurH, Ky) = lowerSchur(H, nx + nu)
    Ahat2 = Ahat - Ghat * Ky[:, 1:nx]
    Bhat2 = Bhat - Ghat * Ky[:, (nx + 1):(nx + nu)]
    Hyy = H[nx + nu + 1:end, nx + nu + 1:end]

    @variable(model, X[1:nx, 1:nx], PSD)
    @variable(model, Y[1:nu, 1:nx])
    M2 = [
      X zeros(nx, ny) X * Ahat2' - Y' * Bhat2' [X Y'] * sqrt(schurH)
      zeros(ny, nx) -Hyy Ghat' zeros(ny, nx + nu)
      (Ahat2 * X - Bhat2 * Y) Ghat X zeros(nx, nx + nu)
      sqrt(schurH) * [X;Y] zeros(nx + nu, nx + ny) I(nx + nu)
       ]
    @constraint(model, M2 in PSDCone())
    @objective(model, Max, tr(X))
    optimize!(model)
    Plmi =inv(value.(X))
    Klmi = value.(Y) * Plmi
    return (Plmi, Klmi, termination_status(model))
end

function bared_Laub_(Ahat::AbstractMatrix{T},
        Bhat::AbstractMatrix{T},
        Ghat::AbstractMatrix{T},
        H::AbstractMatrix{T},
        ) where T <: Real
    nx = size(Ahat, 1) 
    nu = size(Bhat, 2)
    ny = size(Ghat, 2)

    (schurH, Ky) = lowerSchur(H, nx + nu)
    Ahat2 = Ahat - Ghat * Ky[:, 1:nx]
    Bhat2 = Bhat - Ghat * Ky[:, (nx + 1):(nx + nu)]
    
    Qhat = schurH[1:nx, 1:nx]
    Rhat = [schurH[(nx + 1):end, (nx + 1):end] zeros(nu, ny); zeros(ny, nu) H[nx + nu + 1:end, nx + nu + 1:end]]
    Shat = [schurH[1:nx, nx + 1:nx + nu] zeros(nx, ny)]


    X, CLSEIG, F = ared(Ahat2, [Bhat2 Ghat], Rhat, Qhat, Shat) # This works, yaay.
    K = F[1:nu, :]
    P = X
    return (P, K, JuMP.MOI.OPTIMAL)
end

function bared_iterate_(Ahat::AbstractMatrix{T},
        Bhat::AbstractMatrix{T},
        Ghat::AbstractMatrix{T},
        H::AbstractMatrix{T},
        iters::Integer
        ) where T <: Real

    nx = size(Ahat, 1) 
    nu = size(Bhat, 2)
    ny = size(Ghat, 2)


    Pric = zeros(nx, nx) 
    Kric = zeros(nu, nx)
    Mk = zeros(nx + nu + ny, nx + nu + ny)
    Nk = zeros(nx + nu, nx + nu)
    status = JuMP.MOI.OPTIMAL
    for k = 1:iters
        Mk .= [Ahat Bhat Ghat]' * Pric * [Ahat Bhat Ghat] + H
        if maximum(eigvals(Mk[nx + nu + 1:end, nx + nu + 1:end])) >= 0
            status = JUMP.MOI.INFEASIBLE
            break
        end
        Nk .= lowerSchur(Mk, nx + nu)[1]
        if minimum(eigvals(Nk[nx + 1:end, nx + 1:end])) <= 0
            status = JUMP.MOI.INFEASIBLE
            break
        end
        Pric .= lowerSchur(Nk, nx)[1]
        Kric .=lowerSchur(Nk, nx)[2]
    end

    return (Pric, Kric, status)
end

# Helper functions
function assert_dimensions_(A, B, C, D, G, Q, R)
    @argcheck size(A, 1) == size(A, 2) DimensionMismatch("A must be square")
    @argcheck size(Q, 1) == size(Q, 2) DimensionMismatch("Q must be square")
    @argcheck size(R, 1) == size(R, 2) DimensionMismatch("R must be square")
    @argcheck size(A, 1) == size(B, 1) DimensionMismatch("A and B must have the same number of rows")
    @argcheck size(B, 2) == size(R, 2) DimensionMismatch("B and R must have the same number of columns")
    @argcheck size(A, 2) == size(C, 2) DimensionMismatch("A and C must have the same number of columns")
    @argcheck size(A, 1) == size(G, 1) DimensionMismatch("A and D must have the same number of rows")
    @argcheck size(A, 1) == size(Q, 1) DimensionMismatch("A and Q must have the same number of rows")
    @argcheck size(C, 1) == size(D, 1) DimensionMismatch("C and D must have the same number of rows")
end

function get_dims_(A, B, C, D, G, Q)
    nx = size(A, 1)
    nu = size(B, 2)
    ny = size(C, 1)
    nw = size(G, 2)
    nv = size(D, 1)
    return nx, nu, ny, nw, nv
end

# Function to promote an arbitrary number of arrays to common type
function promote_array(arrays...)
  eltype = Base.promote_eltype(arrays...)
  tuple([convert(Array{eltype}, array) for array in arrays]...)
end


function lowerSchur(A::AbstractMatrix{T}, ind::Integer) where T
    @argcheck size(A,1) == size(A,2) DimensionMismatch("A must be square")
    @argcheck ind <= size(A, 1) DimensionMismatch("ind must be less than the number of rows of A")
    A11 = A[1:ind, 1:ind]
    A12 = A[1:ind, ind+1:end]
    A21 = A[ind+1:end, 1:ind]
    A22 = A[ind+1:end, ind+1:end]
    return (A11 - A12 / A22 * A21, A22 \ A21)
end
