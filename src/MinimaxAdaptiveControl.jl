module MinimaxAdaptiveControl

using LinearAlgebra
using JuMP

include("mat_comp.jl")

"""
    Candidate{T<:Number}

# Fields:
`A::AbstractMatrix{T}` System matrix

`B::AbstractMatrix{T}` Input Gain matrix

`K::AbstractMatrix{T}` ``H_\\infty``` feedback gain

`P::AbstractMatrix{T}` Stationary solution to the Riccati equation

`hist::Base.RefValue{<:Real}` History, ``\\sum _{k=0}^N \\lambda^{N-k}|x_{k+1} - Ax_k -Bu_k|^2``

`lam::T` Forgetting factor ``\\lambda ``
"""
struct Candidate{T<:Number}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    K::AbstractMatrix{T}
    P::AbstractMatrix{T}
    hist::Base.RefValue{<:Real}
    lam::T
end

"""
    MAController{T<:Number}

# Fields
`candidates::AbstractArray{Candidate{T},1}` Set of system candidates

`xprev::AbstractArray{T,1}` Previous state

`argmin::Base.RefValue{Int64}` Index of minimizing controller
"""

struct MAController{T<:Number}
    candidates::AbstractArray{Candidate{T},1}
    xprev::AbstractArray{T,1}
    γ::Real
    Q::AbstractMatrix{T}
    R::AbstractMatrix{T}
    argmin::Base.RefValue{Int64}
end

# Constructors
"""
    mac = MAController(As, Bs, Q, R, γ, x0, lam)

Create a MAController (minimax adaptive controller) `mac::MAController{T}`
with matrices containing elements of type T, solving
\\min\\_u\\max\\_w,i\\sum (|x_t|^2\\_Q + |u_t|^2\\_R) - \\gamma^2 \\sum |w_t|^2
...
# Arguments
- `As::AbstractVector{<:AbstractMatrix{T}}` A vector of state matrices
- `Bs::AbstractVector{<:AbstractMatrix{T}}` A vector of input gains
- `Q::AbstractMatrix,`                      State penalty matrix |x|_Q^2
- `R::AbstractMatrix,`                      Control penalty matrix |u|_R^2
- `γ::Real,`                                Disturbance penalty, \\H_\\infty gain
- `x0::AbstractVector{<:Number}`            Initial state
- `lam::Real=1`                             Forgetting factor
...
"""
function MAController(
    As::AbstractVector{<:AbstractMatrix{T}}, Bs::AbstractVector{<:AbstractMatrix{T}},
    Q::AbstractMatrix, R::AbstractMatrix, γ::Real, 
    x0::AbstractVector{<:Number}, lam=1) where T<:Number
    N = length(As)

    @assert length(As) == length(Bs) "As and Bs" must be of the same length
    @assert length(As) > 0 "As and Bs cannot be nonempty"
    @assert γ > 0 "γ must be strictly positive"
    @assert issemiposdef(Q) "Q must be positive semidefinite"
    @assert isposdef(R) "R nust be positive definite"

    n = size(As[1])[1]  # Number of states
    m = size(Bs[1])[2]  # Number of inputs

    # Extend R to put on standard DARE form
    Rext = [R zeros(m,n)
            zeros(n,m) -γ^2*I(n)]
    
    candidates = Vector{Candidate{T}}(undef, N)
    for i = 1:N
        A = copy(As[i])
        B = copy(Bs[i])
        # Extend B to put on standard DARE form
        Bext = [B I(n)]
        P = dare(A, Bext, Matrix{Float64}(Q), Rext)
        Kext = (Rext + Bext'*P*Bext)\(Bext'*P*A)
        K = Kext[1:m, 1:n]
        candidates[i] = Candidate(A, B, K,P, Base.RefValue{Float64}(0),T(lam))
    end
    return MAController{T}(candidates,copy(x0),γ,Matrix{T}(Q),Matrix{T}(R), Base.RefValue{Int64}(1))
end

# Controls
"""
    update!(mac,x, u) 
    
Update the internal states of the controller based 
on current state `x` and control signal `u`.
...
# Arguments:
- `mac::MAController`       MinimaxAdaptiveControl controller object
- `x::AbstractArray{T,1}`   The current state
- `u::Abstractarray{T,1}`   The previous control signal
...

"""
function update!(mac::MAController, 
    x::AbstractArray{T,1}, u::AbstractArray{T,1}) where T<:Number
    lowestcost = Inf
    argmin = 1
    k = 1
    for c in mac.candidates
        c.hist[] += c.lam*(c.A*mac.xprev + c.B*u - x)'*(c.A*mac.xprev + c.B*u - x)
        if c.hist[] < lowestcost
            lowestcost = c.hist[]
            argmin = k
        end
        k +=1
    end
    mac.argmin[] = argmin
    mac.xprev[:] = x
end

"""
    K(mac::MAController)

Select the feedback gain such that u = -Kx

"""
function K(mac::MAController)
    return mac.candidates[mac.argmin[]].K
end

"""
    Vbar(mac, T, x)

Compute the current upper bound of the value function.

...
# Arguments
- `mac::MAController`: Controller object
- `T::AbstractMatrix`: T - matrix, can be synthesized using Tval(...)
- `x::AbstractArray`: Next state
...
"""
function Vbar(mac::MAController, T::AbstractMatrix, x::AbstractArray)
    N = length(mac.candidates)
    maxval = -Inf
    minhist = Inf
    secminhist = Inf
    for i = 1:N
        c = x'*mac.candidates[i].P*x -mac.γ^2*mac.candidates[i].hist[]
        maxval < c ? maxval = c : nothing
        hist = mac.candidates[i].hist[]
        hist < secminhist ? secminhist = hist : nothing
        hist <= minhist ? (minhist, secminhist) = (hist, minhist) : nothing
    end
    d = 
    return max(maxval, x'*T*x - (mac.γ^2)/2*(minhist + secminhist))
end

"""
    T(mac, model, tol = 0.01) 
synthesize a common T using convex programming such that
inequalities (19) and (20) are fulfilled.
...
# Arguments:
- `mac::MAController` MinimaxAdaptiveControl controller object
- `model::JuMP.Model` A user supplied JuMP model. Currently the solvers Mosek and Hypatia works.
- `tol::Real = 0.01`  Forcing (19) and (20) to hold with margin, i.e. T - tol*I >= ...
...
"""
function Tsyn(mac::MAController{M}, model::JuMP.Model, tol::Real = 0.01) where M<:Number
    N_systems = length(mac.candidates)
    n = size(mac.Q)[1]
    @variable(model, T[1:n, 1:n] in PSDCone())
    @variable(model,t)
    @constraint(model, Symmetric(mac.γ^2*I(n)-T) in PSDCone())
    for i = 1:N_systems
        @constraint(model, Symmetric(T - mac.candidates[i].P) in PSDCone())
        for j = 1:N_systems
            @constraint(model, X(mac, T, i,j, tol) in PSDCone())
            if i!=j
                for k = 1:N_systems
                    @constraint(model, Z(mac, T, i,j,k, tol) in PSDCone())
                end
            end
        end
    end

    @constraint(model, Symmetric(t*I(n) - T) in PSDCone()) # Regularize the operator norm of T
    @objective(model, Min, t)
    optimize!(model)
    T = value.(T)
    stat = termination_status(model)
    return T, stat
end

"""
    X(mac,T,i, k, tol) where P<:Number

Synthesize a symmetric matrix X such that inequality (19) is satisfied iff
X is positive semidefinite.
...
# Arguments:
- `mac::MAController`   MinimaxAdaptiveControl controller object
- `T::AbstractMatrix`   Either a matrix or a JuMP variable
- `i::Integer`          Index variable,
- `k::Integer`          Index Variable
- `tol::Real = 0.01`    Forcing (19) to hold with margin, i.e. T - tol*I >= ...
...
"""
function X(mac::MAController{P},
    T::AbstractMatrix, 
    i::Integer, k::Integer, tol::Real = 0.01) where P<:Number
    Q = mac.Q
    R = mac.R
    Kk = mac.candidates[k].K
    Ai = mac.candidates[i].A
    Bi = mac.candidates[i].B
    Pi = mac.candidates[i].P

    Aik = Ai - Bi*Kk

    Xik = [(T - Q - Kk'*R*Kk)-tol*I(size(Pi)[1]) Aik'
            Aik (inv(Pi) - mac.γ^(-2)*I(size(Pi)[1]))]
    return Xik
end

"""
    Z(mac, T, i, j, k)

Synthesize a symmetric matrix Z such that inequality (20) is satisfied iff
Z is positive semidefinite.

...
# Arguments:
- `mac::MAController`   MinimaxAdaptiveControl controller object
- `T::AbstractMatrix`   Either a matrix or a JuMP variable
- `i::Integer`          Index variable
- `j::Integer`          Index variable
- `k::Integer`          Index Variable
- `tol::Real = 0.01`    Forcing (20) to hold with margin, i.e. T - tol*I >= ...
...
"""
function Z(mac::MAController{P},
    T::AbstractMatrix, 
    i::Integer, j::Integer, k::Integer, tol::Real = 0.01) where P<:Number

    Q = mac.Q
    R = mac.R
    γ = mac.γ
    Kk = mac.candidates[k].K
    Ai = mac.candidates[i].A
    Bi = mac.candidates[i].B
    Pi = mac.candidates[i].P

    Aj = mac.candidates[j].A
    Bj = mac.candidates[j].B
    Pj = mac.candidates[j].P

    Aik = Ai - Bi*Kk
    Ajk = Aj - Bj*Kk

    Zijk11 = T - Q - Kk'*R*Kk + γ^2/2*(Aik'*Aik + Ajk'*Ajk) -tol*I(size(Pi)[1])
    Zijk21 = (Aik + Ajk)
    Zijk22 = 4/γ^4*(γ^2*I(size(Pi)[1]) - T)
    Zijk = [Zijk11 Zijk21'
            Zijk21 Zijk22]
    return Zijk
end

export Candidate, MAController, update!, K, Tsyn, Vbar
end