module MinimaxAdaptiveControl

using LinearAlgebra

include("mat_comp.jl")
# Structs
struct Candidate{T<:Number}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    K::AbstractMatrix{T}
    P::AbstractMatrix{T}
    hist::Base.RefValue{<:Real}
    lam::T
end

struct MAController{T<:Number}
    candidates::AbstractArray{Candidate{T},1}
    xprev::AbstractArray{T,1}
    argmin::Base.RefValue{Int64}
end

# Constructors

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
        A = As[i]
        B = Bs[i]
        # Extend B to put on standard DARE form
        Bext = [B I(n)]
        P = dare(A, Bext, Matrix{Float64}(Q), Rext)
        Kext = (Rext + Bext'*P*Bext)\(Bext'*P*A)
        K = Kext[1:m, 1:n]
        candidates[i] = Candidate(A, B, K,P, Base.RefValue{Float64}(0),T(lam))
    end
    return MAController{T}(candidates, x0,Base.RefValue{Int64}(1))
end

# Controls

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

function K(mac::MAController)
    return mac.candidates[mac.argmin[]].K
end

# Analysis
"""
    Pcommon(mac::MAController) synthesizes a common P such that
    P >= Q + K'RK + (A-BK)(P^(-1) - 1/gamma^2*I)^(-1)*(A-BK) for
    all candidates (A,B,K).
"""
function Pcommon(mac::MAController{T}) where T<:Number
end
# TO DO: I'm uncertain about this one.
"""
    T(mac::MAController) synthesizes a common T using convex
    programming such that
    T >= Q + K_j'RK_j + (A_i-B_iK_j)(P^(-1) - 1/gamma^2*I)^(-1)*(A_i-B_iK_j) and
    T >= Q + K_k'RK_k ...
"""
function T(mac::MAController{T}) where T<:Number
end

export Candidate, MAController, update!, K
end