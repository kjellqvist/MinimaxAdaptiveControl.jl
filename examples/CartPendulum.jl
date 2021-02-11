
module CartPendulum

global const g = 9.82

using DifferentialEquations
using ControlSystems
mutable struct Pendulum{T<:Real}
    M::T
    m::T
    l::T
    b::T
    ulims::Tuple{T,T}
    nx::T
    nθ::T
    sgn::T
end

function Pendulum()
    Pendulum{Float64}(1., 0.2, 0.6, 0.1, (-10., 10.), 0.1, 0.01,1)
end

function I(pen::Pendulum{T}) where T<:Real
    return pen.m*pen.l^2/3
end

function linearize(pen::Pendulum{<:Real})
    (M, m, l, b) = (pen.M, pen.m, pen.l, pen.b)
    denom = I(pen) * (M + m) + M * m * l^2
    a22 = -(I(pen) + m * l^2) * b / denom
    a23 = m^2 * g * l^2 / denom;
    a42 = -m * l * b / denom;
    a43 = m * g * l * (M + m) / denom;
    b2 = (I(pen) + m * l^2) / denom;
    b4 = m * l / denom;

    A = [0 1 0 0 
    0 a22 a23 0
    0 0 0 1
    0 a42 a43 0]

    B = [0;b2;0;b4];
    # For now, assume full state feedback
    # C = diagm(ones(4));
    C = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    return ss(A, B, C, 0)
end

# Deterministic part
function f!(dx::AbstractVector{T}, 
    x::AbstractVector{T}, 
    pen::Pendulum{T}, t::T) where T<:Real
    (M, m, l, b, ulims) = (pen.M, pen.m, pen.l, pen.b, pen.ulims)

    lump1 = m^2 * l^2 / (I(pen) + m * l^2)
    c = cos(x[3])
    s = sin(x[3])

    dx[1] = x[2];
    dx[2] = (-b * x[2] + g * lump1 * c * s + m * l * s * x[4]^2 + pen.sgn*sat(x[5], ulims)) / (M + m - lump1 * c^2)
    dx[3] = x[4];
    dx[4] = (-m * l * c * dx[2] - m * g * l * s) / (I(pen) + m * l^2);
    dx[5:end] .= 0; # Controller state is set by callback and held constant otherwise
end

# Stochastic noise
function g!(dx::AbstractVector{T}, 
    x::AbstractVector{T}, 
    pen::Pendulum{T}, t::T) where T<:Real

    (M, m, l, b) = (pen.M, pen.m, pen.l, pen.b)
    (nx, nθ) = (pen.nx, pen.nθ)
    lump1 = m^2 * l^2 / (I(pen) + m * l^2)
    c = cos(x[3])
    s = sin(x[3])

    dx[1] = 0;
    dx[2] = pen.nx/ (M + m - lump1 * c^2)
    dx[3] = 0;
    dx[4] = nθ/(I(pen) + m * l^2)
    dx[5:end] .= 0
end

function sat(val::T, lims::Tuple{T,T}) where T<:Real
    return min(max(val, lims[1]), lims[2])
end
end