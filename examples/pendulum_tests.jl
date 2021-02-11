print("Importing packages")
include("CartPendulum.jl")
import .CartPendulum
include("../src/MinimaxAdaptiveControl.jl")
using .MinimaxAdaptiveControl
using Test
using ControlSystems
using LinearAlgebra
using DifferentialEquations
using Plots

function f!(dx::AbstractVector{T}, 
    x::AbstractVector{T}, 
    p::Tuple{CartPendulum.Pendulum{T}, MAController}, t::T) where T<:Real
    return CartPendulum.f!(dx, x, p[1], t)
end

function g!(dx::AbstractVector{T}, 
    x::AbstractVector{T}, 
    p::Tuple{CartPendulum.Pendulum{T}, MAController}, t::T) where T<:Real
    return CartPendulum.g!(dx, x, p[1], t)
end

function controller(integrator::DiffEqBase.DEIntegrator)
    x=copy(integrator.u[1:4])
    x[3] = mod(x[3],2*pi) - pi
    u=integrator.u[5]
    (_,mac) = integrator.p
    update!(mac,x,[u])
    integrator.u[5] = (-K(mac)*x)[1]
    integrator.u[6] = mac.candidates[1].hist[]
    integrator.u[7] = mac.candidates[2].hist[]
end

function generategif(sol, sys,tspan)
    dt = 0.05
    tu = tspan[1]:dt:tspan[2]
    (M, m, l, b, ulims) = (sys.M, sys.m, sys.l, sys.b, sys.ulims)
    θ = sol(tu)[3,:]

    x1 = sol(tu)[1, :]
    y1 = zeros(length(tu))
    x2 = x1 .+ l*sin.(θ)
    y2 = -l*cos.(θ)

    gr()
    previous_GKSwstype = get(ENV, "GKSwstype", "")
    ENV["GKSwstype"] = "100"
    anim = Animation()
    for i = 1:length(tu)
        str = string("Time = ", round(tu[i]), " sec")
        plot(
            [x1[i], x2[i]],
            [y1[i], y2[i]],
            markersize = 10,
            markershape = :circle,
            label = "",
            title = str,
            title_location = :left,
            aspect_ratio = :equal,
            xlim = (x1[i] - 1, x1[i] + 1),
            ylim = (-1, 1),
            size = (400, 300),
        )
        if i > 9
            plot!(
                [x2[i-3:i]],
                [y2[i-3:i]],
                alpha = 0.15,
                linewidth = 2,
                color = :red,
                label = "",
            )
            plot!(
                [x2[i-5:i-3]],
                [y2[i-5:i-3]],
                alpha = 0.08,
                linewidth = 2,
                color = :red,
                label = "",
            )
            plot!(
                [x2[i-7:i-5]],
                [y2[i-7:i-5]],
                alpha = 0.04,
                linewidth = 2,
                color = :red,
                label = "",
            )
            plot!(
                [x2[i-9:i-7]],
                [y2[i-9:i-7]],
                alpha = 0.01,
                linewidth = 2,
                color = :red,
                label = "",
            )
        end
        frame(anim)
    end
    ENV["GKSwstype"] = previous_GKSwstype
    return anim

end
print("Setting up system dynamics")
sys = CartPendulum.Pendulum()
Δt = 0.01;
x0 = [0; 0; π - 0.05; 0; 0]
tfinal = 20.0

linsys1 = CartPendulum.linearize(sys)
linsys1_d = c2d(linsys1, Δt, :zoh)[1]
(A1,B1,_, _) = ControlSystems.ssdata(linsys1_d)

As = [A1,A1]
Bs = [B1, -B1]
gamma = 1e4
q = [1 1 1e3 1e3];
Q = diagm(q[:])
R = 1e-2*I(1)
mac = MAController(As, Bs, Q, R, gamma, x0[1:4],.9)

tspan = (0,tfinal)
sys.nx = 0.1
sys.nθ = 0.02
condition(u,t,integrator) = t==10
affect!(integrator) = integrator.p[1].sgn = -1.

cb = PeriodicCallback(controller, Δt, 
initial_affect=true, save_positions=(false,false))
cb2 = DiscreteCallback(condition,affect!)

prob = SDEProblem(f!, g!, [x0;0;0;0], tspan, (sys, mac), callback = CallbackSet(cb,cb2))

@time sol = solve(prob)

gifname = "gain_switch_fgfac.gif"
@time anim = generategif(sol, sys, tspan)
gif(anim, gifname, fps=20)
plot(1:0.1:tfinal,sol(1:0.1:tfinal)[6:7,:]', 
labels=["sum(λ^k||*Ax_k + Bu_k - x_{k+1}||)" "sum(λ^k*||Ax_k + -Bu_k - x_{k+1}||)"],
title="Inverted Pendulum - Gain switch @ 10 s",
legend=:right)
savefig("gain_switch_fgfac.png")
