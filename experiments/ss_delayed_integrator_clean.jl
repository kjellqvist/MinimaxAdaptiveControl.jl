using MinimaxAdaptiveControl
using LinearAlgebra
using JuMP
using Plots
using Clarabel 

optimizer_factory=() -> Clarabel.Optimizer

function attemptMACLMIs(syss, γ, T)
    N = length(syss)
    models = [Model(optimizer_factory()) for i = 1:N]
    set_silent.(models)
    (A, B, G, Ks, Hs) = MinimaxAdaptiveControl.reduceSys(syss, γ, models)
    model = Model(optimizer_factory())
    set_silent(model)
    Ps0, Psplus= MACLMIs(A, B, G, Ks, Hs, T, model)
    return termination_status(model)
end

function bisect(syss, fun, T; gammamin = 1.0, gammamax = 500.0, tol = 1e-3)
    γmin = gammamin
    γmax = gammamax
    if fun(syss, gammamax, T) != MOI.OPTIMAL
        error("gammamax is not feasible")
    end
    while γmax - γmin > tol
        γ = (γmin + γmax) / 2
        if fun(syss, γ, T) == MOI.OPTIMAL
            γmax = γ
        else
            γmin = γ
        end
    end
    return γmax
end

A1 = [1. 1; 0 0]
A2 = [1. -1; 0 0]
B0 = [0 1.]'
Q = [1.0 0; 0 1.0]
R = fill(1.0,1,1)
sys1 = SSLinMod(A1, B0, Q, R)
sys2 = SSLinMod(A2, B0, Q, R)
syssA = [sys1, sys2]

A0 = [1. 1; 0 0]
B0 = [0 1.]'

sys3 = SSLinMod(A0, B0, Q, R)
sys4 = SSLinMod(A0, -B0, Q, R)
syssB = [sys3, sys4]


Tmax = 8
gammamins = zeros(Tmax, 2)
for t = 1:Tmax
    print("t = $t\n")
    try
        gammamins[t, 1] = bisect(syssA, attemptMACLMIs, t)
    catch
        gammamins[t, 1] = -1
    end
    try
        gammamins[t, 2] = bisect(syssB, attemptMACLMIs, t)
    catch
        gammamins[t, 2] = -1
    end
end

pyplot()
plot(
     gammamins, 
     seriestype=:scatter, 
     marker=[:x :o], 
     markersize = 8,
    label = ["A" "B"],
    xlabel = "Period τ",
    ylabel = "Gain bound γ"
   )
savefig("ss_delayed_integrator.png")
