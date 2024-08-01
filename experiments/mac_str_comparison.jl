using MinimaxAdaptiveControl, LinearAlgebra
using JuMP, CSV, DataFrames, Random
using Clarabel 
optimizer_factory = () -> Clarabel.Optimizer
# Set seed
Random.seed!(10)

# Construct MAController
a = 1.01
A = [1.0 1.0; 0.0 1.0]
B1 = [0.0 1.0]'
B2 = [0.0 1.0]'
C1 = [-a + 1/a 1/a]
C2 = [-1/a + a a]
D = fill(1.0,1,1) / 10
G = [1.0 0; 0 1.0] / 100
Q = [1.0 0; 0 1.0] / 100
R = fill(1.0,1,1) / 100
γ = 20.0
sys1 = OFLinMod(A, B1, G, C1, D, Q, R)
sys2 = OFLinMod(A, B2, G, C2, D, Q, R)
sys = [sys1, sys2]
models = [Model(optimizer_factory()), Model(optimizer_factory())]
(A, B, G, Ks, Hs) = MinimaxAdaptiveControl.reduceSys(sys, γ, models)
model = Model(optimizer_factory())
set_silent(model)
period = 4
Ps0, Psplus= MACLMIs(A, B, G, Ks, Hs, period, model)
N = length(Hs)
selectionRule = getPeriodicSelectionRule(period)
mac = MAController(zeros(4), A, B, G, Ks, Hs, zeros(N), selectionRule)

# Construct STRcontroller
A0 = randn(2, 2)
B0 = randn(2, 1)
C0 = randn(1, 2)
K0 = randn(2, 1)
S0 = Matrix{Float64}(I(2))
L0 = zeros(1, 2)
Q0 = Matrix{Float64}(I(2))
ρ0 = 1.0
xhat0 = randn(2)
nu = 2
str = SelfTuningLQG(A0, K0, C0, B0, xhat0, 0.0, 1.0, nu)

# Construct simulation model 
Tdur = 2000
Asim = [1.0 1.0; 0.0 1.0]
Bsim = [0.0 1.0]'
Csim = [-a + 1/a 1/a]
Dsim = fill(1.0,1,1) / 10
Gsim = [1.0 0; 0 1.0] / 100
Qsim = [1.0 0; 0 1.0] / 100
Rsim = fill(1.0,1,1) / 100
plantMAC= OFPlant(Asim, Bsim, Gsim, Csim, Dsim, zeros(2))
plantSTR = OFPlant(Asim, Bsim, Gsim, Csim, Dsim, zeros(2))

# Disturbances
processDisturbances = randn(Tdur, 2)
measurementDisturbances = randn(Tdur, 1)

# State, output, control, and metric storage
statesMAC = zeros(Tdur + 1, 2)
statesSTR = zeros(Tdur + 1, 2)
outputsMAC = zeros(Tdur + 1, 1)
outputsSTR = zeros(Tdur + 1, 1)
controlsMAC = zeros(Tdur, 1)
controlsSTR = zeros(Tdur, 1)
vfunMAC = getValueFunction(mac, Ps0, N)
dcMAC = InducedlpGain(0.0, 0.0, 0.0, 2)
dcSTR = InducedlpGain(0.0, 0.0, 0.0, 2)
metricsMAC = [vfunMAC, dcMAC]
metricsSTR = [dcSTR]
metricResultsMAC = zeros(Tdur + 1, length(metricsMAC))
metricResultsSTR = zeros(Tdur + 1, length(metricsSTR))

# Run simulations

simulate!(statesMAC, outputsMAC, controlsMAC, processDisturbances, measurementDisturbances, metricResultsMAC, metricsMAC, plantMAC, mac, Tdur)
simulate!(statesSTR, outputsSTR, controlsSTR, processDisturbances, measurementDisturbances, metricResultsSTR, metricsSTR, plantSTR, str, Tdur)

#df = DataFrame(t = 1:Tdur, ymac = outputs[1:Tdur], umac = controls[:], valmac = metricResults[1:Tdur, 1], dcmac = metricResults[1:Tdur, 2], ystr = outputsSTR[1:Tdur], ustr = controlsSTR[:], dcstr = metricResultsSTR[1:Tdur, 1])
#CSV.write("mac_str_comparison.csv", df)
inds = period:period:Tdur
#dfperiod = DataFrame(t=inds, v=metricResults[inds, 1])
#CSV.write("mac_str_comparison_period.csv", dfperiod)
numPeriods = 6

plot(1:numPeriods*period, metricResultsMAC[1:numPeriods*period, 1], linestyle = :solid, linewidth = 2, color = :black, label = "Vbar")
plot!(inds[1:numPeriods], metricResultsMAC[inds[1:numPeriods], 1], xlabel = "Timestep", ylabel = "Value Function", label = "Periodic Vbar", linewidth = 2, linestyle = :dash, color = :blue, marker = :circle)
savefig("mac_str_value_function.png")

# Plot 2x3 grid with size 800x600
plt = plot(layout = (2, 3), size = (800, 600))
plot!(plt[1], statesMAC, xlabel = "Timestep", label = ["State 1" "State 2"], linewidth = 2, legend = :topright)
plot!(plt[2], controlsMAC, xlabel = "Timestep", ylabel = "Control", label = "Control", linewidth = 2)
plot!(plt[3], metricResultsMAC[1:end-1, 2], xlabel = "Timestep", ylabel = "Induced Lp Gain", label = "Induced Lp Gain", linewidth = 2)
plot!(plt[4], statesSTR, xlabel = "Timestep", label = ["State 1" "State 2"], linewidth = 2)
plot!(plt[5], controlsSTR, xlabel = "Timestep", ylabel = "Control", label = "Control", linewidth = 2)
plot!(plt[6], metricResultsSTR[1:end-1, 1], xlabel = "Timestep", ylabel = "Induced Lp Gain", label = "Induced Lp Gain", linewidth = 2)
savefig("mac_str_simulation.png")
