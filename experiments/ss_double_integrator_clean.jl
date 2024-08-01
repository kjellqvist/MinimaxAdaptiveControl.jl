using MinimaxAdaptiveControl 
using LinearAlgebra
using JuMP
using Plots
using Clarabel 
optimizer_factory = () -> Clarabel.Optimizer

A0 = [
    2.0 -1.0 1.0;
    1.0 0.0 0.0;
    0.0 0.0 0.0
   ]
B0 = [0.0; 0.0; 1.0;;]
Q = Matrix(1.0I, 3, 3)
R = Matrix(1.0I, 1, 1)
γ = 19.0
sys1 = SSLinMod(A0, B0, Q, R)
sys2 = SSLinMod(A0, -B0, Q, R)
sys = [sys1, sys2]
models = [Model(optimizer_factory()), Model(optimizer_factory())]
(A, B, G, Ks, Hs) = reduceSys(sys, γ, models)
model = Model(optimizer_factory())
period = 1
Ps0, Psplus= MACLMIs(A, B, G, Ks, Hs, period, model)
selectionRule = getPeriodicSelectionRule(period)
N = length(Hs)
mac = MAController(zeros(3), A, B, G, Ks, Hs, zeros(N), selectionRule)
Tdur = 2000
states = zeros(Tdur + 1, 3)
outputs = zeros(Tdur + 1, 3)
controls = zeros(Tdur, 1)
processDisturbances = randn(Tdur, 3)
measurementDisturbances = zeros(Tdur, 3)
vfun = getValueFunction(mac, Ps0, N)
dc= InducedlpGain(0.0, 0.0, 0.0, 2)
metrics = [vfun, dc]
metricResults = zeros(Tdur + 1, length(metrics))
plant = SSPlant(A0, -B0, zeros(3))
simulate!(states, outputs, controls, processDisturbances, measurementDisturbances, metricResults, metrics, plant, mac, Tdur)
# Plot 2x2 grid
plot(
     [states[1:Tdur, 1] controls[1:Tdur] metricResults[1:Tdur, :]], layout = (2, 2),
     xlabel = "Timestep",
     ylabel = ["State 1" "Control" "Value Function" "Induced Lp Gain"],
     legend = false,
     linewidth = 2
    )
