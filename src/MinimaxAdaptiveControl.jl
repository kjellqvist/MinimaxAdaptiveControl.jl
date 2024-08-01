module MinimaxAdaptiveControl
    using ArgCheck
    using JuMP, LinearAlgebra
    import MatrixEquations.ared
    export AbstractLinMod, AbstractPlant, AbstractController, AbstractPerformanceMetric
    export SSLinMod, OFLinMod, SSPlant, OFPlant, MAController , InducedlpGain, ValueFunction
    export SelfTuningLQG, RecursiveELS, stateObserver
    export reduceSys
    export fared, bared
    export MACLMIs, getPeriodicSelectionRule, getValueFunction
    export simulate!, update!, update, compute, observe, evaluate
    include("structs.jl")
    include("riccati.jl")
    include("reductions.jl")
    include("MACSynthesis.jl")
    include("simulation.jl")
    include("selfTuningRegulator.jl")
end
