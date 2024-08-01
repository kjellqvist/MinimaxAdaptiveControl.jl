using SafeTestsets, Test

@time begin
    @safetestset "Riccati forward solutions" include("riccati/fw_riccati.jl")
    @safetestset "Riccati backward solutions" include("riccati/bw_riccati.jl")
    @safetestset "Reductions" include("reductions.jl")
end
