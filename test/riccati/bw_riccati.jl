using MinimaxAdaptiveControl, Test
using LinearAlgebra
using JuMP
include("../utils.jl")
using .TestHelpers


function SSDoubleIntegrator()
    (nx, nu, ny) = (3, 1, 1)
    A = [2 -1 1; 1.0 0 0; 0 0 0]
    B = [0 0 1.]'
    Q = [1.0 0 0; 0 1.0 0; 0 0 1.0]
    R = fill(1.0,1,1)
    γ = 19.0
    H = [
         I(nx) zeros(nx, nu + nx)
        zeros(nu, nx) I(nu) zeros(nu, nx)
        zeros(nx, 2 * nx + nu)] -γ^2 * [-A -B I(nx)]' *[-A -B I(nx)]
    expectedK = [1.768 -1.288 1.288] # Reported in Rantzer 2021
    expectedP = [20.607504575161506 -11.090309602220952 11.090309602220964; -11.090309602220952 7.834711219838143 -6.834711219838159; 11.090309602220964 -6.834711219838159 7.8347112198381454] # Computed. Used for regression testing.
    
    Ahat = zeros(nx, nx)
    Bhat = zeros(nx, nu)
    Ghat = Matrix{Float64}(I(nx))
    (PLaub, KLaub, statusLaub) = bared(Ahat, Bhat, Ghat, H, method = :Laub)
    (Piterate, Kiterate, statusiterate) = bared(Ahat, Bhat, Ghat, H, method = :Iterate)
    model = Model(optimizer_factory())
    set_silent(model)
    (Plmi, Klmi, statuslmi) = bared(Ahat, Bhat, Ghat, H, model)

    @test KLaub ≈ expectedK atol = 1e-1
    @test Kiterate ≈ expectedK atol = 1e-1
    @test Klmi ≈ expectedK atol = 1e-1
    @test PLaub ≈ expectedP atol = 1e-4
    @test Piterate ≈ expectedP atol = 1e-4
    @test Plmi ≈ expectedP atol = 1e-4

end

function OFDoubleIntegrator()
    Ahat = [-0.26002762334695195 1.0243830829984153; -0.4347113498360782 1.0199920592726601]
    Ghat = [1.272755173671214; 0.4391023721588011;;]
    Bhat = [0.0; 1.0;;]
    H = [-17.29368872289138 -0.43471132655822153 0.0 17.46837266062767; -0.43471132655822153 1.0199921239573375 0.0 0.43910238444824307; 0.0 0.0 1.0 0.0; 17.46837266062767 0.43910238444824307 0.0 -16.634719736227773]

    expectedK =  [0.8874829503412103 1.8475349967343753]
    expectedP = [4.550131581066114 4.358648409053117; 4.358648409053117 7.147210526659126]

    (PLaub, KLaub, statusLaub) = bared(Ahat, Bhat, Ghat, H, method = :Laub)
    (Piterate, Kiterate, statusiterate) = bared(Ahat, Bhat, Ghat, H, method = :Iterate)
    model = Model(optimizer_factory())
    set_silent(model)
    (Plmi, Klmi, statuslmi) = bared(Ahat, Bhat, Ghat, H, model)
    @test KLaub ≈ expectedK atol = 1e-3
    @test Kiterate ≈ expectedK atol = 1e-3
    @test Klmi ≈ expectedK atol = 1e-3
    @test PLaub ≈ expectedP atol = 1e-3
    @test Piterate ≈ expectedP atol = 1e-3
    @test Plmi ≈ expectedP atol = 1e-3
end

SSDoubleIntegrator()
OFDoubleIntegrator()
