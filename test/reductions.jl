using MinimaxAdaptiveControl, Test
using LinearAlgebra
include("utils.jl")
using .TestHelpers
using JuMP

function SSDoubleIntegrator()
    (nx, nu, ny) = (3, 1, 1)
    A0 = [2 -1 1; 1.0 0 0; 0 0 0]
    B0 = [0 0 1.]'
    Q = [1.0 0 0; 0 1.0 0; 0 0 1.0]
    R = fill(1.0,1,1)
    sys1 = SSLinMod(A0, B0, Q, R)
    sys2 = SSLinMod(A0, -B0, Q, R)
    sys = [sys1, sys2]
    T = 1
    γ = 19.0
    H = [
         I(nx) zeros(nx, nu + nx)
        zeros(nu, nx) I(nu) zeros(nu, nx)
        zeros(nx, 2 * nx + nu)] -γ^2 * [-A0 -B0 I(nx)]' *[-A0 -B0 I(nx)]
    models = [Model(optimizer_factory()) for i in 1:2]
    set_silent.(models)
    (A, B, G, Ks, Hs) = MinimaxAdaptiveControl.reduceSys(sys, γ, models)
    nx = size(A,1)
    ny = size(G,2)
    nu = size(B,2)
    expectedK = [1.768 -1.288 1.288] # Reported in Rantzer 2021

    @test norm(A) == 0
    @test norm(B) == 0
    @test norm(G - I(nx)) == 0
    @test norm(Ks[1] + Ks[2]) == 0
    @test Ks[1] ≈ expectedK atol = 1e-1 # Reported in Rantzer 2021
    @test Hs[1] ≈ H atol = 1e-6
end

function OFDoubleIntegrator()
    A0 = [1.0 1.0; 0 1.0]
    B0 = [0 1.0]'
    C0 = [1.0 0]
    D0 = fill(1.0,1,1) 
    G0 = [1.0 0; 0 1.0]
    Q = [1.0 0; 0 1.0]
    R = fill(1.0,1,1)
    γ = 50.0

    sys1 = OFLinMod(A0, B0, G0, C0, D0, Q, R)
    sys2 = OFLinMod(A0, B0, G0, -C0, D0, Q, R)


    sys = [sys1, sys2]
    model = Model(optimizer_factory())
    set_silent(model)
    (S, Ahat, Ghat, H, status) = fared(A0, B0, C0, D0, G0, Q, R, γ, model, method = :LMI2)
    (PLaub, KLaub, statusLaub) = bared(Ahat, B0, Ghat, H, method = :Laub)
    models = [Model(optimizer_factory()) for i in 1:2]
    set_silent.(models)
    (A, B, G, Ks, Hs) = MinimaxAdaptiveControl.reduceSys(sys, γ, models)

    (nx, nu, ny) = (2, 1, 1)
    @test A[1:nx, 1:nx] ≈ Ahat atol = 1e-6
    @test norm(B[1:nx, :] - B0) == 0
    @test G[1:nx, 1:ny] ≈ Ghat atol = 1e-6 
    @test KLaub ≈ Ks[1][1:nu, 1:nx] atol = 1e-4
end


OFDoubleIntegrator()
SSDoubleIntegrator()
