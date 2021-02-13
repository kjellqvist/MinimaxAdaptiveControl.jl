using LinearAlgebra
using JuMP
using Hypatia
using MinimaxAdaptiveControl
using Test
##
@testset "Recreate Anders' results" begin
    A = [2 -1 1.0; 1 0 0; 0 0 0]
    B1 = Matrix{Float64}(undef, 3, 1)
    B1[:] = [0; 0; 1.0]
    B2 = -B1
    Q = I(3)
    γ = 19
    R = I(1)
    As = [A, A]
    Bs = [B1, B2]

    mac = MinimaxAdaptiveControl.MAController(As, Bs, Q, R, γ, [0.; 0; 0])
    PAnders = [
        20.628 -11.101 11.101
        -11.101 7.842 -6.842
        11.101 -6.842 7.842
    ]
    KAnders = [1.786 -1.288 1.288]

    TAnders = [155 -84.4 84.4
                -84.4 89.0 -87.5
                84.4 -87.5 89.0]
    @test norm(mac.candidates[1].P - mac.candidates[2].P) ≈ 0
    @test norm(mac.candidates[1].K + mac.candidates[2].K) ≈ 0
    @test norm(mac.candidates[1].P - PAnders) ≈ 0 atol = 1e-1
    @test norm(mac.candidates[1].K - KAnders) ≈ 0 atol = 1e-3
end

@testset "Validate Tsyn" begin
    A = [2 -1 1.0; 1 0 0; 0 0 0]
    B1 = Matrix{Float64}(undef, 3, 1)
    B1[:] = [0; 0; 1.0]
    B2 = -B1
    Q = I(3)
    γ = 19
    R = I(1)
    As = [A, A]
    Bs = [B1, B2]

    mac = MAController(As, Bs, Q, R, γ, [0.; 0; 0])
    model = Model(Hypatia.Optimizer)
    unset_silent(model) # For now broken...
    Tval, stat = Tsyn(mac, model);

    P = mac.candidates[1].P
    B = B1
    K = mac.candidates[1].K
    # Tval = Tval + 1e-4*I(3) # Might be needed
    ineq13 = P - (Q + K' * R * K + ((A - B * K)' / (inv(P) - γ^(-2) * I(3))) * (A - B * K))
    ineq14 = Tval - (Q + K' * R * K + ((A + B * K)' / (inv(P) - γ^(-2) * I(3))) * (A + B * K))
    ineq15 = Tval - (Q + K' * (R - γ^2 * B' * B) * K + (A' / (inv(Tval) - γ^(-2) * I(3)) * A))

    @test minimum(eigvals(ineq13)) > 0
    @test minimum(eigvals(ineq14)) > 0
    @test minimum(eigvals(ineq15)) > 0
    @test maximum(eigvals(Tval)) < γ^2
end

@testset "Testing X and Z" begin
    # Simple. one-dim
    A1, A2, B1, B2, Q, R = [fill(i * 1., 1, 1) for i in [1,2,3,5,7,11]]
    γ = 13
    As = [A1, A2]
    Bs = [B1, B2]
    mac = MinimaxAdaptiveControl.MAController(As, Bs, Q, R, γ, [0.; 0; 0])
    K1 = mac.candidates[1].K
    P1 = mac.candidates[1].P
    K2 = mac.candidates[2].K
    P2 = mac.candidates[2].P

    model = Model(Hypatia.Optimizer)
    @variable(model, T[1:1,1:1] in PSDCone())

    X12_ref = [(T - Q - K2' * R * K2) (A1 - B1 * K2)
            (A1 - B1 * K2)' inv(P1) - I(1) / γ^2]
    @test X12_ref == MinimaxAdaptiveControl.X(mac, T, 1, 2, 0)

    A11 = A1 - B1 * K1
    A21 = A2 - B2 * K1
    Z121_ref11 = T - Q - K1' * R * K1 + γ^2 / 2 * (A11' * A11 + A21' * A21) 
    Z121_ref21 = (A11 + A21)
    Z121_ref12 = (A11 + A21)'
    Z121_ref22 = 4 / γ^4 * (γ^2 * I(1) - T)
    Z121_ref = [Z121_ref11 Z121_ref12
                Z121_ref21 Z121_ref22]
    
    @test Z121_ref == MinimaxAdaptiveControl.Z(mac, T, 1, 2, 1, 0)
end

@testset "test valuefunction" begin
    A = [3. -1;1 0]
    B1 = Matrix{Float64}(undef, 2, 1)
    B1[:,:] = [1. 0]
    B2 = Matrix{Float64}(undef, 2, 1)
    B2[:,:] = [5 0]

    As = [A,A]
    Bs = [B1,B2]
    Q = I(2) * 1.
    gamma = 22
    R = I(1) * 1.

    mac = MAController(As, Bs, Q, R, gamma, [0.; 0])
    model = Model(Hypatia.Optimizer)
    unset_silent(model)
    Tval, stat = Tsyn(mac, model, 0.01)

    @testset "V is monotonic" begin
        N = 100
        x = [3;-5.]
        u = [0.]
        Vs = zeros(N)
        for k = 1:N
            update!(mac, x, u)
            u = -K(mac)*x
            x = A*x + B1*u
            Vs[k] = Vbar(mac,Tval, x)
        end
        Vdiffs = Vs[2:N] - Vs[1:N-1]
        @test maximum(Vdiffs) <= 0
    end

    @testset "Reduction to ±B" begin
        A = [2 -1 1.0; 1 0 0; 0 0 0]
        B1 = Matrix{Float64}(undef, 3, 1)
        B1[:] = [0; 0; 1.0]
        B2 = -B1
        Q = I(3)
        γ = 19
        R = I(1)*1.
        As = [A, A]
        Bs = [B1, B2]
    
        x0 = [0.; 0; 0]
        mac = MinimaxAdaptiveControl.MAController(As, Bs, Q, R, γ,x0 )
        model = Model(Hypatia.Optimizer)
        unset_silent(model)
        Tval, stat = Tsyn(mac, model, 1e-4)


        x = [3;-5.;7]
        u = [2.]
        function Vval(mac, x,x0, u)
            P = mac.candidates[1].P
            c1 = x'*P*x - γ^2*(-x + A*x0 +B1*u)'*(-x + A*x0 +B1*u)
            c2 = x'*P*x - γ^2*(-x + A*x0 -B1*u)'*(-x + A*x0 -B1*u)
            c3 = x'*Tval*x - γ^2*((-x + A*x0)'*(-x + A*x0) + u'*B1'*B1*u)
            return maximum([c1, c2, c3])
        end

        update!(mac, x,u)
        @test Vval(mac,x,x0,u) == Vbar(mac,Tval, x)
    end
end