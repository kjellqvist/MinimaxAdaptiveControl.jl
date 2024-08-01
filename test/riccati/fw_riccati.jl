using MinimaxAdaptiveControl, Test
using LinearAlgebra
include("../utils.jl")
using .TestHelpers
using JuMP

function intergratorTest()
    A = fill(1.0,1,1)
    B = A
    C = A
    D = A
    G = A
    Q = A
    R = A
    γ = 2
    expected = sqrt(57) / 2 - 3/2
    model = Model(optimizer_factory())
    set_silent(model)
    @test fared(A, B, C, D, G, Q, R, γ, model, method = :LMI1)[1][1] ≈ expected atol = 1e-6
    model = Model(optimizer_factory())
    set_silent(model)
    @test fared(A, B, C, D, G, Q, R, γ, model, method = :LMI2)[1][1] ≈ expected atol = 1e-6
    @test fared(A, B, C, D, G, Q, R, γ, method = :Iterate, iters = 1000)[1][1] ≈ expected atol = 1e-6
    @test fared(A, B, C, D, G, Q, R, γ, method = :Laub)[1][1] ≈ expected atol = 1e-6
    @test_throws DimensionMismatch MinimaxAdaptiveControl.assert_dimensions_([1 1; 2 3], B, C, D, G, Q, R)
end

function doubleIntegratorTest()
    A = [1.0 1.0; 0 1.0]
    B = [0 1.0]'
    C = [1.0 0]
    D = fill(1.0,1,1) 
    G = [1.0 0; 0 1.0]
    Q = [1.0 0; 0 1.0]
    R = fill(1.0,1,1)
    γ = 5.0

    expectedS = [8.951317357713993 -7.534539776289985; -7.534539776289985 14.215085538024647]
    expectedAhat = [-0.31505277678093757 1.1069116495670608; -0.4775178981665567 1.0870150705985537]
    expectedGhat = [1.3698466415310293; 0.49741447587025817;;]
    expectedH = [-3.8991628994663783 -0.477517917330192 0.0 4.061628034640479; -0.4775179173301902 1.0870151061335722 0.0 0.4974144742126705; 0.0 0.0 1.0 0.0; 4.061628034640479 0.4974144742126705 0.0 -3.189195866815868]

    model = Model(optimizer_factory())
    set_silent(model)
    LMI1result = fared(A, B, C, D, G, Q, R, γ, model, method = :LMI1)
    @test LMI1result[1] ≈ expectedS atol = 1e-6
    @test LMI1result[2] ≈ expectedAhat atol = 1e-6
    @test LMI1result[3] ≈ expectedGhat atol = 1e-6
    @test LMI1result[4] ≈ expectedH atol = 1e-6

    model = Model(optimizer_factory())
    set_silent(model)
    LMI2result = fared(A, B, C, D, G, Q, R, γ, model, method = :LMI2)
    @test LMI2result[1] ≈ expectedS atol = 1e-6
    @test LMI2result[2] ≈ expectedAhat atol = 1e-6
    @test LMI2result[3] ≈ expectedGhat atol = 1e-6
    @test LMI2result[4] ≈ expectedH atol = 1e-6

    interateResult = fared(A, B, C, D, G, Q, R, γ, method = :Iterate, iters = 1000) 
    @test interateResult[1] ≈ expectedS atol = 1e-6
    @test interateResult[2] ≈ expectedAhat atol = 1e-6
    @test interateResult[3] ≈ expectedGhat atol = 1e-6
    @test interateResult[4] ≈ expectedH atol = 1e-6

    laubResult = fared(A, B, C, D, G, Q, R, γ, method = :Laub)  
    @test laubResult[1] ≈ expectedS atol = 1e-6
    @test laubResult[2] ≈ expectedAhat atol = 1e-6
    @test laubResult[3] ≈ expectedGhat atol = 1e-6
    @test laubResult[4] ≈ expectedH atol = 1e-6
end

intergratorTest()
doubleIntegratorTest()
