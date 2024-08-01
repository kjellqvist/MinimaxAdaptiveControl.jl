
abstract type AbstractRecursiveEstimator{T} end

"""
    RecursiveELS{T}

Extended Least Squares (ELS) estimator for recursive parameter estimation.

# Fields
- `parameterEstimates::Vector{T}`: Vector of current parameter estimates.
- `inverseErrorCovariance::Matrix{T}`: Inverse of the error covariance matrix.
- `regressors::Vector{T}`: Vector of regressor values.
- `ny::Int`: Number of outputs.
- `nu::Int`: Number of inputs.
- `ne::Int`: Number of errors.
- `regularization::T`: Regularization parameter for numerical stability.
"""
struct RecursiveELS{T} <: AbstractRecursiveEstimator{T}
    parameterEstimates::Vector{T}
    inverseErrorCovariance::Matrix{T}
    regressors::Vector{T}
    ny::Int
    nu::Int
    ne::Int
    regularization::T
end


"""
    stateObserver{T}

State observer, also known as a Kalman Filter, for estimating the state of a system.

# Fields
- `A::Matrix{T}`: State transition matrix.
- `K::Matrix{T}`: Kalman gain matrix.
- `C::Matrix{T}`: Measurement matrix.
- `B::Matrix{T}`: Control input matrix.
- `xhat::Vector{T}`: Estimated state vector.
"""
struct stateObserver{T}
    A::Matrix{T}
    K::Matrix{T}
    C::Matrix{T}
    B::Matrix{T}
    xhat::Vector{T}
end


"""
SelfTuningLQG(A0::Matrix{T}, K0::Matrix{T}, C0::Matrix{T}, B0::Matrix{T}, xhat0::Vector{T}, 
                  regularization::T, ρ::T, nu::Int) where T

Initialize the self-tuning LQG controller.

# Arguments
- `A0::Matrix{T}`: Initial state transition matrix.
- `K0::Matrix{T}`: Initial Kalman gain matrix.
- `C0::Matrix{T}`: Initial measurement matrix.
- `B0::Matrix{T}`: Initial control input matrix.
- `xhat0::Vector{T}`: Initial state estimate.
- `regularization::T`: Regularization parameter for numerical stability.
- `ρ::T`: Regularization parameter for the Riccati equation.
- `nu::Int`: Number of control inputs.

# Returns
- `SelfTuningLQG{T}`: Initialized self-tuning LQG controller.
"""
struct SelfTuningLQG{T} <: AbstractController{T}
    observer::stateObserver{T}
    estimator::RecursiveELS{T}
    L::Matrix{T}
    S::Matrix{T}
    Q::Matrix{T}
    ρ::T
end

# Initialize the self-tuning LQG controller
function SelfTuningLQG(A0::Matrix{T}, K0::Matrix{T}, C0::Matrix{T}, B0::Matrix{T}, xhat0::Vector{T}, regularization::T, ρ::T, nu::Int) where T
    # Initialize the state observer
    observer = stateObserver(A0, K0, C0, B0, xhat0)
    ny = size(A0, 1)
    ne = size(K0, 1)
    # Initialize the RecursiveELS estimator
    parameterEstimates = zeros(T, ny + nu + ne)
    inverseErrorCovariance = Matrix{Float64}(I(ny + nu + ne))  # Identity matrix for simplicity
    regressors = zeros(T, ny + nu + ne)
    estimator = RecursiveELS(parameterEstimates, inverseErrorCovariance, regressors, ny, nu, ne, regularization)

    # Initialize the LQG controller
    L = zeros(T, 1, ny)
    S = Matrix{Float64}(I(ny))
    Q = Matrix{Float64}(I(ny))
    return SelfTuningLQG(observer, estimator, L, S, Q, ρ)
end


"""
    update!(estimator::RecursiveELS{T}, output::T, input::T) where T

Update the Recursive ELS estimator with new output and input data.

# Arguments
- `estimator::RecursiveELS{T}`: The Recursive ELS estimator to be updated.
- `output::T`: The new output measurement.
- `input::T`: The new input measurement.

# Returns
- `T`: The posteriori prediction error after updating the estimator.
"""
function update!(estimator::RecursiveELS{T}, output::T, input::T) where T
    # Dimensionality checks (example)
    @assert length(estimator.parameterEstimates) == estimator.ny + estimator.nu + estimator.ne "Dimension mismatch in parameter estimates."
    @assert size(estimator.inverseErrorCovariance) == (estimator.ny + estimator.nu + estimator.ne, estimator.ny + estimator.nu + estimator.ne) "Dimension mismatch in inverse error covariance."

    # Extracting elements from the estimator for clarity and convenience
    θ = @view estimator.parameterEstimates[:]
    Pinv = @view estimator.inverseErrorCovariance[:, :]
    ϕ = @view estimator.regressors[:]
    (ny, nu, ne) = (estimator.ny, estimator.nu, estimator.ne)

    # Adding regularization term to maintain numerical stability
    Pinv .+= estimator.regularization * I(ny + nu + ne)

    # Compute the a priori prediction error
    prioriPE = output - dot(ϕ, θ)

    # Update the inverse error covariance matrix
    Pinv .+= ϕ * ϕ'

    # Update the parameter estimates (using the backslash operator for efficiency and stability)
    θ .+= (Pinv \ ϕ) * prioriPE

    # Compute the a posteriori prediction error (optional, can be used for monitoring)
    posterioriPE = output - dot(ϕ, θ)

    # Update the regressor vectors
    update_regressors!(ϕ, ny, nu, ne, output, input, posterioriPE)

    return posterioriPE  # Returning posteriori prediction error if needed for monitoring or other purposes
end



"""
    updateObserverState!(obs::stateObserver{T}, output::T, input::T) where T

Update the state of the observer with new output and input data.

# Arguments
- `obs::stateObserver{T}`: The state observer to be updated.
- `output::T`: The new output measurement.
- `input::T`: The new input measurement.

# Returns
- `Vector{T}`: The updated state estimate.

"""
function updateObserverState!(obs::stateObserver{T}, output::T, input::T) where T
    obs.xhat .= obs.A * obs.xhat + obs.B * input + obs.K * (output - (obs.C * obs.xhat)[1])
    return obs.xhat
end

"""
    updateObserverParameters!(obs::stateObserver{T}, estimator::RecursiveELS{T}) where T

Update the parameters of the observer using the Recursive ELS estimator.

# Arguments
- `obs::stateObserver{T}`: The state observer to be updated.
- `estimator::RecursiveELS{T}`: The Recursive ELS estimator providing new parameter estimates.
"""
function updateObserverParameters!(obs::stateObserver{T}, estimator::RecursiveELS{T}) where T
    ny = estimator.ny
    nu = estimator.nu
    ne = estimator.ne
    ACoeffs = -estimator.parameterEstimates[1:estimator.ny]
    BCoeffs = estimator.parameterEstimates[estimator.ny+1:estimator.ny+estimator.nu]
    CCoeffs = estimator.parameterEstimates[estimator.ny+estimator.nu+1:end]
    A = zeros(T, estimator.ny, estimator.ny)
    # Set hyperdiagonal to 1
    for i = 1:(estimator.ny - 1)
        A[i, i + 1] = 1
    end
    A[:, 1] .= -ACoeffs
    B = zeros(T, estimator.ny, 1)
    B[(ny -nu + 1):end] .= BCoeffs
    K = CCoeffs - ACoeffs
    C = zeros(T, 1, estimator.ny)
    C[1, 1] = 1
    obs.A .= A
    obs.B .= B
    obs.C .= C
    obs.K .= K
end


"""
    update_regressors!(ϕ::AbstractVector{T}, ny::Int, nu::Int, ne::Int, output::T, input::T, PE::T) where T

Update the regressor vector with new output, input, and prediction error data.

# Arguments
- `ϕ::AbstractVector{T}`: The regressor vector to be updated.
- `ny::Int`: Number of outputs.
- `nu::Int`: Number of inputs.
- `ne::Int`: Number of errors.
- `output::T`: The new output measurement.
- `input::T`: The new input measurement.
- `PE::T`: The prediction error.
"""
function update_regressors!(ϕ::AbstractVector{T}, ny::Int, nu::Int, ne::Int, output::T, input::T, PE::T) where T
    ϕy = @view ϕ[1:ny]
    ϕu = @view ϕ[ny+1:ny+nu]
    ϕe = @view ϕ[ny+nu+1:end]

    # Update the regressor vectors with new data
    circshift!(ϕy, 1)
    ϕy[1] = output
    circshift!(ϕu, 1)
    ϕu[1] = input
    circshift!(ϕe, 1)
    ϕe[1] = PE
end


"""
    riccatiStep!(S::Matrix{T}, L::Matrix{T}, A::Matrix{T}, B::Matrix{T}, Q::Matrix{T}, ρ::T) where T

Perform a single Riccati equation update step.

# Arguments
- `S::Matrix{T}`: The current solution to the Riccati equation.
- `L::Matrix{T}`: The state feedback gain matrix.
- `A::Matrix{T}`: The state transition matrix.
- `B::Matrix{T}`: The control input matrix.
- `Q::Matrix{T}`: The state weighting matrix.
- `ρ::T`: The regularization parameter.
"""
function riccatiStep!(S::Matrix{T}, L::Matrix{T}, A::Matrix{T}, B::Matrix{T}, Q::Matrix{T}, ρ::T) where T
    L .= (ρ * I + B' * S * B) \ (B' * S * A)
    S .= (A - B * L)' * S * (A - B * L) + Q + ρ * L' * L
end


"""
    update!(controller::SelfTuningLQG{T}, output::Vector{T}, input::Vector{T}; n = 1000) where T

Update the Self-Tuning LQG controller with new output and input data.

# Arguments
- `controller::SelfTuningLQG{T}`: The Self-Tuning LQG controller to be updated.
- `output::Vector{T}`: The new output measurement vector.
- `input::Vector{T}`: The new input measurement vector.
- `n::Int`: Number of Riccati equation steps to perform (default: 1000).
"""
function update!(controller::SelfTuningLQG{T}, output::Vector{T}, input::Vector{T}; n = 1000) where T
    # Update the estimator
    posterioriPE = update!(controller.estimator, output[1], input[1])

    # Update the observer parameters
    updateObserverParameters!(controller.observer, controller.estimator)

    # Update the observer state
    updateObserverState!(controller.observer, output[1], input[1])

    # Update the Riccati equation
    for t = 1:n
        riccatiStep!(controller.S, controller.L, controller.observer.A, controller.observer.B, controller.Q, controller.ρ)
    end
end


"""
    compute(controller::SelfTuningLQG{T}) where T

Compute the control input for the Self-Tuning LQG controller.

# Arguments
- `controller::SelfTuningLQG{T}`: The Self-Tuning LQG controller.

# Returns
- `Vector{T}`: The computed control input.
"""
function compute(controller::SelfTuningLQG{T}) where T
    return -controller.L * controller.observer.xhat .+ randn()
end
