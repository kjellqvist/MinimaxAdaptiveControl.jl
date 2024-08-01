"""
    simulate!(states::AbstractMatrix{T}, outputs::AbstractMatrix{T}, controls::AbstractMatrix{T},
              processDisturbances::AbstractMatrix{T}, measurementDisturbances::AbstractMatrix{T},
              metricResults::AbstractMatrix{X}, metrics::AbstractVector{M}, plant::AbstractPlant{T},
              controller::AbstractController{T}, duration::N) where T<:Real where N<:Integer where M <: AbstractPerformanceMetric where X

Simulate the control system for a given duration, updating states, outputs, controls, and metrics.

# Arguments
- `states::AbstractMatrix{T}`: Matrix to store the state vectors over time.
- `outputs::AbstractMatrix{T}`: Matrix to store the output vectors over time.
- `controls::AbstractMatrix{T}`: Matrix to store the control vectors over time.
- `processDisturbances::AbstractMatrix{T}`: Matrix of process disturbance vectors.
- `measurementDisturbances::AbstractMatrix{T}`: Matrix of measurement disturbance vectors.
- `metricResults::AbstractMatrix{X}`: Matrix to store metric evaluation results.
- `metrics::AbstractVector{M}`: Vector of performance metrics to be evaluated.
- `plant::AbstractPlant{T}`: The plant model being controlled.
- `controller::AbstractController{T}`: The controller used to control the plant.
- `duration::N`: The duration for which the simulation runs.

# Description
This function simulates the control system over the specified duration. It updates the plant states, controller, and performance metrics at each time step.

# Example
"""
function simulate!(
        states::AbstractMatrix{T}, 
        outputs::AbstractMatrix{T}, 
        controls::AbstractMatrix{T},
        processDisturbances::AbstractMatrix{T},
        measurementDisturbances::AbstractMatrix{T},
        metricResults::AbstractMatrix{X},
        metrics::AbstractVector{M},
        plant::AbstractPlant{T},
        controller::AbstractController{T},
        duration::N
    ) where T<:Real where N<:Integer where M <: AbstractPerformanceMetric where X
    
    for t = 1:duration
        controls[t, :] .= compute(controller)
        update!(plant, controls[t, :], processDisturbances[t, :])
        states[t + 1, :] .= state(plant)
        outputs[t+1, :] .= observe(plant,measurementDisturbances[t, :])
        if typeof(plant) <: SSPlant
            update!(controller, outputs[t + 1, :], controls[t, :]) # Must be corrected for state feedback / output feedback
        else
            update!(controller, outputs[t, :], controls[t, :])
        end
        for i in 1:length(metrics)
            metrics[i] = update(metrics[i], 
                                              vcat(processDisturbances[t, :], measurementDisturbances[t, :]), 
                                              vcat(controls[t, :], outputs[t+1, :])
                                             )
            metricResults[t, i] = evaluate(metrics[i])
        end
    end
end

"""
    update!(plant::SSPlant{T}, controls::AbstractVector{T}, processDisturbances::AbstractVector{T}) where T<:Real

Update the state of a state-space plant given control inputs and process disturbances.

# Arguments
- `plant::SSPlant{T}`: The state-space plant to be updated.
- `controls::AbstractVector{T}`: The control input vector.
- `processDisturbances::AbstractVector{T}`: The process disturbance vector.

# Returns
- `Vector{T}`: The updated state vector.
"""
function update!(plant::SSPlant{T}, controls::AbstractVector{T}, processDisturbances::AbstractVector{T}) where T<:Real
    return plant.x .= plant.A * plant.x + plant.B * controls + processDisturbances
end

"""
    observe(plant::SSPlant{T}, measurementDisturbance::AbstractVector{T}) where T<:Real

Observe the output of a state-space plant given measurement disturbances.

# Arguments
- `plant::SSPlant{T}`: The state-space plant.
- `measurementDisturbance::AbstractVector{T}`: The measurement disturbance vector.

# Returns
- `Vector{T}`: The observed output vector.
"""
function observe(plant::SSPlant{T}, measurementDisturbance::AbstractVector{T}) where T<:Real
    return plant.x + measurementDisturbance
end

"""
    update!(plant::OFPlant{T}, controls::AbstractVector{T}, processDisturbances::AbstractVector{T}) where T<:Real

Update the state of an output-feedback plant given control inputs and process disturbances.

# Arguments
- `plant::OFPlant{T}`: The output-feedback plant to be updated.
- `controls::AbstractVector{T}`: The control input vector.
- `processDisturbances::AbstractVector{T}`: The process disturbance vector.

# Returns
- `Vector{T}`: The updated state vector.
"""
function update!(plant::OFPlant{T}, controls::AbstractVector{T}, processDisturbances::AbstractVector{T}) where T<:Real
    return plant.x .= plant.A * plant.x + plant.B * controls + plant.G * processDisturbances
end

"""
    observe(plant::OFPlant{T}, measurementDisturbance::AbstractVector{T}) where T<:Real

Observe the output of an output-feedback plant given measurement disturbances.

# Arguments
- `plant::OFPlant{T}`: The output-feedback plant.
- `measurementDisturbance::AbstractVector{T}`: The measurement disturbance vector.

# Returns
- `Vector{T}`: The observed output vector.

# Example
"""
function observe(plant::OFPlant{T}, measurementDisturbance::AbstractVector{T}) where T<:Real
    return plant.C * plant.x + plant.D * measurementDisturbance
end

"""
    update(id::InducedlpGain{T, N}, input::AbstractVector{T}, output::AbstractVector{T}) where T<:Real where N<:Integer

Update the induced l_p gain metric with new input and output data.

# Arguments
- `id::InducedlpGain{T, N}`: The induced l_p gain metric to be updated.
- `input::AbstractVector{T}`: The input vector.
- `output::AbstractVector{T}`: The output vector.

# Returns
- `InducedlpGain{T, N}`: The updated induced l_p gain metric.
"""
function update(id::InducedlpGain{T, N}, input::AbstractVector{T}, output::AbstractVector{T}) where T<:Real where N<:Integer
    sumpInput = id.sumpInput + sum(input.^id.p)
    sumpOutput = id.sumpOutput + sum(output.^id.p)
    gain = (sumpOutput / sumpInput)^(1/id.p)
    return InducedlpGain{T, N}(gain, sumpOutput, sumpInput, id.p)
end

"""
    update(v::ValueFunction{T}, input::AbstractVector{T}, output::AbstractVector{T}) where T<:Real

Update the value function with new input and output data.

# Arguments
- `v::ValueFunction{T}`: The value function to be updated.
- `input::AbstractVector{T}`: The input vector.
- `output::AbstractVector{T}`: The output vector.

# Returns
- `ValueFunction{T}`: The updated value function.
"""
function update(v::ValueFunction{T}, input::AbstractVector{T}, output::AbstractVector{T}) where T<:Real
    vals = [v.controller.z' * v.Ps[i] * v.controller.z + v.weights[i]' * v.controller.rs for i in 1:length(v.Ps)]
    return ValueFunction{T}(maximum(vals), v.Ps, v.weights, v.controller)
end

function update(m::InternalStates{T}, input::AbstractVector{T}, output::AbstractVector{T}) where T<:Real
    states = [m.controller.z;m.controller.rs]
    return InternalStates{T}(states, m.controller)
end

"""
    evaluate(v::AbstractPerformanceMetric{T}) where T<:Real

Evaluate the value of a performance metric.

# Arguments
- `v::AbstractPerformanceMetric{T}`: The performance metric to be evaluated.

# Returns
- `T`: The value of the performance metric.
"""
function evaluate(v::AbstractPerformanceMetric{T}) where T<:Real
    return return v.value
end

"""
    state(plant::SSPlant{T}) where T<:Real

Get the state vector of a state-space plant.

# Arguments
- `plant::SSPlant{T}`: The state-space plant.

# Returns
- `Vector{T}`: The state vector of the plant.
"""
function state(plant::SSPlant{T}) where T<:Real
    return plant.x
end

"""
    state(plant::OFPlant{T}) where T<:Real

Get the state vector of an output-feedback plant.

# Arguments
- `plant::OFPlant{T}`: The output-feedback plant.

# Returns
- `Vector{T}`: The state vector of the plant.
"""
function state(plant::OFPlant{T}) where T<:Real
    return plant.x
end
