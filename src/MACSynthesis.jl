"""
    MACLMIs(A::AbstractMatrix{T}, B::AbstractMatrix{T}, G::AbstractMatrix{T}, Ks::AbstractVector{M}, 
            Hs::AbstractVector{M}, period::Int, model::Model) where M<:AbstractMatrix{T} where T<:Real

Solve the Linear Matrix Inequalities (LMIs) associated with the periodic dissipation inequality for the given system matrices and optimization model.

# Arguments
- `A::AbstractMatrix{T}`: State transition matrix of the system.
- `B::AbstractMatrix{T}`: Control input matrix of the system.
- `G::AbstractMatrix{T}`: Disturbance input matrix of the system.
- `Ks::AbstractVector{M}`: Vector of feedback-gain matrices.
- `Hs::AbstractVector{M}`: Vector of symmetric cost matrices.
- `period::Int`: Control period.
- `model::Model`: JuMP optimization optimization model.

# Returns
- `Ps0::Dict{NTuple{2, Int}, Array{VariableRef, 2}}`: Dictionary of positive semi-definite matrices at time 0.
- `Psplus::Dict{NTuple{4, Int}, Array{VariableRef, 2}}`: Dictionary of positive semi-definite matrices at future time steps.
"""
function MACLMIs(A::AbstractMatrix{T}, 
        B::AbstractMatrix{T}, 
        G::AbstractMatrix{T}, 
        Ks::AbstractVector{M},
        Hs::AbstractVector{M}, 
        period::Int,
        model::Model
    ) where M<:AbstractMatrix{T} where T<:Real 

    nx = size(A,1)
    ny = size(G,2)
    nu = size(B,2)

    N = length(Hs)
    function getIneq(lhs, rhs, i, j, k)
    ineq = [lhs zeros(nx, ny)
                         zeros(ny, nx) zeros(ny, ny)] - 
                        [A - B * Ks[k] G]' * rhs * [A - B * Ks[k] G] -
                        [I(nx) zeros(nx, ny); -Ks[k] zeros(nu, ny); zeros(ny, nx) I(ny)]' *(Hs[i] + Hs[j]) * [I(nx) zeros(nx, ny); -Ks[k] zeros(nu, ny); zeros(ny, nx) I(ny)] / 2
                        return Symmetric(ineq)
    end

    Ps0 = Dict{NTuple{2, Int}, Array{VariableRef, 2}}(
                                          key => @variable(model, [1:nx, 1:nx], Symmetric, base_name = "P0$(key[1])$(key[2])") for key in Iterators.product(1:N, 1:N)
                                )
    Psplus = Dict{NTuple{4, Int}, Array{VariableRef, 2}}(
                                                         key => @variable(model, [1:nx, 1:nx], Symmetric, base_name = "Pplus$(key[1])$(key[2])$(key[3])$(key[4])") for key in Iterators.product(1:N, 1:N, 1:N, 1:period)
                                )
    for i = 1:N
        for j = 1:N
            @constraint(model, Ps0[i, j] ∈ PSDCone())
            @constraint(model, Ps0[i, j] .== Ps0[j, i])
            for k = 1:N
                for t = 1:period
                    @constraint(model, Psplus[i, j, k, t] .== Psplus[j, i, k, t])
                end
                if i != j && j == k continue end
                @constraint(model, getIneq(Psplus[i, j, k, 1], Ps0[i, j], i, j, k) in PSDCone())
                @constraint(model, Ps0[j, k] - Psplus[i, j, k, period] in PSDCone())
                for t = 1:period-1
                    @constraint(model, getIneq(Psplus[i, j, k, t + 1], Psplus[i, j, k, t], i, j, k) in PSDCone())
                end
            end
        end
    end

    optimize!(model)
    return Ps0, Psplus
end

"""
    getPeriodicSelectionRule(period::Int)

Create a selection rule for periodic control.

# Arguments
- `period::Int`: Control period.

# Returns
- `selectionRule::Function`: A function that selects the index of the control input based on the period and reward signals.

# Example
```julia
selectionRule = getPeriodicSelectionRule(4)
index = selectionRule(z, rs)
```
"""
function getPeriodicSelectionRule(period::Int)
    time = Ref{Union{Int, Nothing}}(nothing)
    ind = Ref{Union{Int, Nothing}}(nothing)
    function selectionRule(z, rs)
            if time[] == nothing
                time[] = 0
            else
                time[] = (time[] + 1) % (period)
            end
            if time[] == 0 
                ind[] = argmax(rs)
            elseif ind[] == nothing
                ind[] = argmax(rs)
                @warn "ind is nothing"
            end
            return ind[]
        end
        return selectionRule
end


"""
    getValueFunction(mac::MAController{T}, Ps0::Dict{Tuple{Int, Int}, Matrix{VariableRef}}, N::Int) where T<:Real

Create a [ValueFunction](@ref) for the given multi-agent controller and positive semi-definite matrices.

# Arguments
- `mac::MAController{T}`: Minimax adaptive controller.
- `Ps0::Dict{Tuple{Int, Int}, Matrix{VariableRef}}`: Dictionary of positive semi-definite matrices associated with the first time step in each period.
- `N::Int`: Number of models.

# Returns
- `ValueFunction`: A value function object for the multi-agent controller.

# Example
```julia
vfun = getValueFunction(mac, Ps0, N)
```
"""
function getValueFunction(mac::MAController{T}, Ps0::Dict{Tuple{Int, Int}, Matrix{VariableRef}}, N::Int) where T<:Real
    function unitv(i::Int, n::Int)
        e = zeros(n)
        e[i] = 1
        return e
    end

    weights0 = [unitv(i, N) + unitv(j, N) for i = 1:N, j = 1:N] / 2
    Ps0 = [value.(Ps0[i, j]) for i = 1:N, j = 1:N]
    return ValueFunction(0.0, Ps0[:], weights0[:], mac)
end

"""
    update!(controller::MAController{T}, output::AbstractVector{T}, control::AbstractVector{T}) where T<:Real

Update the multi-agent controller with new output and control data.

# Arguments
- `controller::MAController{T}`: The multi-agent controller to be updated.
- `output::AbstractVector{T}`: The new output measurement vector.
- `control::AbstractVector{T}`: The new control input vector.

# Description
This function updates the internal state `z` and the reward signals `rs` of the `MAController` based on the new output and control input data. The reward signals are updated for each matrix `H` in the controller's `Hs` collection. The internal state `z` is updated using the estimated system matrices `Ahat`, `Bhat`, and `Ghat`.
"""
function update!(controller::MAController{T}, output::AbstractVector{T}, control::AbstractVector{T}) where T<:Real
    for i in 1:length(controller.Hs)
        controller.rs[i] += [controller.z; control; output]' * controller.Hs[i] * [controller.z; control; output]
    end
    controller.z .= controller.Ahat * controller.z + controller.Bhat * control + controller.Ghat * output
end


"""
    compute(controller::MAController{T}) where T<:Real

Compute the control input for the multi-agent controller.

# Arguments
- `controller::MAController{T}`: The multi-agent controller.

# Returns
- `AbstractVector{T}`: The computed control input vector.

# Description
This function computes the control input for the `MAController` by selecting the appropriate control law from the `Ks` collection based on the current internal state `z` and the reward signals `rs`. The selection is made using the `selectionRule` of the controller.
"""
function compute(controller::MAController{T}) where T<:Real
    k = controller.selectionRule(controller.z, controller.rs)
    return -controller.Ks[k] * controller.z
end
