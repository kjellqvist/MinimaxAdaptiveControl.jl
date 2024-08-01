```@index
Pages = ["simulation.md"]
```


```@docs
evaluate(v::AbstractPerformanceMetric{T}) where T<:Real
observe(plant::SSPlant{T}, measurementDisturbance::AbstractVector{T}) where T<:Real
observe(plant::OFPlant{T}, measurementDisturbance::AbstractVector{T}) where T<:Real
simulate!
update!(plant::SSPlant{T}, controls::AbstractVector{T}, processDisturbances::AbstractVector{T}) where T<:Real
update!(plant::OFPlant{T}, controls::AbstractVector{T}, processDisturbances::AbstractVector{T}) where T<:Real
update(id::InducedlpGain{T, N}, input::AbstractVector{T}, output::AbstractVector{T}) where T<:Real where N<:Integer
update(v::ValueFunction{T}, input::AbstractVector{T}, output::AbstractVector{T}) where T<:Real
```
