```@index
Pages = ["bellman.md"]
```


```@docs
MACLMIs
compute(controller::MAController{T}) where T<:Real
getPeriodicSelectionRule
getValueFunction
update!(controller::MAController{T}, output::AbstractVector{T}, control::AbstractVector{T}) where T<:Real
```
