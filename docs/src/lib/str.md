# Self-Tuning LQG Controller Documentation
```@index
Pages = ["bellman.md"]
```

## Functions
```@docs
compute(controller::SelfTuningLQG{T}) where T
update!(estimator::RecursiveELS{T}, output::T, input::T) where T
update!(controller::SelfTuningLQG{T}, output::Vector{T}, input::Vector{T}; n = 1000) where T
```
