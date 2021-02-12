# MinimaxAdaptiveControl

[![Unit Tests](https://github.com/kjellqvist/MinimaxAdaptiveControl.jl/workflows/CI/badge.svg)](https://github.com/kjellqvist/MinimaxAdaptiveControl.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/kjellqvist/MinimaxAdaptiveControl.jl/branch/master/graph/badge.svg?token=C0M1GL5BOQ)](https://codecov.io/gh/kjellqvist/MinimaxAdaptiveControl.jl)

## Julia package for Minimax adaptive control of a finite set of linear systems
Synthesize controllers and lyapunov functions from https://arxiv.org/abs/2011.10814

## Installing
```
] add https://github.com/kjellqvist/MinimaxAdaptiveControl.jl
```

## Example
Create a controller object
```
  using LinearAlgebra
  using JuMP
  using Hypatia
  using MinimaxAdaptiveControl

  A = [3. -1;1 0]
  B1 = Matrix{Float64}(undef,2,1)
  B1[:,:] = [1. 0]
  B2 = Matrix{Float64}(undef, 2,1)
  B2[:,:] = [5 0]

  As = [A,A]
  Bs = [B1,B2]
  Q = I(2)*1.
  gamma = 22
  R = I(1)*1.

  mac = MAController(As, Bs, Q, R, gamma, [0.; 0])

```

Synthesize a T-matrix which satisfies inequalities (19) & (20)
```
  model = Model(Hypatia.Optimizer)
  unset_silent(model)
  Tval, stat = Tsyn(mac, model, 0.01)
```

Update the controller with new data
```
  x0 = [0.;0]
  u0 = [1.]
  x1 = A*x0 + B1*u0   # Next state
  update!(mac, x, u)  # Update the controller
  u1 = -K(mac)x1      # Next control signal
```
