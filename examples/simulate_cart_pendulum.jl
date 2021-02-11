## Preamble to handle logging

# Directory for storing logs and files
dir = "runs"
logsuffix = "_cart_pendulum.log"

# Redefines print, println and printf to write both to stdout and a logfile
# Also performs git diff
include("preamble.jl")

# Much of this script is adapted from https://github.com/zaman13/Double-Pendulum-Motion-Animation
using Plots
using DifferentialEquations
using Dierckx                   # Julia library for 1-d and 2-d splines
using ProgressMeter
using LinearAlgebra
using ControlSystems
pyplot()

println("Setting up cart pendulum parameters:")
println()

M = 1;    # Mass of the cart, [kg]
m = 0.2;    # Mass of pendulum, [kg]
l = 0.6;    # Effective length of pendulum [m]
g = 9.82;   # Gravitational constant [m/s^2]
b = 0.1;    # Friction coefficient [N/m/s]
I = m * l^2 / 3;  # Mass moment of inertia [kgm^2]
h = 0.05;   # Sample interval [s]
x0 = [0; 0; 0.1; 0; 0];    # Initial state
(umin, umax) = (-10, 10)    # Actuation constraints
tfinal = 20.0;

printf("M = {},   Mass< of pendulum 1, [kg]\n", M)
printf("m = {},   Mass of pendulum 2, [kg]\n", m)
printf("l = {},   Effective length of pendulum [m]\n", l)
printf("b = {},   Friction coefficient [N/m/s]\n", b)
printf("I = {},   Mass moment of inertia [kgm^2]\n", I)
printf("g = {},    Gravitational constant [m/s^2]\n", g)
println()
println("Initial conditions:")
printf("x0[1] = {},    Cart position\n", x0[1])
printf("x0[2] = {},    Cart velocity\n", x0[2])
printf("x0[3] = {},    Pendulum angle\n", x0[3])
printf("x0[4] = {},    Pendulum angular velocity\n", x0[4])
printf("x0[5] = {},    Control signal\n", x0[5])
printf("Simulating from t = 0 to t = {}\n", tfinal)
println()
## 



function controllergain()

    denom = I * (M + m) + M * m * l^2
    a22 = -(I + m * l^2) * b / denom
    a23 = m^2 * g * l^2 / denom;
    a42 = -m * l * b / denom;
    a43 = m * g * l * (M + m) / denom;
    b2 = (I + m * l^2) / denom;
    b4 = m * l / denom;

    A = [0 1 0 0 
    0 a22 a23 0
    0 0 0 1
    0 a42 a43 0]

    B = [0;b2;0;b4];
    # For now, assume full state feedback
    # C = diagm(ones(4));
    C = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    sys_c = ss(A, B, C, 0)
    sys_d = c2d(sys_c, h, :foh)[1]
    q = [1 1 1e4 1e4];
    # Q = diagm(q[:])
    Q = [q[1] 0 0 0
    0 q[2] 0 0 
    0 0 q[3] 0
    0 0 0 q[4]]
    R = Array{Float64,2}(undef, 1, 1)
    R[1,1] = 1e-2

    K = dlqr(sys_d.A, sys_d.B, Q, R)
    return K
end

function lqrcontroller(integrator)
    x = integrator.u
    Δx = [x[1];x[2];x[3] - pi; x[4]]
    upper = min(umax, -(K * Δx)[1])
    uval = max(umin, upper);
    integrator.u[5] = uval;
end
function swingup(integrator)
    θ = integrator.u[3]
    dθ = integrator.u[4]
    k = 100.;
    E_0 = 0.1; # Target energy
    E = -m * g * l * (cos(θ) + 1) + 2 * I * dθ^2; # Energy
    upper = min(umax, k * (E - E_0) * sign(dθ * cos(θ))) # saturate upper value
    uval = max(umin, upper); # Saturate lower value
    integrator.u[5] = uval;
end
function ctr(integrator)
    θ = integrator.u[3]
    dθ = integrator.u[4]
    E = -m * g * l * (cos(θ) + 1) + 2 * I * dθ^2;
    if (1 + cos(θ)) < 0.1
        lqrcontroller(integrator)
    else
        swingup(integrator)
    end
end

function energ(sol)
    θ = sol[3,:]
    dθ = sol[4,:]
    E = -m * g * l * (cos.(θ) .+ 1) .+ 2 * I * dθ.^2;
end

println("Computing lqr gain")
K = controllergain()

# Create a periodic call back to set the controller state
cb = PeriodicCallback(ctr, h, initial_affect=true, save_positions=(false, false))

function cart_pendulum!(dx, x, p, t)
    # dx    Derivatives
    # x     States
    # p     parameters
    # t     time

    M = p[1];   # Mass of the cart, [kg]
    m = p[2];   # Mass of pendulum, [kg]
    l = p[3];   # Effective length of pendulum [m]
    g = p[4];   # Gravitational constant [m/s^2]
    b = p[5];   # Friction coefficient [N/m/s]
    I = p[6];   # Mass moment of inertia [kgm^2]
    F = x[5];   # Acting force, control signal [N]

    lump1 = m^2 * l^2 / (I + m * l^2)
    c = cos(x[3])
    s = sin(x[3])
    dx[1] = x[2];
    dx[2] = (-b * x[2] + g * lump1 * c * s + m * l * s * x[4]^2 + x[5]) / (M + m - lump1 * c^2)
    dx[3] = x[4];
    dx[4] = (-m * l * c * dx[2] - m * g * l * s) / (I + m * l^2);
    dx[5] = 0; # Controller state is set by callback and held constant otherwise
end

##
println("Starting simulation ...")
p           = [M; m; l; g; b; I];
tspan       = (0.0, tfinal);
prob        = ODEProblem(cart_pendulum!, x0, tspan, p);
@time sol   = solve(prob, Vern7(), reltol=1e-6, callback=cb);
println("Done")

##
tt = sol.t;
x1 = sol[1,:];
y1 = zeros(length(x1));

x2 = x1 .+ l * sin.(sol[3,:]);
y2 = -l * cos.(sol[3,:]);

#= 
    Interpolating to get uniformely spaced variables. 
    This is necessary to generate an animation where each frame corresponds to each time step. =#

dt = 0.05;
tu = 0:dt:tfinal;   # Uniform spacing
# Define splines
printf("Setting up uniform time grid, dt = {}", dt)
println()
sp_x1 = Spline1D(tt, x1);
sp_y1 = Spline1D(tt, y1);
sp_x2 = Spline1D(tt, x2);
sp_y2 = Spline1D(tt, y2);

# uniformely spaced states
x1_u = sp_x1(tu);
y1_u = sp_y1(tu);
x2_u = sp_x2(tu);
y2_u = sp_y2(tu);

println("Setting up animation")
#= 
    References for animation
    1. http://docs.juliaplots.org/latest/attributes/
    2. http://docs.juliaplots.org/latest/animations/ =#

anim = Animation()
@time @showprogress for i = 1:length(tu)
    str = string("Time = ", round(tu[i]), " sec")
    plot([x1_u[i],x2_u[i]], [y1_u[i],y2_u[i]],markersize=10, markershape=:circle,label="",
    title=str, title_location=:left, aspect_ratio=:equal,
    xlim=(x1_u[i] - 1, x1_u[i] + 1),ylim=(-1, 1),size=(400, 300))
    if i > 9
        plot!([x2_u[i - 3:i]], [y2_u[i - 3:i]], alpha=0.15, linewidth=2, color=:red, label="");
        plot!([x2_u[i - 5:i - 3]], [y2_u[i - 5:i - 3]], alpha=0.08, linewidth=2, color=:red, label="");
        plot!([x2_u[i - 7:i - 5]], [y2_u[i - 7:i - 5]], alpha=0.04, linewidth=2, color=:red, label="");
        plot!([x2_u[i - 9:i - 7]], [y2_u[i - 9:i - 7]], alpha=0.01, linewidth=2, color=:red, label="");
    end
    frame(anim)
end
println("Done")
gifname = string(dir, "/", datet, "_pendulum_cart.gif")
gif(anim, gifname, fps=20)
printf("Animation stored in {}", gifname)
close(logger)
