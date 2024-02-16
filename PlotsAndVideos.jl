using DifferentialEquations, DynamicalSystems
using GLMakie
using LinearAlgebra: norm, dot
using OrdinaryDiffEq
using InteractiveDynamics

function mem_system(du, u, p, t)
    x, y, z, v = u
    a, b, c, d, e, α, β = p

    du[1] = -(a*y^2 - b)*x - (α*v + β)*z
    du[2] = -c*x - d*y + e*y*x^2
    du[3] = x
    du[4] = z

    return nothing
end


# https://juliadynamics.github.io/InteractiveDynamics.jl/dev/dynamicalsystems/

u0 = [-1, -0.5, -0.5, -3] # Initial condition
p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.005, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u0 , p1, diffeq =integrationMethod)

N = 10^5 # takes around 7 sec
lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)

total_time = 300
sampling_time = 0.01
transient_time = 0
Y, t = trajectory(syst_1, total_time; Ttr = transient_time , Δt = sampling_time)

# Plot the evolution of the four variables over time
fig = Figure()
for (i, var) in enumerate(columns(Y))
    ax = Axis(fig[i, 1]; xlabel = "time", ylabel = "variable $i")
    lines!(ax, t, Y[:, i])
end
fig

fig = Figure()
# Time series x
ax1 = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
lines!(ax1, t, Y[:, 1])
# Time series y
ax2 = Axis(fig[1, 2]; xlabel = "y", ylabel = "z")
lines!(ax2, t, Y[:, 2])
# Time series z
ax3 = Axis(fig[2, 1]; xlabel = "z", ylabel = "u")
lines!(ax3, t, Y[:, 3])
# Time series v
ax4 = Axis(fig[2, 2]; xlabel = "v", ylabel = "x")
lines!(ax4, t, Y[:, 4])
fig

# -------------------- Phase portrait Projections -------------------- #
fig = Figure(size = (600, 600))
# Proyección x-y
ax1 = Axis(fig[1, 1]; xlabel = L"x_1", ylabel = L"x_2", xlabelsize = 30, ylabelsize = 30)
lines!(ax1, Y[:, 1], Y[:, 2])
# Proyección z-y
ax2 = Axis(fig[1, 2]; xlabel = L"x_2", ylabel = L"x_3", xlabelsize = 30, ylabelsize = 30)
lines!(ax2, Y[:, 2], Y[:, 3])
# Proyección v-z
ax3 = Axis(fig[1, 3];  xlabel = L"x_3", ylabel = L"x_4", xlabelsize = 30, ylabelsize = 30)
lines!(ax3, Y[:, 3], Y[:, 4])
# Proyección x-v
ax4 = Axis(fig[1, 4]; xlabel = L"x_4", ylabel = L"x_1", xlabelsize = 30, ylabelsize = 30)
lines!(ax4, Y[:, 4], Y[:, 1])

root = joinpath(@__DIR__, "images", "chaoticAtractor.png")
save(root, fig)
# -------------------- x1-x2 Projections for several a -------------------- #
total_time = 600
sampling_time = 0.01
transient_time = 0

a1, a2, a3 = 0.00006, 0.02, 1.18
b1, b2, b3 = 0.5, 0.37, 0.5

set_parameter!(syst_1, 1, a1)
set_parameter!(syst_1, 2, b1)
Y1, t1 = trajectory(syst_1, total_time; Ttr = transient_time , Δt = sampling_time)

set_parameter!(syst_1, 1, a2)
set_parameter!(syst_1, 2, b2)
Y2, t2 = trajectory(syst_1, total_time; Ttr = transient_time , Δt = sampling_time)

set_parameter!(syst_1, 1, a3)
set_parameter!(syst_1, 2, b3)
Y3, t3 = trajectory(syst_1, total_time; Ttr = transient_time , Δt = sampling_time)

# set_parameter!(syst_1, 1, a4)
# Y4, t4 = trajectory(syst_1, total_time; Ttr = transient_time , Δt = sampling_time)

fig = Figure(size = (600, 600))

ax1 = Axis(fig[1, 1]; xlabel = L"x_1", ylabel = L"x_2", xlabelsize = 30, ylabelsize = 30, title = L"a = 0.00006, b = 0.05", titlesize = 30)
lines!(ax1, Y1[:, 1], Y1[:, 2])

ax2 = Axis(fig[1, 2]; xlabel = L"x_1", ylabel = L"x_2", xlabelsize = 30, ylabelsize = 30, title = L"a = 0.02, b= ", titlesize = 30)
lines!(ax2, Y2[:,1], Y2[:, 2])

ax3 = Axis(fig[1, 3];  xlabel = L"x_1", ylabel = L"x_2", xlabelsize = 30, ylabelsize = 30,  title = L"a = 1.18, b = 0.5", titlesize = 30)
lines!(ax3, Y3[:, 1], Y3[:, 2])

# ax4 = Axis(fig[1, 4]; xlabel = L"x_4", ylabel = L"x_1", xlabelsize = 30, ylabelsize = 30)
# lines!(ax4, Y4[:, 1], Y4[:, 2])
root = joinpath(@__DIR__, "images", "chaoticAtractor_several_a.png")
save(root, fig)


# ----------------------------------------------------- #
# Interactive trajectory 3D (no sliders)--->??
# ----------------------------------------------------- #

u0 = [-1, -0.5, -0.5, -3] # Initial condition
p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.005, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u0 , p1, diffeq =integrationMethod)

u0s = [
    [-1, -0.5, -0.5, -3] ,
    [-1, -0.5, -0.5, -3] .+ 10^(-4),
    [-1, -0.5, -0.5, -3] .- 10^(-4)
]

lims = (
    (-3.0, 3.0),
    (-15.0, 15.0),
    (-15.0, 25.0),
)

# Define the figure evolution object
# idxs --> [1,2,3] for 3D plot of x1,x2,x3
# obs --> Information for each initial condition
# tsidxs = nothing (Does not plot the xi time evolution)
fig, obs, step, = interactive_evolution(
    syst_1, u0s; 
    tail = 10000, 
    add_controls = false, 
    idxs = [1, 2, 3],
    tsidxs = [1, 2, 3],
    # lims = lims,
    figure = (resolution = (1200, 700),),
)

fig

ax1 = content(fig[1,1][1,1])
ax1.xlabel = L"x_1"
ax1.ylabel = L"x_2"
ax1.zlabel = L"x_3"
ax1.xlabelsize = 30
ax1.ylabelsize = 30
ax1.zlabelsize = 30

ax2 = content(fig[1,2][1,1])
ax2.xlabel = L"t"
ax2.ylabel = L"x_1"
ax2.xlabelsize = 30
ax2.ylabelsize = 30

ax3 = content(fig[1,2][2,1])
ax3.xlabel = L"t"
ax3.ylabel = L"x_2"
ax3.xlabelsize = 30
ax3.ylabelsize = 30

ax3 = content(fig[1,2][3,1])
ax3.xlabel = L"t"
ax3.ylabel = L"x_3"
ax3.xlabelsize = 30
ax3.ylabelsize = 30

root = joinpath(@__DIR__, "images", "x1x2x3.mp4")
record(fig, root; framerate = 60) do io
    for i in 1:1000
        recordframe!(io)
        # Step multiple times per frame for "faster" animation
        for j in 1:30; step[] = 0; end
    end
end


# ----------------------------------------------------- #
# Interactive trajectory for 2D plots
# This part have been use for the 2D Movies
# ----------------------------------------------------- #
u0s = [
    [-1, -0.5, -0.5, -3] ,
    [-1, -0.5, -0.5, -3] .+ 10^(-4),
    [-1, -0.5, -0.5, -3] .- 10^(-4)
]

u0 = [-1, -0.5, -0.5, -3] # Initial condition
p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.005, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u0 , p1, diffeq =integrationMethod)

lims1 = (
    (-4.0, 4.0),
    (-20.0, 20.0),
)

# Periodic --> a = 0.04, b = 0.25; 0.06, 0.37
a1, a2, a3 = 0.00006, 0.008, 1.18
a1 = 0.75
set_parameter!(syst_1, 1, a1)
set_parameter!(syst_1, 2, 0.2)
N = 10^5 # takes around 7 sec
# lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)
fig1, obs1, step1, = interactive_evolution(
    syst_1, u0s; 
    tail = 10000, 
    add_controls = false, 
    idxs = [1, 2],
    # tsidxs = nothing,
    lims = lims1,
    figure = (resolution = (1200, 600),),
)

fig1
ax1 = content(fig1[1,1][1,1])
ax1.xlabel = L"x_1"
ax1.ylabel = L"x_2"
ax1.xlabelsize = 30
ax1.ylabelsize = 30
ax1.title = L"a = 0.75, b = 0.2"
ax1.titlesize = 30

ax2 = content(fig1[1,2][1,1])
ax2.xlabel = L"t"
ax2.ylabel = L"x_1"
ax2.xlabelsize = 30
ax2.ylabelsize = 30

ax3 = content(fig1[1,2][2,1])
ax3.xlabel = L"t"
ax3.ylabel = L"x_2"
ax3.xlabelsize = 30
ax3.ylabelsize = 30

root = joinpath(@__DIR__, "images", "PROBANDO4_x1x2_a3b3.mp4")
record(fig1, root; framerate = 60) do io
    for i in 1:3000
        recordframe!(io)
        # Step multiple times per frame for "faster" animation
        for j in 1:30; step1[] = 0; end
    end
end

# ----------------------------------------------------- #
# Interactive trajectory with Sliders 
# ----------------------------------------------------- #
# For the sliders
# a, b, c, d, e, α, β = p
ps = Dict(
    1 => 0.01:0.01:1.3,
    2 => 0.01:0.01:0.6,
    3 => 0.01:0.01:1.0,
    4 => 1:0.01:15.0,
    5 => 0.01:0.01:5.0,
    6 => 0.01:0.01:2.0,
    7 => 0.01:0.01:2.0
)
pnames = Dict(1 => "a", 2 => "b", 3 => "c", 4 => "d", 5 => "e", 6 => "α", 7 => "β" )

# Limits for the 3D plot and corresponding 2D plot
lims = (
    (-3.0, 3.0),
    (-15.0, 15.0),
    (-15.0, 25.0),
)

# Set of Initial Conditions
u0s = [
    [-1, -0.5, -0.5, -3] ,
    [-1, -0.5, -0.5, -3] .+ 10^(-5),
    [-1, -0.5, -0.5, -3] .- 10^(-5)
]

figure, obs, step, paramvals = interactive_evolution(
    syst_1, u0s; 
    ps = ps, 
    tail = 1000, 
    pnames = pnames, 
    lims = lims,
    idxs = [1, 2, 3],
    figure = (resolution = (1200, 600),),
)

# ----------------------------------------------------- #
# Obtain the lyapunovspectrum for u0
# ----------------------------------------------------- #
N = 10^4

as = 0.01:0.0001:1.2
as= 0.01:0.5:1.2
collect(as)
λs = zeros(length(as), 4)

for (i, a) in enumerate(as)
    @show i
    set_parameter!(syst_1, 1, a)
    λs[i, :] .= lyapunovspectrum(syst_1, 10000; Ttr = 500)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "a", ylabel = "λ")
lines!(ax, as, λs[:, 1], label = "λ1")
lines!(ax, as, λs[:, 2], label = "λ2")
lines!(ax, as, λs[:, 3], label = "λ3")
lines!(ax, as, λs[:, 4], label = "λ4")

# ----------------------------------------------------- #
# Bifurcation Diagram?? --> Dont seems to work
# ----------------------------------------------------- #
using NLsolve
function mem_systemv2(du, u)
    x, y, z, v = u
    b, c, d, e, α, β = [0.5, 0.5, 10, 4, 0.1, 1]

    du[1] = -(av2*y^2 - b)*x - (α*v + β)*z
    du[2] = -c*x - d*y + e*y*x^2
    du[3] = x
    du[4] = z

    return nothing
end

u0 = [-1, -0.5, -0.5, -3] # Initial condition
av2 = 0.1
sol = nlsolve(mem_systemv2, u0)
sol.zero


# Using the "produce_orbitdiagram". Not really working :(
plane = 3
coord = 4
p_index = 1
pvalues = 0.01:0.01:1.2
output = produce_orbitdiagram(syst_1, plane, coord,
                     p_index, pvalues)


fig = Figure()
ax1 = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
for (j, p) in enumerate(pvalues)
    l = length(output[j])
    v = fill(p, l)
    scatter!(ax1, v, output[j], color = "black")
end
# ----------------------------------------------------- #
# Poincare
# ----------------------------------------------------- #
u0s = [
    [-1, -0.5, -0.5, -3] ,
    [-1, -0.5, -0.5, -3] .+ 10^(-9),
    [-1, -0.5, -0.5, -3] .- 10^(-9)
]


# SVector(1,2,4) --> chose the 3 variables (x1,x2,x3,x4)
trs = [trajectory(syst_1, 10000, u0)[1][:, SVector(1,2,3)] for u0 ∈ u0s]
for i in 2:length(u0s)
    append!(trs[1], trs[i])
end
# Dimension of the plane for scanning
j = 3

brainscan_poincaresos(trs, j; linekw = (transparency = true,))