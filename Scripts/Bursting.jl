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


u01 = [0.0, 2.0, 1.0, 0.0] # Initial condition
u02 = [2.0, -1.0, 1.0, 2.0] # Initial condition
p1 = [0.1, 0.4, 0.2, 0.1, 4.0, 0.1, 3.0] #  a, b, c, d, e, α, β
integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.005, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u01 , p1, diffeq =integrationMethod)
syst_2 = CoupledODEs(mem_system, u02 , p1, diffeq =integrationMethod)

total_time = 300
sampling_time = 0.02
transient_time = 100
Y1, t1 = trajectory(syst_1, total_time; Ttr = transient_time , Δt = sampling_time)
Y2, t2 = trajectory(syst_2, total_time; Ttr = transient_time , Δt = sampling_time)

N = 2*10^4
lya_spec1 = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)
lya_spec2 = lyapunovspectrum(syst_2, N; Ttr = 100, show_progress = true) 

# --------------------- PLOTS --------------------- #
fig = Figure()
# Time series x
ax1 = Axis(fig[1, 1]; xlabel = L"t", ylabel = L"x_1", xlabelsize = 30, ylabelsize = 30)
lines!(ax1, t1, Y1[:, 1], label = L"\vec{x}_0 = \vec{x}_1", color =RGBf(0.4, 0.1, 0.9))
lines!(ax1, t2, Y2[:, 1], label = L"\vec{x}_0 = \vec{x}_2", color =RGBf(0.2, 0.5, 0.7))
axislegend(ax1, labelsize = 20)
# Time series y
ax2 = Axis(fig[1, 2]; xlabel = L"t", ylabel = L"x_2", xlabelsize = 30, ylabelsize = 30)
lines!(ax2, t1, Y1[:, 2], label = L"\vec{x}_0 = \vec{x}_1", color =RGBf(0.4, 0.1, 0.9))
lines!(ax2, t2, Y2[:, 2], label = L"\vec{x}_0 = \vec{x}_2", color =RGBf(0.2, 0.5, 0.7))
axislegend(ax2, labelsize = 20)
# Proyection x-y
ax3 = Axis(fig[1, 3]; xlabel = L"x_1", ylabel = L"x_2", xlabelsize = 30, ylabelsize = 30)
lines!(ax3, Y1[:, 1], Y1[:, 2], label = L"\vec{x}_0 = \vec{x}_1", color =RGBf(0.4, 0.1, 0.9))
lines!(ax3, Y2[:, 1], Y2[:, 2], label = L"\vec{x}_0 = \vec{x}_2", color =RGBf(0.2, 0.5, 0.7))
axislegend(ax3, labelsize = 20)

root = joinpath(@__DIR__, "images", "burstSeries.png")
save(root, fig)

fig = Figure()
# Time series x
ax1 = Axis(fig[1, 1]; xlabel = "\$x_1\$", ylabel = "\$x_2\$")
lines!(ax1, t1, Y1[:, 1])
lines!(ax1, t2, Y2[:, 1])
# Time series y
ax2 = Axis(fig[1, 2]; xlabel = "y", ylabel = "z")
lines!(ax2, t1, Y1[:, 2])
lines!(ax2, t2, Y2[:, 2])
# Time series u
ax3 = Axis(fig[1, 3]; xlabel = "x", ylabel = "y")
lines!(ax3, t1, Y1[:, 4])
lines!(ax3, t1, Y2[:, 4])

# ----------------------------------------------------- #
# Interactive trajectory for 2D plots
# ----------------------------------------------------- #
u0s = [
    [0.0, 2.0, 1.0, 0.0]  ,
    [2.0, -1.0, 1.0, 2.0],
]

lims1 = (
    (-1.5, 1.5),
    (-15.0, 12.0),
)

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

root = joinpath(@__DIR__, "images", "bursting_t0.mp4")
record(fig1, root; framerate = 60) do io
    for i in 1:2000
        recordframe!(io)
        # Step multiple times per frame for "faster" animation
        for j in 1:15; step1[] = 0; end
    end
end

# ----------------------------------------------------- #
# Interactive trajectory for 3D plots
# ----------------------------------------------------- #
u0s = [
    [0.0, 2.0, 1.0, 0.0]  ,
    [2.0, -1.0, 1.0, 2.0],
]

lims1 = (
    (-1.5, 1.5),
    (-15.0, 12.0),
    (-15.0, 12.0),
)

fig2, obs1, step1, = interactive_evolution(
    syst_1, u0s; 
    tail = 10000, 
    add_controls = false, 
    idxs = [4, 2, 1],
    # tsidxs = nothing,
    lims = lims1,
    figure = (resolution = (1200, 600),),
)

fig2

root = joinpath(@__DIR__, "images", "bursting3D_124.mp4")
record(fig2, root; framerate = 60) do io
    for i in 1:1000
        recordframe!(io)
        # Step multiple times per frame for "faster" animation
        for j in 1:30; step1[] = 0; end
    end
end