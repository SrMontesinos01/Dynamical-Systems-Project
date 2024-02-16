using DifferentialEquations, DynamicalSystems
using GLMakie
using LinearAlgebra: norm, dot
using OrdinaryDiffEq
using InteractiveDynamics

import Plots

using LinearAlgebra, Polynomials


function findlocalmaxima(t, u::Vector{T}, dudt::Vector{T}) where T <: Real
    """
    https://discourse.julialang.org/t/how-to-find-the-local-extrema-of-the-solutions-when-solving-a-differential-equation/94628/14

    findlocalmaxima(t, u::Vector{T}, dudt::Vector{T}) where T <: Real

    Compute and return the local maxima of a time series `u`. Inputs are discrete 
    samples of continuous variables with `u[n]` = u(`t[n]`) and similarly for dudt.
    Outputs are vectors tmax, umax specifying the t and u values of local maxima of u(t).


    Examples:
    t = range(-5pi, 5pi, 64)
    u = cos.(t)
    dudt = -sin.(t)
    tmax, umax = findlocalmaxima(t,u,dudt)
    
    """
    N = length(t)
    @assert length(u) == N
    @assert length(dudt) == N
    
    # find indices of du/dt zero crossings downwards, indicating local maxima of u
    signchange = [false; [dudt[i] >= 0 && dudt[i+1] < 0 for i in 1:N-1]]
    maxlocations = findall(signchange)

    Nmaxima = length(maxlocations)
    umax = zeros(Nmaxima)  # u values for local maxima of u(t)
    tmax = zeros(Nmaxima)  # t values for local maxima of u(t)
    s = [-1; 0; 1]  # auxiliary time variable for nbrhd of local max, s = (t-t[n])/Δt
    
    # print("todo bien")
    for i in 1:(Nmaxima - 1)
        n = maxlocations[i]           # index of local maximum of u data
        upoly = fit(s, u[n .+ s])     # fit local polynomial u(s) to s,u data, order set by length(s)
        tpoly = fit(s, t[n .+ s])     # fit local polynomail t(s) to s,t data (overkill for uniform time steps)
        dudspoly = derivative(upoly)  # compute du/ds(s)
        smax = roots(dudspoly)[1]     # solve du/ds(s) = 0 for s
        umax[i] = upoly(smax)         # evaluate polynomial u(s)
        tmax[i] = tpoly(smax)         # evaluate polynomial t(s)
    end
    
    return tmax, umax
end 

function uniqueValues(v1::Vector{T}, v2::Vector{S}, bound::Real) where {T<:Real, S}
    # Filter the values
    valores_unicos = [v1[i] for i in 1:length(v1) if count(x -> abs(x - v1[i]) < bound, v1) == 1]
    
    # Take the values of the other vector
    pares_valores = [(v2[i], v1[i]) for i in 1:length(v1) if v1[i] in valores_unicos]
    
    return pares_valores
end

# ------------------- system and time series ------------------- #
function mem_system(du, u, p, t)
    x, y, z, v = u
    a, b, c, d, e, α, β = p

    du[1] = -(a*y^2 - b)*x - (α*v + β)*z
    du[2] = -c*x - d*y + e*y*x^2
    du[3] = x
    du[4] = z

    return nothing
end

u0 = [-1, -0.5, -0.5, -3] # Initial condition
# p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
# p1 = [0.2, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
# p1 = [0.8, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.001, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u0 , p1, diffeq =integrationMethod)

total_time = 300
sampling_time = 0.01
transient_time = 100
Y, t = trajectory(syst_1, total_time; Ttr = transient_time , Δt = sampling_time)

# Analyze the time series and search for minimum/maximums
u = Y[:,1]
t = t
a, b, c, d, e, α, β = p1
dudt = -(a.*Y[:,2].^2 .- b).*Y[:,1] - (α.*Y[:,4] .+ β).*Y[:,3]

tmax, umax = findlocalmaxima(t, u, dudt)
tmin, umin = findlocalmaxima(t, -u, -dudt)
umin = -umin
size(umax)
size(umin)

# tmaxUniq, umaxUniq = uniqueValues(umax, tmax, 0.00001)
# tminUniq, uminUniq = uniqueValues(umin, tmin, 0.00001)

# Time series x
fig = Figure()
ax1 = Axis(fig[1, 1]; xlabel = L"t", ylabel = L"x_1(t)", xlabelsize = 30, ylabelsize = 30, 
            title = L"a = 0.1", titlesize = 30, limits = (230, 350, -3, 3))
lines!(ax1, t, Y[:, 1], label = L"x_1(t)")
scatter!(ax1, tmax, umax, color =:darkorange, label = L"\text{max}[x_1]")
# scatter!(ax1, tmin, umin)
axislegend(ax1, labelsize = 20)
fig
root = joinpath(@__DIR__, "images", "xmax_a01.png")
save(root, fig)

# Bifurcation Scatter (for a single value of the parameter)
fig = Figure()
ax1 = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
scatter!(ax1, ones(length(umax))*α, umax)
scatter!(ax1, ones(length(umin))*α, umin)

# ----------------------- Complete Bifurcation Diagram  ---------------------- #
aArr = [x for x in 0.001:0.001:1.201]
total = 300
Δt = 0.01
wait = 100

u0 = [-1, -0.5, -0.5, -3] # Initial condition
p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
a, b, c, d, e, α, β = p1

integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.001, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u0 , p1, diffeq =integrationMethod)

maxArrList = Vector{Vector{Float64}}()
minArrList = Vector{Vector{Float64}}()
for a in aArr
    # @show a
    set_parameter!(syst_1, 1, a)
    Y, t = trajectory(syst_1, total; Ttr = wait , Δt = Δt)
    u = Y[:,1]

    dudt = -(a.*Y[:,2].^2 .- b).*Y[:,1] - (α.*Y[:,4] .+ β).*Y[:,3]
    tmax, umax = findlocalmaxima(t, u, dudt)
    tmin, umin = findlocalmaxima(t, -u, -dudt)
    umin = -umin

    # Eliminates x = 0
    filter!(x -> x != 0, umax)
    filter!(x -> x != 0, umin)

    push!(maxArrList, umax)
    push!(minArrList, umin)
end

# Bifurcation Scatter maxima
fig = Figure()
ax1 = Axis(fig[1, 1]; xlabel = L"a", ylabel = L"\text{max}[x_1]", xlabelsize = 30, ylabelsize = 30)
for (i, a) in enumerate(aArr)
    # print(a)
    scatter!(ax1, ones(length(maxArrList[i]))*a, maxArrList[i], color =:darkorange, markersize = 0.05)
end

root = joinpath(@__DIR__, "images", "BifurcationMax.png")
save(root, fig)
# Bifurcation Scatter minima
fig = Figure()
ax1 = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
for (i, a) in enumerate(aArr)
    # print(a)
    scatter!(ax1, ones(length(minArrList[i]))*a, minArrList[i], color =:orange, markersize = 0.05)
end

