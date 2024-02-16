using  Polynomials

# Solving the characterics equation for the eigenvalues of the Jacobian
x1, x2, x3, x4  = -1, -0.5, -0.5, -3 # Initial condition
a, b, c, d, e, α, β = 0.1, 0.5, 0.5, 10, 4, 0.1, 1 #  a, b, c, d, e, α, β

p = Polynomial([ d*(α*x4 + β), (α*x4+β-b*d) , (d-b), 1.0])
sol = real.(filter(isReal, roots(p)))[1]
sol = roots(p)

# ------------------ Lyapunov Espectrum ----------------- #
using DifferentialEquations, DynamicalSystems
using GLMakie
using LinearAlgebra: norm, dot
using OrdinaryDiffEq
using InteractiveDynamics
using DelimitedFiles

function mem_system(du, u, p, t)
    x, y, z, v = u
    a, b, c, d, e, α, β = p

    du[1] = -(a*y^2 - b)*x - (α*v + β)*z
    du[2] = -c*x - d*y + e*y*x^2
    du[3] = x
    du[4] = z

    return nothing
end

u0 = [-1.0, -0.5, -0.5, -3.0] # Initial condition
p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.005, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u0 , p1, diffeq =integrationMethod)

root = joinpath(@__DIR__, "LyaSpec_v2.txt") # Fichero donde se guarda el mapa
file = open(root, "w")

as = 0.001:0.001:1.2
# as= 0.01:0.5:1.2
collect(as)
λs = zeros(length(as), 4)

N = 10^5
if true
for (i, a) in enumerate(as)
    @show i
    set_parameter!(syst_1, 1, a)
    λs[i, :] .= lyapunovspectrum(syst_1, N; Ttr = 100)

    write(file, "$a")
    for num in λs[i, :]
        write(file, "\t$num")
    end
    write(file, "\n")
end
close(file) # Important
end

# ---- Reading and Ploting --- #
root = joinpath(@__DIR__, "LyaSpec_v1.txt") # 
xtData = readdlm(root, '\t', Float64) # r

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"a", ylabel = "Lyapunov Exponents", xlabelsize = 30, ylabelsize = 30)
lines!(ax, xtData[:, 1], xtData[:, 2], label = L"LE_1", color=:blue)
lines!(ax, xtData[:, 1], xtData[:, 3], label = L"LE_2", color =:red)
lines!(ax, xtData[:, 1], xtData[:, 4], label = L"LE_3", color=:green)
# lines!(ax, xtData[:, 1], xtData[:, 5], label = L"LE_4")
axislegend(ax, labelsize = 20)

root = joinpath(@__DIR__, "images", "LyaSpec.png")
save(root, fig)