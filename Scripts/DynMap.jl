using DynamicalSystems
using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using DelimitedFiles


# ------------------ Auxiliary Functions ------------------ #
function clasificar_caos(exponentes_lyapunov)
    # Count positive, zero, and negative exponents
    cant_positivos = count(x -> x >= 10^(-4), exponentes_lyapunov)
    cant_ceros = count(x -> (x < 10^(-4)) & (x > -10^(-4)), exponentes_lyapunov)

    # print("pos:", cant_positivos,"\n")
    # print("zeros:", cant_ceros,"\n")
    if cant_positivos == 0
        if cant_ceros == 0
            # Fixed point: All exponents are negative.
            return 1
        elseif cant_ceros == 1
            # Periodicity: One exponent is 0 and the others are negative.
            return 2
        elseif cant_ceros == 2
            # Periodicity: Two exponents are 0 and the rest are negative.
            return 2
        elseif cant_ceros == 3
            # 3-torus: Three exponents are 0 and the rest are negative.
            return 3
        end
    elseif cant_positivos == 1 
        # Normal chaos: One positive exponent, one 0, and the rest negative.
        return 4
    elseif cant_positivos == 2 
         # Hyperchaos: Two positive exponents, one 0, and the rest negative.
        return 5
    elseif cant_ceros == 2 
        # Quasiperiodicity (2-torus): Two 0 exponents and the rest are negative.
        return 6
    else
        # Unclassified behavior.
        return 0
    end
end

function mem_system(du, u, p, t)
    x, y, z, v = u
    a, b, c, d, e, α, β = p

    du[1] = -(a*y^2 - b)*x - (α*v + β)*z
    du[2] = -c*x - d*y + e*y*x^2
    du[3] = x
    du[4] = z

    return nothing
end

# ------------------ Defining the System ------------------ #
u0 = [-1, -0.5, -0.5, -3] # Initial condition
p1 = [0.1, 0.5, 0.5, 10, 4, 0.1, 1] #  a, b, c, d, e, α, β
integrationMethod = (alg = Tsit5(), adaptive = true, dense = true, dt = 0.005, reltol=1e-8, abstol=1e-8)
syst_1 = CoupledODEs(mem_system, u0 , p1, diffeq =integrationMethod)

N = 10^4
lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)
clasificar_caos([1,-10^(-3), 10^(-3), -4])
clasificar_caos(lya_spec)

set_parameter!(syst_1, 2, 0.297)
set_parameter!(syst_1, 1, 0.001)
lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)

# ------------------ Determining the Precision of Exponents ------------------ #
# With these parameters, the system is known to be non chaotic
set_parameter!(syst_1, 2, 0.5)
set_parameter!(syst_1, 1, 1.18)

N = 10^3 # takes around 1 sec
lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)
"""
0.00140218241198051
-0.002873385113572653
-0.17465281757265116
-3.5326169465015815
"""

N = 10^4 # takes around 1 sec
lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)
"""
0.0002492804423395031
-0.0004593363327491549
-0.29125012867831174
-0.7000245016516771
"""

N = 10^5 # takes around 7 sec
lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100, show_progress = true)
"""
3.0439560808501764e-5
-7.023705814719427e-5
-0.03964506558055715
-0.08068184454803742
"""

# ------------------ Calculating the Dynamic Map ------------------ #
root = joinpath(@__DIR__, "DynMap2_v7.txt") # File to save the map
file = open(root, "w")
collect(0.001:0.003:0.601)
collect(0.001:0.006:1.201)
201*201* 1/3600

bArr = reverse([x for x in 0.01:0.01:0.60])
aArr = [x for x in 0.01:0.01:1.20]
N = 10^5 

if true
for b in bArr
    set_parameter!(syst_1, 2, b)
    print("------------- b = $b -------------\n")
    print("(a, num) = ")
    for a in aArr
        set_parameter!(syst_1, 1, a)

        lya_spec = lyapunovspectrum(syst_1, N; Ttr = 100)
        # print(lya_spec)
        num = clasificar_caos(lya_spec)
        print("($a, $num), ")
        
        write(file, "$num\t")
    end
    write(file, "\n")
end
close(file) 
end

# ------------------ Read the Dynamical Map as a Matrix ------------------ #
root = joinpath(@__DIR__, "DynMap2_v7.txt")

contenido = readlines(root) 
M = []

# Process every line 
for linea in contenido
    # Divide tje line using "\t" and " " 
    numeros = split(linea, ['\t', ' '])
    
    # Turn them into Int vairbles
    push!(M, parse.(Int, filter(x -> x ≠ "", numeros)))
end

# M[1]
M= hcat(M...)
# println(M) # Muestra la matriz resultante
# M[1,:]
valores_unicos = unique(M)

# Creates a heatmap
fig, ax, hm = heatmap(aArr, bArr, M)
Colorbar(fig[:, end+1], hm)

ax.xlabel = L"a"
ax.ylabel = L"b"
ax.xlabelsize = 30
ax.ylabelsize = 30

root = joinpath(@__DIR__, "images", "DynMap.png")
save(root, fig)

fig


