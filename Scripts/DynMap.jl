using DynamicalSystems
using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using DelimitedFiles


# ------------------ Funciones Auxiliares ------------------ #
function clasificar_caos(exponentes_lyapunov)
    # Contar los exponentes positivos, 0 y negativos
    # cant_positivos = count(x -> x >= 10^(-3), exponentes_lyapunov)
    # cant_ceros = count(x -> (x < 10^(-3)) & (x > -10^(-3)), exponentes_lyapunov)

    # cant_positivos = count(x -> x >= 10^(-5), exponentes_lyapunov)
    # cant_ceros = count(x -> (x < 10^(-5)) & (x > -10^(-5)), exponentes_lyapunov)

    cant_positivos = count(x -> x >= 10^(-4), exponentes_lyapunov)
    cant_ceros = count(x -> (x < 10^(-4)) & (x > -10^(-4)), exponentes_lyapunov)

    # print("pos:", cant_positivos,"\n")
    # print("zeros:", cant_ceros,"\n")
    if cant_positivos == 0
        if cant_ceros == 0
            # Punto fijo: Todos los exponentes son negativos.
            return 1
        elseif cant_ceros == 1
            # Periodicidad: Un exponente es 0 y los demás son negativos.
            return 2
        elseif cant_ceros == 2
            # Periodicidad: Dos exponentes son 0 y el resto son negativos.
            return 2
        elseif cant_ceros == 3
            # 3-toro: Tres exponentes son 0 y el resto son negativos.
            return 3
        end
    elseif cant_positivos == 1 
        # Caos normal: Un exponente positivo, uno 0 y el resto negativos.
        return 4
    elseif cant_positivos == 2 
        # Hipercaos: Dos exponentes positivos, uno 0 y el resto negativos.
        return 5
    elseif cant_ceros == 2 
        # Quasiperiodicidad (2-toro): Dos exponentes 0 y el resto son negativos.
        return 6
    else
        # Comportamiento no clasificado.
        return 0
    end
end

function asignar_color(numero)
    if numero == 1
        # Punto fijo: Todos los exponentes son negativos. (Rojo oscuro)
        return RGB(139, 0, 0) / 255
    elseif numero == 2
        # Periodicidad: Un exponente es 0 y los demás son negativos. (Amarillo)
        return RGB(255, 255, 0) / 255
    elseif numero == 3
        # 3-toro: Tres exponentes son 0 y el resto son negativos. (Verde oscuro)
        return RGB(0, 100, 0) / 255
    elseif numero == 4
        # Caos normal: Un exponente positivo, uno 0 y el resto negativos. (Azul claro)
        return RGB(173, 216, 230) / 255
    elseif numero == 5
        # Hipercaos: Dos exponentes positivos, uno 0 y el resto negativos. (Violeta)
        return RGB(148, 87, 235) / 255
    elseif numero == 6
        # Quasiperiodicidad (2-toro): Dos exponentes 0 y el resto son negativos. (Naranja)
        return RGB(255, 165, 0) / 255
    else
        # Comportamiento no clasificado. (Gris)
        return RGB(169, 169, 169) / 255
    end
end

function f_jac(J,u,p,t)
    J[1,1] = 2.0 - 1.2 * u[2]
    J[1,2] = -1.2 * u[1]
    J[2,1] = 1 * u[2]
    J[2,2] = -3 + u[1]
    nothing
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

# ------------------ Definiendo el sistema ------------------ #
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

# ------------------ Determinando la precision de los exponentes ------------------ #
# Con estos parametros, se ha comprobado que el sistema no es caótico.
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

# ------------------ Calculando el mapa dinámico ------------------ #
root = joinpath(@__DIR__, "DynMap2_v7.txt") # Fichero donde se guarda el mapa
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
close(file) # Importante cerrar el archivo al terminar
end

# ------------------ Leer el mapa dinámico como matriz ------------------ #
root = joinpath(@__DIR__, "DynMap2_v7.txt")

contenido = readlines(root) # Lee todo el archivo
M = []
# Procesa cada línea del archivo
for linea in contenido
    # Divide la línea usando "\t" y " " como delimitadores
    numeros = split(linea, ['\t', ' '])
    
    # Convierte los elementos de la cadena a números y los agrega a la matriz
    push!(M, parse.(Int, filter(x -> x ≠ "", numeros)))
end
M[1]

M= hcat(M...)
println(M) # Muestra la matriz resultante
M[1,:]
valores_unicos = unique(M)

# Crea el mapa de calor utilizando la función heatmap
fig, ax, hm = heatmap(aArr, bArr, M)
Colorbar(fig[:, end+1], hm)

ax.xlabel = L"a"
ax.ylabel = L"b"
ax.xlabelsize = 30
ax.ylabelsize = 30

root = joinpath(@__DIR__, "images", "DynMap.png")
save(root, fig)

fig


