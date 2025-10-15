#using QuantumOptics
#import QuantumOptics:destroy, ⊗
using FFTW
using QuantumToolbox
using QuantumToolbox:QuantumObjectEvolution as QObjEvo
using QuantumToolbox:QuantumObject as QObj
using PyPlot
using Interpolations
#using Interpolations:interpolate, extrapolate, Bspline
#using DataInterpolations
using Random: Xoshiro, rand, TaskLocalRNG
using NumericalIntegration
using LinearAlgebra
using LinearAlgebra: blockdiag

A = blockdiag([1 2; 3 4], [5 6; 7 8])
println(A)

#using SparseArrays
#using LinearAlgebra:blockdiag as blockdiag

#=
I will store the SLH's as NamedTuples, G = (S = 1, L = 1, H = 1).
S, L and H are matrices, but the entries of those matrices are QObj's

=#

PyPlot.matplotlib.use("tkagg")

code_in = [0, 1, 0, 1]
N = length(code_in) # number of photons, equal to the number of dual-rail qubits
# note: fock(3, 0) = 1|0> + 0|1> + 0|2>

# given operator A, create operator I ⊗ I ⊗ ... ⊗ A ⊗ ... ⊗ I ⊗ I
# with A at position j, and n in total
function embed(A, j, n)
    res = reduce(⊗, [k == j ? A : qeye(2) for k in 1:n])
    return res
end


des = destroy(2)
des2 = qeye(1)
des2 = des2 ⊗ qeye(2)
des2 = des2 ⊗ des
des2 = des2 ⊗ qeye(2)
des2 = des2 ⊗ qeye(2)

as = [embed(des, j, 2*N) for j in 1:2*N] # array of annie operators, one for each rail




# take a bitstring
# create the corresponding dual-rail state
function make_input(code)
    ψ = QuantumObject([1])
    for c in code
        if c == 0
            ψ = ψ ⊗ fock(2, 1) ⊗ fock(2, 0);
        elseif c == 1
            ψ = ψ ⊗ fock(2, 0) ⊗ fock(2, 1);
        else
            error("code must be a bitstring")
        end
    end
    return ψ; 

end


#take two QuantumObject's that correspond to matrices
#take their direct sum, then return the corresponding QObj
function ⊕(M1, M2)
    #Zu = 
    return QObj(blockdiag(Matrix(M1.data), Matrix(M2.data)))
end


function concatenate(G1, G2)
    S = G1.S ⊕ G2.S
    L = QObj(vcat(G1.L.data, G2.L.data)) # vertical "stacking"
    H = G1.H + G2.H
    return (S = S, L = L, H = H)
end

function renumber(routes)
    ins, outs = routes
    N = length(ins)
    for j = 1:N
        in = ins[j]
        dec_in = findall(x -> x > in, ins[j+1:N])
        dec_in .+= j
        ins[dec_in] .-= 1
        out = outs[j]
        dec_out = findall(x -> x > out, outs[j+1:N])
        dec_out .+= j
        outs[dec_out] .-= 1
        # .-= is wonderful notation. a -= b means a = a - b
        # a .- b means a - b*ones(length(a)); that is, elementwise subtraction of scalar from vector (matlab style)
        # apparently you can combine them!
    end
    return [ins, outs]
end

# G1 and G2 are SLH triples (NamedTuples)
# routes is a vector, and each element has two entries, both ints that correspond to outputs and inputs, respectively
# e.g. [[3, 1], [1, 4]] means "route output 3 into input 1, and output 1 into input 4
# 
function feedback_reduce(G, routes)
    routes = renumber(routes)
    ins, outs = routes
    S, L, H = G.S, G.L, G.H

    #in notation: x is output, y is input
    for j in eachindex(ins)
        in = ins[j]
        out = outs[j]
        Sxbyb = S[[1:out-1; out+1:end], [1:in-1; in+1:end]]
        Sxby = S[[1:out-1; out+1:end], in]
        Sxy = S[out, in]
        Sxyb = S[out, [1:in-1; in+1:end]]
        I = qeye(1)
        println("Sxy: ", Sxy)
        println("Sxy - 1: ", Sxy .- 1)
        S_red = Sxbyb .+ Sxby .* QObj(inv(1 .- reshape(Sxy.data, 1, 1))) .* Sxyb # dividing by operators??

        Lx = L[out, :]
        Lxb = L[[1:out-1; out+1:end], :]
        L_red = Lxb .+ Sxby.*inv(1 .- reshape(Sxy.data, 1, 1)) .* Lx

        Sy = S[:, in]

        H_red = H - 0.5im*(L'*Sy*inv(I-Sxy)*Lx - Lx'*inv(I-Sxy')*Sy'*L)


        return (S = S_red, L = L_red, H = H_red)


    end

end



I2 = qeye(2)
I3 = qeye(3)


input = make_input(code_in);

#SLH triples

# balanced beamsplitter

H0 = 0*embed(qeye(2), 1, 2*N)
I_2N = embed(qeye(2), 1, 2*N)



lem = [QObj([1]) QObj([1]); QObj([1]) QObj([-1])]

G_BS = (S = 1/sqrt(2)*lem, L = [QObj([0]); QObj([0])], H = QObj([0]))
G_mir = (S = -I_2N, L = I_2N, H = H0)
# reshape(A, 1, 1) takes the singleton matrix A = [x] and forces it to be a 1x1 matrix


G_cav_kerr = (S = I_2N, L = as[2], H = as[2]'*as[2] + as[2]'*as[2]*as[2]'*as[2])

H2 = as[2]'*as[2]*as[2]'*as[2]
println("H0: ", H0.dims)
println("H2: ", H2.dims)
Hs = H0+H2
println("Hs: ", Hs.dims)



G_tot = concatenate(G_cav_kerr, G_mir)

G_red = feedback_reduce(G_tot, [[1], [2]])