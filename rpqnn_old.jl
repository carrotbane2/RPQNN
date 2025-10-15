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
#using SparseArrays
#using LinearAlgebra:blockdiag as blockdiag

#=
I will store the SLH's as NamedTuples, G = (S = 1, L = 1, H = 1).
S, L and H are matrices, but the entries of those matrices are QObj's

=#

PyPlot.matplotlib.use("tkagg")

code_in = [1, 0]
N = length(code_in) # number of photons, equal to the number of dual-rail qubits
cutoff = 3 #dimension of Hilbert space for one mode
# the dimension of the total Hilbert space is going to be cutoff^(2N) (2N being the number of rails and cavities)
# note: fock(3, 0) = 1|0> + 0|1> + 0|2>

# given operator A, create operator I ⊗ I ⊗ ... ⊗ A ⊗ ... ⊗ I ⊗ I
# with A at position j
function embed(A, j, N, cutoff)
    res = reduce(⊗, [k == j ? A : qeye(cutoff) for k in 1:2*N])
    return QObj(res.data) # flatten
end


des = destroy(cutoff)

as = [embed(des, j, N, cutoff) for j in 1:2*N] # array of annie operators, one for each rail




# take a bit array
# create the corresponding dual-rail state
function dual_rail(code, cutoff)
    ψ = QuantumObject([1])
    for c in code
        if c == 0
            ψ = ψ ⊗ fock(cutoff, 1) ⊗ fock(cutoff, 0);
        elseif c == 1
            ψ = ψ ⊗ fock(cutoff, 0) ⊗ fock(cutoff, 1);
        else
            error("code must be a bitstring")
        end
    end
    return QObj(ψ.data); # flattens out the dimensions, e.g. [2, 2] -> [4]
end

function get_matrix(obj)
    if typeof(obj) == QObj
        return obj.data
    elseif typeof(obj) == QObjEvo
        return obj.data.L.A
    end
end

# take two matrices
# take their direct sum, then return the corresponding QObj
# return the terms that make up a direct sum
function ⊕(M1, M2)
    M1u = hcat(M1, zeros(size(M1)[1], size(M2)[2]))
    M1l = zeros(size(M2)[1], size(M1)[2] + size(M2)[2])
    M1_new = vcat(M1u, M1l)
    M2u = zeros(size(M1)[1], size(M1)[2] + size(M2)[2])
    M2l = hcat(zeros(size(M2)[1], size(M1)[2]), M2)
    M2_new = vcat(M2u, M2l)
    return [M1_new, M2_new]
end


function concatenate(G1, G2)
    S1n, S2n = G1.S.data.L ⊕ G2.S.data.L
    S1 = QObjEvo(G1.S.data.λ * S1n)
    S2 = QObjEvo(G2.S.data.λ * S2n)
    S = S1 + S2

    L1 = G1.L.data.L
    L2 = G2.L.data.L
    L1 = QObjEvo(G1.L.data.λ * vcat(L1, zeros(size(L2)[1], size(L2)[2])))
    L2 = QObjEvo(G2.L.data.λ * vcat(zeros(size(L1), L2)))
    L = L1 + L2
    H = G1.H + G2.H
    return (S = S, L = L, H = H)
end

function concatenate_vec(Gs)
    return reduce(concatenate, Gs) # this is concatenate(concatenate(Gs[1], Gs[2]), Gs[3]) and so on
end

function renumber(routes)
    outs, ins = routes
    N = length(ins)
    for j = 1:N
        in = ins[j]
        dec_in = findall(x -> x > in, ins[j+1:N]) # find all subsequent routes that involve a later index
        dec_in .+= j
        ins[dec_in] .-= 1
        out = outs[j]
        dec_out = findall(x -> x > out, outs[j+1:N])
        dec_out .+= j
        outs[dec_out] = outs[dec_out] .- 1
        # .-= is wonderful notation. a -= b means a = a - b
        # a .- b means a - b*ones(length(a)); that is, elementwise subtraction of scalar from vector (matlab style)
        # apparently you can combine them!
    end
    return [outs, ins]
end

# G1 and G2 are SLH triples (NamedTuples)
# routes is a vector, and each element has two entries, both ints that correspond to outputs and inputs, respectively
# e.g. [[3, 1], [1, 4]] means "route output 3 into input 1, and output 1 into input 4
# 
function feedback_reduce(G, routes, dim)
    routes = renumber(routes)
    outs, ins = routes
    S, L, H = G.S.data.L, G.L.data.L, G.H.data.L


    #in notation: x is output, y is input
    for j in eachindex(ins)
        in = ins[j]
        out = outs[j]
        Sxbyb = S[[1:dim*(out-1); dim*out+1:end], [1:dim*(in-1); dim*in+1:end]]
        Sxby = S[[1:dim*(out-1); dim*out+1:end], dim*(in-1)+1:dim*in]
        Sxy = S[dim*(out-1)+1:dim*out, dim*(in-1)+1 : dim*in]
        Sxyb = S[dim*(out-1)+1:dim*out, [1:dim*(in-1); dim*in+1:end]]
        I = qeye(dim).data
        S_red = Sxbyb + Sxby * inv(I - Sxy) * Sxyb
        Lx = L[dim*(out-1)+1:dim*out, :]
        Lxb = L[[1:dim*(out-1); dim*out+1:end], :]
        L_red = Lxb + Sxby *inv(I - Sxy) * Lx

        Sy = S[:, dim*(in-1)+1:dim*in]

        H_red = H - 0.5im*(L'*Sy*inv(I-Sxy)*Lx - Lx'*inv(I-Sxy')*Sy'*L)
        
        S = S_red
        L = L_red
        H = H_red
    end
    if size(L, 1) > 0
        return (S = QObjEvo(G.S.λ * S), L = QObjEvo(G.L.λ * L), H = QObjEvo(G.H.λ * H))
    else # this corresponds to a closed system: it has no scattering or coupling
        return (S = nothing, L = nothing, H = QObjEvo(G.H.λ * H))
    end
end

# G may be a QObjEvo, or it may be an AddedOperator
# a trivial QObjEvo is just a QObj and the function "one"

χ3 = 1
dim = cutoff^(2*N)
I = qeye(dim)

κ(p, t) = (t > p.ti & t < p.tf) ? p.A : 0 # rect pulse
one(p, t) = 1
two(p, t) = 2

# doing .size on these things sometimes fails because AbstractMatrix doesn't have the field called "size"
# instead, call size(...)

G_BS = (S = 1/sqrt(2)*QObjEvo(QObj([1 0 (-1) 0; 0 1 0 (-1); 1 0 1 0; 0 1 0 1] ⊗ I.data), one), L = QObjEvo(QObj([0; 0; 0; 0 ]⊗I.data), one), H = QObjEvo(QObj([0]⊗ I), one))

# a beamsplitter interacts with four channels. I number them like this: right upper, right lower, left upper, left lower
G_mir = (S = QObjEvo(exp(1im*pi) * I, one), L = QObjEvo(QObj([0]) ⊗ I, one), H = QObjEvo( QObj([0]) ⊗ I, one)) # one input, one output
G_mirs = concatenate_vec([G_mir for _ in 1:2*N])
G_BS_pair = concatenate(G_BS, G_BS)
G_BS_pair = feedback_reduce(G_BS_pair, [[6 8 1 3], [2 4 5 7]], dim)
G_BS_odd = concatenate_vec([G_BS_pair for _ in 1:N])
G_cavs_kerr = [(S = I, L = QObjEvo(as[j], κ), H = QObjEvo(as[j]'*as[j], one) + χ3*as[j]'*as[j]'*as[j]*as[j]) for j in 1:2*N]
G_cavs_kerr = concatenate_vec(G_cavs_kerr)


Gt = concatenate(G_mirs, G_BS_odd)
rights = [i + 2*N for i in 1:4*N if mod(i-1, 2) == 0]
lefts = [i + 2*N for i in 1:4*N if mod(i-1, 2) == 1]
Gt = feedback_reduce(Gt, [[1:2*N; lefts], [rights; 1:2*N]], dim)
# two-way cascade of mirror and BS layer into each other

Gt = concatenate(Gt, G_BS_even)
Gt = feedback_reduce(Gt, [[1:2*N; lefts], [rights; 1:2*N]])

Gt = concatenate(Gt, G_BS_odd)
Gt = feedback_reduce(Gt, [[1:2*N; lefts], [rights; 1:2*N]])

Gt = concatenate(Gt, G_cavs_kerr)
Gt = feedback_reduce(Gt, [[2*N+1:4*N; 1:2*N], [1:2*N; 2*N+1:4*N]], dim)


ψ0 = dual_rail(code_in, cutoff)
H = QObj(Gt.H.data)
tlist = LinRange(0.0, 10.0, 100)


e_ops = [as[j]'*as[j] for j in 1:2*N]
e_ops = [QObj(e.data) for e in e_ops]
println(size(H.data))
println(size(ψ0.data))
println(size(e_ops[1].data))

p = (ti = 2, tf = 7, A = 1)

Ht = QObjEvo(H, coupling)

prob = sesolveProblem(H, ψ0, tlist, e_ops = e_ops, params = p)
sol = sesolve(prob)

figure()
title("Occupation numbers")
xlabel(L"$Time [s]$")
ylabel("Occupation")
for j in 1:2*N
    plot(tlist, real(sol.expect[j, :]), label = L"\langle a_{$j}^\dagger a_{$j} \rangle")
end
legend()
PyPlot.show()
PyPlot.pause(0.1)