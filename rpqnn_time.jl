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
import SciMLBase: AddedOperator
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

as = [embed(des, j, N, cutoff).data for j in 1:2*N] # array of annie operators, one for each rail




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

function expand_first(A, size2)
    h, w = size2
    Au = hcat(A, zeros(size(A)[1], w))
    Al = zeros(h, w+size(A)[2])
    return vcat(Au, Al)
end

function expand_second(A, size1)
    h, w = size1
    Au = zeros(h, size(A)[2] + w)
    Al = hcat(zeros(size(A)[1], w), A)
    return vcat(Au, Al)
end


function concatenate(G1, G2)
    size1 = size(G1.S[1][1])
    size2 = size(G2.S[1][1])
    S1_new = [[expand_first(S[1], size2), S[2]] for S in G1.S]
    S2_new = [[expand_second(S[1], size1), S[2]] for S in G2.S]
    S = S1_new ∪ S2_new

    L1_new = [[vcat(L[1], zeros(size2[1], size(L[1])[2])), L[2]] for L in G1.L]
    L2_new = [[vcat(zeros(size1[1], size(L[1])[2]), L[1]), L[2]] for L in G2.L]
    L = L1_new ∪ L2_new

    H = G1.H ∪ G2.H

    return (S = S, L = L, H = H)
end

function concatenate_vec(Gs)
    return reduce(concatenate, Gs) # this is concatenate(concatenate(Gs[1], Gs[2]), Gs[3]) and so on
end

function multiply(ops1, ops2)
    ops = []
    for op1 in ops1
        for op2 in ops2
            A = op1[1] * op2[1]
            f = (p, t) -> op1[2](p, t) * op2[2](p, t)
            push!(ops, [A, f])
        end
    end
    return ops
end

function simplify(ops)
    ops_dict = Dict{Any, Any}()
    for op in ops
        key = op[2]
        if haskey(ops_dict, key)
            ops_dict[key] += op[1]
        else
            ops_dict[key] = op[1]
        end
    end
    return [[ops_dict[key], key] for key in keys(ops_dict)]
end

function simplify_G(G)
    return (S = simplify(G.S), L = simplify(G.L), H = simplify(G.H))
end

function to_added(ops)
    return reduce(+, [QObjEvo((QObj(op[1]), op[2])) for op in ops])
end

function from_added(added)
    ops_list = []
    for op in added.data.ops
        push!(ops_list, [op.L, op.λ])
    end
end

function scale(s, ops)
    return [[s * op[1], op[2]] for op in ops]
end

function adjoint_list(ops)
    return [[op[1]', op[2]] for op in ops] # this assumes all functions are real-valued
    # ensure that phase is absorbed into matrix operator, maybe.
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
    S, L, H = G.S, G.L, G.H


    #in notation: x is output, y is input
    for j in eachindex(ins)
        println("starting $j")
        println("S length: $(length(S))", " L length: $(length(L))", " H length: $(length(H))")
        S, L, H = simplify(S), simplify(L), simplify(H)
        println("S length: $(length(S))", " L length: $(length(L))", " H length: $(length(H))")

        in = ins[j]
        out = outs[j]
        Sxbyb = [[Si[1][[1:dim*(out-1); dim*out+1:end], [1:dim*(in-1); dim*in+1:end]], Si[2]] for Si in S]
        Sxby = [[Si[1][[1:dim*(out-1); dim*out+1:end], dim*(in-1)+1:dim*in], Si[2]] for Si in S]
        Sxy = [[Si[1][dim*(out-1)+1:dim*out, dim*(in-1)+1 : dim*in], Si[2]] for Si in S]
        Sxyb = [[Si[1][dim*(out-1)+1:dim*out, [1:dim*(in-1); dim*in+1:end]], Si[2]] for Si in S]
        one(p, t) = 1
        I = [[qeye(dim).data, one]]
        println("now bleh")
        ISxy = I ∪ [[(-1) * Sxyi[1], Sxyi[2]] for Sxyi in Sxy]
        println("then bleh")
        invISxy = [[zeros(dim, dim), one]]
        println("before try")
        try
            invISxy = from_added(inv(to_added(ISxy)))
        catch e
            if e isa SingularException
                invISxy = [[zeros(dim, dim), one]]
                #println("Warning: singular matrix encountered during feedback reduction. Setting inverse to zero matrix.")
            else
                rethrow(e)
            end
        end
        println("after try")
        S_red = Sxbyb ∪ reduce(multiply, [Sxby, invISxy, Sxyb])
        Lx = [[Li[1][dim*(out-1)+1:dim*out, :], Li[2]] for Li in L]
        Lxb = [[Li[1][[1:dim*(out-1); dim*out+1:end], :], Li[2]] for Li in L]
        L_red = Lxb ∪ reduce(multiply, [Sxby, invISxy, Lx])

        println("before Sy")

        Sy = [[Si[1][:, dim*(in-1)+1:dim*in], Si[2]] for Si in S]
        invISxy_a = I ∪ [[(-1) * Si[1]', Si[2]] for Si in Sxy] # adjoint
        H_red = H ∪ reduce(multiply, [scale(-0.5, adjoint_list(L)), Sy, invISxy, Lx]) ∪ reduce(multiply, [scale(-0.5, adjoint_list(Lx)), invISxy_a, adjoint_list(Sy), L])
        
        S = simplify(S_red)
        L = simplify(L_red)
        H = simplify(H_red)
    end
    if size(L[1][1])[1] > 0
        return (S = simplify(S_red), L = simplify(L_red), H = simplify(H_red))
    else # this corresponds to a closed system: it has no scattering or coupling
        return (S = nothing, L = nothing, H = H)
    end
end

# G may be a QObjEvo, or it may be an AddedOperator
# a trivial QObjEvo is just a QObj and the function "one"

χ3 = 1
dim = cutoff^(2*N)
I = qeye(dim)

κ(p, t) = (t > p.ti & t < p.tf) ? p.A : 0 # rect pulse
one(p, t) = 1
two(p, t) = t^2

# S, L and H will each be a list, and each element is an array of two elements:
# a QObj and a function of (p, t)

G_BS = (S = [[1/sqrt(2)*[1 (-1) 0 0; 0 1 0 (-1); 1 0 1 0; 0 1 0 1] ⊗ I.data, one]], L = [[[0; 0; 0; 0 ]⊗I.data, one]], H = [[[0]⊗ I.data, one]])


# a beamsplitter interacts with four channels. I number them like this: right upper, right lower, left upper, left lower
G_mir = (S = [[exp(1im*pi) * I.data, one]], L = [[[0] ⊗ I.data, one]], H = [[[0]⊗ I.data, one]]) # one input, one output
println("ah")
G_mirs = concatenate_vec([G_mir for _ in 1:2*N])
println("oh")
G_BS_pair = concatenate(G_BS, G_BS)
println("boo")
G_BS_pair = feedback_reduce(G_BS_pair, [[6 8 1 3], [2 4 5 7]], dim)
println("geh")
G_BS_odd = concatenate_vec([G_BS_pair for _ in 1:N])
G_cavs_kerr = [(S = [[I.data, one]], L = [[as[j], κ]], H = [[as[j]'*as[j], one], [χ3*as[j]'*as[j]'*as[j]*as[j], one]]) for j in 1:2*N]
G_cavs_kerr = concatenate_vec(G_cavs_kerr)




Gt = concatenate(G_mirs, G_BS_odd)
rights = [i + 2*N for i in 1:4*N if mod(i-1, 2) == 0]
lefts = [i + 2*N for i in 1:4*N if mod(i-1, 2) == 1]
Gt = feedback_reduce(Gt, [[1:2*N; lefts], [rights; 1:2*N]], dim)
# two-way cascade of mirror and BS layer into each other

Gt = concatenate(Gt, G_BS_even)
Gt = feedback_reduce(Gt, [[1:2*N; lefts], [rights; 1:2*N]], dim)
Gt = simplify_G(Gt)

println("added second BS layer")
println(length(Gt.S))

Gt = concatenate(Gt, G_BS_odd)
Gt = feedback_reduce(Gt, [[1:2*N; lefts], [rights; 1:2*N]], dim)
Gt = simplify_G(Gt)
println("added third BS layer")
println(length(Gt.S))

Gt = concatenate(Gt, G_cavs_kerr)
Gt = feedback_reduce(Gt, [[2*N+1:4*N; 1:2*N], [1:2*N; 2*N+1:4*N]], dim)
Gt = simplify_G(Gt)
println("added cavities")

ψ0 = dual_rail(code_in, cutoff)
H = reduce(+, [QObjEvo((QObj(h[1]), h[2])) for h in Gt.H])
tlist = LinRange(0.0, 10.0, 100)


e_ops = [as[j]'*as[j] for j in 1:2*N]
e_ops = [QObj(e.data) for e in e_ops]

p = (ti = 2, tf = 7, A = 1)

Ht = QObjEvo(H, coupling)

prob = sesolveProblem(H, ψ0, tlist, e_ops = e_ops, params = p)
sol = sesolve(prob)

figure()
title("Occupation numbers")
xlabel(L"$Time [s]$")
ylabel("Occupation")
plot(tlist, [κ(p, t) for t in tlist], label = L"$\kappa(t)$", color = "k", ls = ":")
for j in 1:2*N
    plot(tlist, real(sol.expect[j, :]), label = L"\langle a_{$j}^\dagger a_{$j} \rangle")
end
legend()
PyPlot.show()
PyPlot.pause(0.1)