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

code_in = [0, 1]
N = length(code_in) # number of photons, equal to the number of dual-rail qubits
cutoff = 3 # a max of two photons at a place
# note: fock(3, 0) = 1|0> + 0|1> + 0|2>

# given operator A, create operator I ⊗ I ⊗ ... ⊗ A ⊗ ... ⊗ I ⊗ I
# with A at position j
function embed(A, j, N)
    res = reduce(⊗, [k == j ? A : qeye(3) for k in 1:2*N])
    return Qobj(res.data)
end


des = destroy(3)

as = [embed(des, j, N) for j in 1:2*N] # array of annie operators, one for each rail




# take a bit array
# create the corresponding dual-rail state
function make_input(code)
    ψ = QuantumObject([1])
    for c in code
        if c == 0
            ψ = ψ ⊗ fock(3, 1) ⊗ fock(3, 0);
        elseif c == 1
            ψ = ψ ⊗ fock(3, 0) ⊗ fock(3, 1);
        else
            error("code must be a bitstring")
        end
    end
    return QObj(ψ.data);
end

function scat_BS(N, m)
        S = qeye(2*N).data
    for _ = 1:m
        phases_odd = zeros(2*N) #randn(2*N)
        vec_odd = [[exp(1im*phases_odd[2*j-1])-exp(1im*phases_odd[2*j]) (-1)*exp(1im*phases_odd[2*j-1])-exp(1im*phases_odd[2*j]); exp(1im*phases_odd[2*j-1]) + exp(1im*phases_odd[2*j]) exp(1im*phases_odd[2*j-1]) - exp(1im*phases_odd[2*j])] for j in 1:N]
        S_odd = 0.5*blockdiag_vec(vec_odd)
        
        phases_even = zeros(2*N)
        vec_even = [[exp(1im*phases_even[2*j])-exp(1im*phases_even[2*j+1]) (-1)*exp(1im*phases_even[2*j])-exp(1im*phases_even[2*j+1]); exp(1im*phases_even[2*j]) + exp(1im*phases_even[2*j+1]) exp(1im*phases_even[2*j]) - exp(1im*phases_even[2*j+1])] for j in 1:(N-1)]
        S_even = blockdiag_vec(vec_even)
        S_even = blockdiag(reshape([exp(1im*phases_even[1])], 1, 1), S_even)
        S_even = 0.5*blockdiag(S_even, reshape([exp(1im*phases_even[end])], 1, 1))

        S = S_even * S_odd * S # reverse multiplication order
    end
    # TODO: multiply by one final odd layer
    phases_odd = zeros(2*N) #randn(2*N)
    vec_odd = [[exp(1im*phases_odd[2*j-1])-exp(1im*phases_odd[2*j]) (-1)*exp(1im*phases_odd[2*j-1])-exp(1im*phases_odd[2*j]); exp(1im*phases_odd[2*j-1]) + exp(1im*phases_odd[2*j]) exp(1im*phases_odd[2*j-1]) - exp(1im*phases_odd[2*j])] for j in 1:N]
    S_odd = 0.5*blockdiag_vec(vec_odd)
    S = S_odd*S

    return QObj(S ⊗ qeye(3^(2*N)).data)
end


# take two QuantumObject's that correspond to matrices
# take their direct sum, then return the corresponding QObj
function ⊕(M1, M2)
    M = QuantumObject
    M.data = blockdiag(M1.data, M2.data)
    return M1
end

function blockdiag(s, t)
    u = [s zeros(s.size[1], t.size[2]); zeros(t.size[1], s.size[2]) t]
    return u
end

function blockdiag_vec(mats)
    r = mats[1]
    for j = 2:length(mats)
        r = blockdiag(r, mats[j])
    end
    return r
end

function vcat_vec(mats)
    r = mats[1]
    for j = 2:length(mats)
        r = vcat(r, mats[j])
    end
    return r
end

ψ0 = make_input(code_in);

#SLH triples

# balanced beamsplitter

m = 3 # number of odd BS layers. In total there will be 2m + 1 BS layers.
num_passes = 3 # number of interactions with the cavities. This means there will be 2*k passes through the BS system
Deltas = ones(2*N, num_passes)
gammas = ones(2*N, num_passes)
chi3 = 1

S = qeye(2*N*3^(2*N))
L = QObj(zeros(2*N*3^(2*N), 3^(2*N)))
H = QObj(zeros(3^(2*N), 3^(2*N)))

for k = 1:num_passes
    a = destroy(3)
    S_before = scat_BS(N, m)
    S_after = scat_BS(N, m)

    S_ = S_after*S_before

    #as2 = as[2]
    # println("as2: ", as2)
    # println("as2.data: ", as2.data)

    L_ = S_after * QObj(vcat_vec([sqrt(gammas[j, k]) * as[j].data for j in 1:2*N]))

    Hs = [Deltas[j, k] * as[j]'*as[j] + gammas[j, k]*as[j]'*as[j]*as[j]'*as[j] for j in 1:2*N] # ensure the ordering is correct: (a'a)^2 vs. (a')^2 a^2.
    H_ = QObj(sum(Hs))
    # examine

    S = S_*S
    println("S_: ", S_.dims)
    println("L: ", L.dims)
    println("S_*L: ", (S_*L).dims)
    println("L_: ", L_.dims)
    println("H_: ", H_.dims)
    println("H: ", H.dims)

    L_'*S_*L
    L'*S_*L_
    println("now: ", (L_'*S_*L).dims)
    H = QObj(-0.5im*(L_'*S_*L - L'*S_'*L_).data + (H_ + H).data)

    # .+ H: works



end

H = QObjEvo(H)

cache_operator(H, ψ0)
one(p, t) = 1

# does it matter if i add the elements of c_ops together?
c_ops = [QObjEvo(QObj(L.data[(3^(2*N)*(j-1)+1):3^(2*N)*j, :])) for j in 1:(2*N)] #[QObjEvo(sqrt(γ)*c, one), QObjEvo(au, guc), QObjEvo(av, gvc)]

for c in c_ops
    cache_operator(c, ψ0)
end

e_ops = [as[j]' * as[j] for j in 1:2*N]

for e in e_ops
    cache_operator(e, ψ0)
end

tlist = LinRange(0, 10, 100)

p = () # named tuple of parameter

sol = mesolve(H, ψ0, tlist, c_ops, params = p; e_ops = e_ops)

#print(sol.expect)

figure()
title("Occupation numbers")
xlabel(L"$Time [s]$")
ylabel("Occupation")
plot(tlist, real(sol.expect[1, :]), label = L"$\langle a_u^\dagger a_u \rangle$")
plot(tlist, real(sol.expect[2, :]), label = L"$\langle c^\dagger c \rangle$")
plot(tlist, real(sol.expect[3, :]), label = L"$\langle a_v^\dagger a_v \rangle$")
legend()
PyPlot.show()
PyPlot.pause(0.1)