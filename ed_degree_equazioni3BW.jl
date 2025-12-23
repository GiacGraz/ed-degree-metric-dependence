using HomotopyContinuation
using LinearAlgebra
using Random

# Bombieri–Weyl metric on Sym^2(C^3) in coordinates
# x = [x11, x12, x13, x22, x23, x33]
const MBW = Diagonal(ComplexF64[1, 2, 2, 1, 2, 1])

function ED_det_symmetric3_BW(; seed_u::Int=1, show_info::Bool=true)
    @var x11 x12 x13 x22 x23 x33 λ
    x = [x11, x12, x13, x22, x23, x33]

    # symmetric 3x3 matrix
    X = [
        x11  x12  x13
        x12  x22  x23
        x13  x23  x33
    ]

    f = det(X)  # hypersurface: det = 0

    # gradient wrt x11,x12,x13,x22,x23,x33
    grad = [
        differentiate(f, x11),
        differentiate(f, x12),
        differentiate(f, x13),
        differentiate(f, x22),
        differentiate(f, x23),
        differentiate(f, x33)
    ]

    # generic target u in C^6
    Random.seed!(seed_u)
    u = randn(ComplexF64, 6)

    # ED equations for hypersurface with metric MBW:
    # f(x)=0 and MBW*(x-u)=λ*grad f(x)
    eqs = Any[f]
    for i in 1:6
        push!(eqs, MBW[i,i] * (x[i] - u[i]) - λ * grad[i])
    end

    vars = [x11, x12, x13, x22, x23, x33, λ]
    sys = System(eqs, vars)

    res = solve(sys; start_system = :total_degree, show_progress=false)
    sols = solutions(res; only_nonsingular=true)

    if show_info
        println("BW metric (diag(1,2,2,1,2,1)), seed_u=$seed_u")
        println("nonsingular solutions = $(length(sols))")
    end

    return length(sols)
end

function batch_test_BW(; trials_u::Int=5)
    counts = Int[]
    for su in 1:trials_u
        push!(counts, ED_det_symmetric3_BW(seed_u=su, show_info=false))
    end
    println("Counts over seeds: ", counts)
    println("Unique counts observed: ", sort(unique(counts)))
end

# Example run:
batch_test_BW(trials_u=5)
