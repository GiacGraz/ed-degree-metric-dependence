using HomotopyContinuation
using LinearAlgebra
using Random

# --- helper: random symmetric invertible metric on C^6 ---
function random_metric_M(; seed::Int=1)
    Random.seed!(seed)
    while true
        A = randn(ComplexF64, 6, 6)
        M = A + transpose(A)                 # symmetric (bilinear), not Hermitian
        if abs(det(M)) > 1e-8
            return M
        end
    end
end

# --- compute ED-degree for det=0 in Sym^2(C^3) with metric M ---
function ED_det_symmetric3(; seed_u::Int=1, seed_M::Int=1, show_info::Bool=true)
    # variables: x11, x12, x13, x22, x23, x33 and Lagrange multiplier λ
    @var x11 x12 x13 x22 x23 x33 λ

    # pack x as vector in C^6
    x = [x11, x12, x13, x22, x23, x33]

    # symmetric 3x3 matrix
    X = [
        x11  x12  x13
        x12  x22  x23
        x13  x23  x33
    ]

    f = det(X)  # defining equation of rank <= 2

    # gradient of det wrt the 6 coordinates
    grad = [
        differentiate(f, x11),
        differentiate(f, x12),
        differentiate(f, x13),
        differentiate(f, x22),
        differentiate(f, x23),
        differentiate(f, x33)
    ]

    # random target u in C^6
    Random.seed!(seed_u)
    u = randn(ComplexF64, 6)

    # random metric matrix M on C^6 (symmetric invertible)
    M = random_metric_M(seed=seed_M)

    # ED critical equations for hypersurface f=0:
    #   f(x)=0
    #   M*(x-u) = λ * grad f(x)
    eqs = Any[f]
    for i in 1:6
        push!(eqs, sum(M[i,j] * (x[j] - u[j]) for j in 1:6) - λ * grad[i])
    end

    # solve system in variables (x11,x12,x13,x22,x23,x33,λ)
    vars = [x11, x12, x13, x22, x23, x33, λ]
    sys = System(eqs, vars)
    res = solve(sys; start_system=:total_degree, show_progress=false)

    sols = solutions(res; only_nonsingular=true)
    if show_info
        println("seed_u=$seed_u, seed_M=$seed_M, det(M)=$(det(M))")
        println("nonsingular solutions = $(length(sols))")
    end
    return length(sols)
end

# --- run a small stability test ---
function batch_test(; trials_u=3, trials_M=5)
    counts = Dict{Tuple{Int,Int},Int}()
    for sM in 1:trials_M
        for su in 1:trials_u
            c = ED_det_symmetric3(seed_u=su, seed_M=sM, show_info=false)
            counts[(su,sM)] = c
        end
    end
    println("Counts (seed_u, seed_M) -> #solutions:")
    for ((su,sM), c) in sort(collect(counts); by=x->x[1])
        println("  ($su,$sM) -> $c")
    end
    println("Unique counts observed: ", sort(unique(values(counts))))
end

# Example:
batch_test(trials_u=3, trials_M=5)
