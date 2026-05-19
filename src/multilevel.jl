struct MGSolver{T}
    ml::MultiLevel
    b::Vector{T}
end

struct AMGCoarseSolver{C, A, K} <: CoarseSolver
    ml::C
    args::A
    kwargs::K
end


function Base.show(io::IO, ml::AMGCoarseSolver)
    str = """
    AMG
    """
    print(io, str)
    Base.show(io, ml.ml)
    str2 = """
    -----------------
    """
    print(io, str2)
end

struct RugeStubenCoarseSolver{A, K}
    args::A
    kwargs::K
    function RugeStubenCoarseSolver(args...; kwargs...)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end

struct SmoothedAggregationCoarseSolver{A, K}
    args::A
    kwargs::K
    function SmoothedAggregationCoarseSolver(args...; kwargs...)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end

(ctor::RugeStubenCoarseSolver)(A) = AMGCoarseSolver(ruge_stuben(A, ctor.args...; ctor.kwargs...), ctor.args, ctor.kwargs)
(ctor::SmoothedAggregationCoarseSolver)(A) = AMGCoarseSolver(smoothed_aggregation(A, ctor.args...; ctor.kwargs...), ctor.args, ctor.kwargs)

function (amg::AMGCoarseSolver)(x::Vector, b::Vector)
    solve_res = AMG._solve!(x, amg.ml, b, amg.args...; maxiter = 1, calculate_residual=false, amg.kwargs...)
    if solve_res isa Tuple
        x_amg, _ = solve_res
        x .= x_amg
    else
        x .= solve_res
    end
end

function solve!(solt::MGSolver, args...; kwargs...)
    _solve(solt.ml, solt.b, args...; kwargs...)
end

"""
    CachedLinearCoarseSolver <: CoarseSolver

Coarse solver that wraps a `LinearSolve.LinearCache`.  On the first solve after
a hierarchy rebuild (signalled by `isfresh = true` on the underlying cache),
the numeric factorization is updated.  If `UMFPACKFactorization(reuse_symbolic=true)`
is used (the default), the symbolic analysis is also reused, so only the numeric
LU is recomputed.  All subsequent coarse solves within the same Newton step reuse
the cached factorization without any recomputation.
"""
struct CachedLinearCoarseSolver{C <: LinearSolve.LinearCache} <: CoarseSolver
    linsolve::C
end

function (c::CachedLinearCoarseSolver)(x::AbstractVector, b::AbstractVector)
    c.linsolve.b = b
    x .= LinearSolve.solve!(c.linsolve).u
    return x
end

"""
    CachedLinearSolveCoarseSolverBuilder(alg)

A coarse solver builder that reuses the `LinearSolve` factorization cache across
hierarchy rebuilds (e.g. Newton iterations on a fixed mesh).

On the **first** call with a matrix `A`, a full symbolic + numeric factorization
is performed.  On subsequent calls, `LinearSolve.reinit!` marks the cache as fresh
with the new matrix, so only the numeric factorization is repeated.  This is
particularly effective with `UMFPACKFactorization(reuse_symbolic=true)` (the default),
which stores the UMFPACK fill-reducing permutation and symbolic LU across solves.

The sparsity pattern of the coarse matrix is guaranteed to be fixed for polynomial
and geometric multigrid on a fixed mesh, so symbolic reuse is always valid here.
"""
struct CachedLinearSolveCoarseSolverBuilder
    alg::LinearSolve.SciMLLinearSolveAlgorithm
    solver_ref::Ref{Any}
end

function CachedLinearSolveCoarseSolverBuilder(alg::LinearSolve.SciMLLinearSolveAlgorithm)
    return CachedLinearSolveCoarseSolverBuilder(alg, Ref{Any}(nothing))
end

function (b::CachedLinearSolveCoarseSolverBuilder)(A::AbstractMatrix)
    if b.solver_ref[] === nothing
        rhs_tmp = zeros(eltype(A), size(A, 1))
        u_tmp   = zeros(eltype(A), size(A, 2))
        linprob = LinearSolve.LinearProblem(A, rhs_tmp; u0 = u_tmp, alias_A = false, alias_b = false)
        linsolve = LinearSolve.init(linprob, b.alg)
        b.solver_ref[] = CachedLinearCoarseSolver(linsolve)
    else
        # Updating .A triggers setproperty! which sets isfresh = true.
        # On the next solve! call, UMFPACKFactorization will call lu!(cacheval, A)
        # reusing the symbolic factorization when the sparsity pattern is unchanged.
        b.solver_ref[].linsolve.A = A
    end
    return b.solver_ref[]
end

"""
    solve(A, b, dhh, chh, config; pcoarse_solver, kwargs...)

Solve `Ax = b` using polynomial multigrid given a pre-built
`DofHandlerHierarchy` / `ConstraintHandlerHierarchy`.
`kwargs` are forwarded to both the multigrid setup and the iterative solve
(e.g. `maxiter`, `reltol`, `log`).
"""
function solve(A::AbstractMatrix, b::AbstractVector,
               dhh::DofHandlerHierarchy, chh::ConstraintHandlerHierarchy,
               pgrid_config::PMultigridConfiguration = pmultigrid_config();
               pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    @timeit_debug "init"   solver = init(A, b, dhh, chh, pgrid_config; pcoarse_solver, kwargs...)
    @timeit_debug "solve!" solve!(solver; kwargs...)
end

"""
    init(A, b, dhh, chh, config; pcoarse_solver, kwargs...) -> MGSolver

Build a polynomial multigrid solver and return an [`MGSolver`](@ref).
"""
function init(A, b, dhh::DofHandlerHierarchy, chh::ConstraintHandlerHierarchy,
              pgrid_config::PMultigridConfiguration = pmultigrid_config();
              pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    ml = pmultigrid(A, dhh, chh, pgrid_config, pcoarse_solver; kwargs...)
    return MGSolver(ml, b)
end
