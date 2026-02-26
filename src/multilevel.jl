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
    solve(A, b, dh, ch, config; pcoarse_solver, kwargs...)
    solve(A, b, dhh, chh, config; pcoarse_solver, kwargs...)

Solve `Ax = b` using polynomial multigrid.  Accepts either a single
`AbstractDofHandler` / `ConstraintHandler` pair (hierarchy built automatically)
or a pre-built `DofHandlerHierarchy` / `ConstraintHandlerHierarchy`.
`kwargs` are forwarded to both the multigrid setup and the iterative solve
(e.g. `maxiter`, `reltol`, `log`).
"""
function solve(A::AbstractMatrix, b::AbstractVector,
               dh::AbstractDofHandler, ch::ConstraintHandler,
               pgrid_config::PMultigridConfiguration = pmultigrid_config();
               pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    @timeit_debug "init"   solver = init(A, b, dh, ch, pgrid_config; pcoarse_solver, kwargs...)
    @timeit_debug "solve!" solve!(solver; kwargs...)
end

function solve(A::AbstractMatrix, b::AbstractVector,
               dhh::DofHandlerHierarchy, chh::ConstraintHandlerHierarchy,
               pgrid_config::PMultigridConfiguration = pmultigrid_config();
               pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    @timeit_debug "init"   solver = init(A, b, dhh, chh, pgrid_config; pcoarse_solver, kwargs...)
    @timeit_debug "solve!" solve!(solver; kwargs...)
end

"""
    init(A, b, dh, ch, config; pcoarse_solver, kwargs...) -> MGSolver
    init(A, b, dhh, chh, config; pcoarse_solver, kwargs...) -> MGSolver

Build a polynomial multigrid solver and return an [`MGSolver`](@ref).
"""
function init(A, b, dh::AbstractDofHandler, ch::ConstraintHandler,
              pgrid_config::PMultigridConfiguration = pmultigrid_config();
              pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    ml = pmultigrid(A, dh, ch, pgrid_config, pcoarse_solver; kwargs...)
    return MGSolver(ml, b)
end

function init(A, b, dhh::DofHandlerHierarchy, chh::ConstraintHandlerHierarchy,
              pgrid_config::PMultigridConfiguration = pmultigrid_config();
              pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    ml = pmultigrid(A, dhh, chh, pgrid_config, pcoarse_solver; kwargs...)
    return MGSolver(ml, b)
end
