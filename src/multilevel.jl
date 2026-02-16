struct PMGSolver{T}
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

"""
    solve(A::AbstractMatrix, b::Vector, fe_space::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(), pcoarse_solvertype = SmoothedAggregationCoarseSolver, args...; kwargs...)
This function solves the linear system `Ax = b` using polynomial multigrid methods with a coarse solver of type `pcoarse_solvertype`.

# Arguments
- `A`: The system matrix.
- `b`: The right-hand side vector.
- `fe_space`: See [`FESpace`](@ref) for details on the finite element space.   
- `pgrid_config`: Configuration for the polynomial multigrid method, see [`PMultigridConfiguration`](@ref) for details.
- `pcoarse_solvertype`: The type of coarse solver to use (e.g., `SmoothedAggregationCoarseSolver`, `Pinv`).
- `args...`: Additional arguments for the init and solve.

# Keyword arguments
- `pcoarse_solver`: The coarse solver for the polynomial multigrid.
- `kwargs...`: Additional keyword arguments for the init and solve.
"""
function solve(A::AbstractMatrix, b::Vector, fe_space::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(), args...; pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    @timeit_debug "init" solver = init(A, b, fe_space, pgrid_config, args...; pcoarse_solver, kwargs...)
    @timeit_debug "solve!" solve!(solver, args...; kwargs...)
end

function init(A, b, fine_fespace::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(), args...; pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    ml = pmultigrid(A, fine_fespace, pgrid_config, pcoarse_solver, args...; kwargs...)
    return PMGSolver(ml, b)
end

function solve!(solt::PMGSolver, args...; kwargs...)
    _solve(solt.ml, solt.b, args...; kwargs...)
end
