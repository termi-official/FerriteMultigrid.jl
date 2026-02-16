struct PMGSolver{T}
    ml::MultiLevel
    b::Vector{T}
end

struct SmoothedAggregationCoarseSolver{TK,TKW} <: CoarseSolver
    args::TK
    kwargs::TKW
end

SmoothedAggregationCoarseSolver(args...; kwargs...) = SmoothedAggregationCoarseSolver(args, kwargs)

struct RugeStubenCoarseSolver{TK,TKW} <: CoarseSolver 
    args::TK
    kwargs::TKW
end

RugeStubenCoarseSolver(args...; kwargs...) = RugeStubenCoarseSolver(args, kwargs)

struct AMGCoarseSolver{TA,TG<:AMGAlg,TK,TKW} <: CoarseSolver
    A::TA
    alg::TG
    args::TK
    kwargs::TKW
end

AMGCoarseSolver(A, alg::AMGAlg, args...; kwargs...) = AMGCoarseSolver(A, alg, args, kwargs)

function (sa::SmoothedAggregationCoarseSolver)(A)
    return AMGCoarseSolver(A, SmoothedAggregationAMG(),sa.args...; sa.kwargs...)
end

function (rs::RugeStubenCoarseSolver)(A)
    return AMGCoarseSolver(A, RugeStubenAMG(), rs.args...; rs.kwargs...)
end

function (amg::AMGCoarseSolver)(x::Vector, b::Vector)
    solve_res = AMG.solve(amg.A, b, amg.alg, amg.args...; amg.kwargs...)
    if solve_res isa Tuple
        x_amg, _ = solve_res
        x .= x_amg
    else
        x .= solve_res
    end
end

"""
    solve(A::AbstractMatrix, b::Vector, fe_space::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(), pcoarse_solvertype::Type{<:CoarseSolver} = SmoothedAggregationCoarseSolver, args...; kwargs...)
This function solves the linear system `Ax = b` using polynomial multigrid methods with a coarse solver of type `pcoarse_solvertype`.

# Fields
- `A`: The system matrix.
- `b`: The right-hand side vector.
- `fe_space`: See [`FESpace`](@ref) for details on the finite element space.   
- `pgrid_config`: Configuration for the polynomial multigrid method, see [`PMultigridConfiguration`](@ref) for details.
- `pcoarse_solvertype`: The type of coarse solver to use (e.g., `SmoothedAggregationCoarseSolver`, `Pinv`).
- `args...`: Additional arguments for the coarse solver.
- `kwargs...`: Additional keyword arguments for the coarse solver.
"""
function solve(A::AbstractMatrix, b::Vector, fe_space::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(), pcoarse_solvertype::Type{<:CoarseSolver} = SmoothedAggregationCoarseSolver, args...; kwargs...)
    solver = init(A, b, fe_space, pgrid_config, pcoarse_solvertype, args...; kwargs...)
    solve!(solver, args...; kwargs...)
end

function init(A, b, fine_fespace::FESpace, pgrid_config::PMultigridConfiguration, pcoarse_solvertype = SmoothedAggregationCoarseSolver, args...; kwargs...)
    ml = @timeit_debug "pmultigrid hierarchy" pmultigrid(A, fine_fespace, pgrid_config, setup_coarse_solver(pcoarse_solvertype,args...;kwargs...) ;kwargs...)
    PMGSolver(ml, b)
end

function solve!(solt::PMGSolver, args...; kwargs...)
    @timeit_debug "AMG _solve" _solve(solt.ml, solt.b, args...; kwargs...)
end

setup_coarse_solver(solvertype ,args...;kwargs...) = solvertype
setup_coarse_solver(solvertype::Type{<:SmoothedAggregationCoarseSolver}, args...; kwargs...) =  SmoothedAggregationCoarseSolver(args...; kwargs...)
setup_coarse_solver(solvertype::Type{<:RugeStubenCoarseSolver}, args...; kwargs...) =  RugeStubenCoarseSolver(args...; kwargs...)

