"""
    GMultigridCoarseSolver(ml)

Wraps a geometric multigrid `MultiLevel` as a coarse solver for polynomial multigrid.
One V-cycle of `ml` is applied when the solver is invoked.
"""
struct GMultigridCoarseSolver{ML} <: CoarseSolver
    ml::ML
end

function (c::GMultigridCoarseSolver)(x::Vector, b::Vector)
    solve_res = AMG._solve!(x, c.ml, b; maxiter = 1, calculate_residual = false)
    if solve_res isa Tuple
        x .= first(solve_res)
    else
        x .= solve_res
    end
end

"""
    GMultigridCoarseSolverBuilder(gh, dhh, chh; gconfig, pcoarse_solver)

A factory that builds a [`GMultigridCoarseSolver`](@ref) from a given matrix.
Intended as the `pcoarse_solver` argument to `pmultigrid` / [`MultigridPreconBuilder`](@ref)
to chain polynomial and geometric multigrid levels.

# Arguments
- `gh`             – [`GridHierarchy`](@ref) (index 1 = coarsest, `end` = finest)
- `dhh`            – [`DofHandlerHierarchy`](@ref) with one handler per grid level
- `chh`            – [`ConstraintHandlerHierarchy`](@ref) with one handler per grid level
- `gconfig`        – [`GMultigridConfiguration`](@ref) (default: `gmultigrid_config()`)
- `pcoarse_solver` – coarse solver for the geometric hierarchy (default: `SmoothedAggregationCoarseSolver()`)
"""
struct GMultigridCoarseSolverBuilder{GH, DHH, CHH, GC, CS}
    gh::GH
    dhh::DHH
    chh::CHH
    gconfig::GC
    pcoarse_solver::CS
end

function GMultigridCoarseSolverBuilder(gh, dhh, chh;
        gconfig        = gmultigrid_config(),
        pcoarse_solver = SmoothedAggregationCoarseSolver())
    return GMultigridCoarseSolverBuilder(gh, dhh, chh, gconfig, pcoarse_solver)
end

function (b::GMultigridCoarseSolverBuilder)(A::SparseMatrixCSC)
    ml = gmultigrid(A, b.gh, b.dhh, b.chh, b.gconfig, b.pcoarse_solver)
    return GMultigridCoarseSolver(ml)
end

"""
    PMultigridPreconBuilder(dh, ch, pgrid_config; cycle, pcoarse_solver, blocksize, kwargs...)

A callable preconditioner builder for use with `LinearSolve.KrylovJL_CG(precs = builder)`.
When called as `builder(A, p)`, it assembles the polynomial multigrid hierarchy and returns
`(aspreconditioner(ml), I)`.

`pcoarse_solver` may be any coarse-solver factory (e.g. `SmoothedAggregationCoarseSolver()`,
`GMultigridCoarseSolverBuilder(gh, dhh, chh)`) allowing arbitrary chaining of multigrid
strategies.
"""
struct PMultigridPreconBuilder{Tk, CS, C}
    dhh::DofHandlerHierarchy
    chh::Union{ConstraintHandlerHierarchy, Nothing}
    pgrid_config::PMultigridConfiguration
    pcoarse_solver::CS
    blocksize::Int
    cycle::C
    kwargs::Tk
end

function PMultigridPreconBuilder(
        dh::DofHandlerHierarchy, ch::ConstraintHandlerHierarchy,
        pgrid_config::PMultigridConfiguration = pmultigrid_config();
        cycle          = AMG.V(),
        pcoarse_solver = SmoothedAggregationCoarseSolver(),
        blocksize      = 1,
        kwargs...
    )
    return PMultigridPreconBuilder(dh, ch, pgrid_config, pcoarse_solver, blocksize, cycle, kwargs)
end

function PMultigridPreconBuilder(
        dh::DofHandlerHierarchy,
        pgrid_config::PMultigridConfiguration{<:Galerkin} = pmultigrid_config();
        cycle          = AMG.V(),
        pcoarse_solver = SmoothedAggregationCoarseSolver(),
        blocksize      = 1,
        kwargs...
    )
    return PMultigridPreconBuilder(dh, nothing, pgrid_config, pcoarse_solver, blocksize, cycle, kwargs)
end

function (b::PMultigridPreconBuilder)(A::AbstractSparseMatrixCSC, p = nothing)
    return b(SparseMatrixCSC(A), p)
end

function (b::PMultigridPreconBuilder)(A::SparseMatrixCSC, p = nothing)
    ml = @timeit_debug "pmultigrid hierarchy" pmultigrid(A, b.dhh, b.chh, b.pgrid_config, b.pcoarse_solver, Val{b.blocksize}; b.kwargs...)
    return (aspreconditioner(ml, b.cycle), I)
end
