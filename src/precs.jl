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

The prolongation and restriction operators are assembled once at construction time via
[`gmultigrid_symbolic`](@ref) and reused across all subsequent calls (i.e. across Newton
iterations).

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
    geo_ref::Ref{Any}
end

function GMultigridCoarseSolverBuilder(gh, dhh, chh;
        gconfig        = gmultigrid_config(),
        pcoarse_solver = SmoothedAggregationCoarseSolver())
    # For Galerkin, geo is built on first call (when A is available).
    # For Rediscretization, build it now (no A needed).
    geo_ref = if gconfig.coarse_strategy isa Galerkin
        Ref{Any}(nothing)
    else
        Ref{Any}(gmultigrid_symbolic(gh, dhh, gconfig))
    end
    return GMultigridCoarseSolverBuilder(gh, dhh, chh, gconfig, pcoarse_solver, geo_ref)
end

function (b::GMultigridCoarseSolverBuilder)(A::SparseMatrixCSC)
    if b.geo_ref[] === nothing
        b.geo_ref[] = gmultigrid_symbolic(b.gh, b.dhh, b.gconfig, A)
    end
    ml = gmultigrid_numeric!(b.geo_ref[], A, b.gh, b.dhh, b.chh, b.gconfig, b.pcoarse_solver)
    return GMultigridCoarseSolver(ml)
end

"""
    PMultigridPreconBuilder(dh, ch, pgrid_config; cycle, pcoarse_solver, blocksize, kwargs...)

A callable preconditioner builder for use with `LinearSolve.KrylovJL_CG(precs = builder)`.
When called as `builder(A, p)`, it assembles the polynomial multigrid hierarchy and returns
`(aspreconditioner(ml), I)`.

For the Galerkin strategy, the full symbolic phase (prolongators, restrictors, and RAP
workspace pre-allocation) is performed on the **first** call when `A` is available, and the
result is cached for all subsequent Newton iterations.  For Rediscretization, the P/R
operators are built eagerly at construction time.

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
    geo_ref::Ref{Any}
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
    geo_ref = if pgrid_config.coarse_strategy isa Galerkin
        Ref{Any}(nothing)
    else
        Ref{Any}(pmultigrid_symbolic(dh, pgrid_config))
    end
    return PMultigridPreconBuilder(dh, ch, pgrid_config, pcoarse_solver, blocksize, cycle, geo_ref, kwargs)
end

function PMultigridPreconBuilder(
        dh::DofHandlerHierarchy,
        pgrid_config::PMultigridConfiguration{<:Galerkin} = pmultigrid_config();
        cycle          = AMG.V(),
        pcoarse_solver = SmoothedAggregationCoarseSolver(),
        blocksize      = 1,
        kwargs...
    )
    return PMultigridPreconBuilder(dh, nothing, pgrid_config, pcoarse_solver, blocksize, cycle, Ref{Any}(nothing), kwargs)
end

function (b::PMultigridPreconBuilder)(A::AbstractSparseMatrixCSC, p = nothing)
    return b(SparseMatrixCSC(A), p)
end

function (b::PMultigridPreconBuilder)(A::SparseMatrixCSC, p = nothing)
    if b.geo_ref[] === nothing
        b.geo_ref[] = pmultigrid_symbolic(b.dhh, b.pgrid_config, A)
    end
    ml = @timeit_debug "pmultigrid hierarchy" pmultigrid_numeric!(b.geo_ref[], A, b.dhh, b.chh, b.pgrid_config, b.pcoarse_solver, Val{b.blocksize}; b.kwargs...)
    return (aspreconditioner(ml, b.cycle), I)
end
