struct PMultigridPreconBuilder{Tk, CS, C}
    dh::AbstractDofHandler
    ch::ConstraintHandler
    pgrid_config::PMultigridConfiguration
    pcoarse_solver::CS
    blocksize::Int
    cycle::C
    kwargs::Tk
end

function PMultigridPreconBuilder(dh::AbstractDofHandler, ch::ConstraintHandler, pgrid_config::PMultigridConfiguration = pmultigrid_config(); cycle = AMG.V(), pcoarse_solver = SmoothedAggregationCoarseSolver(), blocksize = 1, kwargs...)
    return PMultigridPreconBuilder(dh, ch, pgrid_config, setup_coarse_solver(pcoarse_solvertype; kwargs...), blocksize, kwargs)
end

function (b::PMultigridPreconBuilder)(A::AbstractSparseMatrixCSC, p = nothing)
    return b(SparseMatrixCSC(A), p)
end

function (b::PMultigridPreconBuilder)(A::SparseMatrixCSC, p = nothing)
    ml = @timeit_debug "pmultigrid hierarchy" pmultigrid(A, b.dh, b.ch, b.pgrid_config, b.pcoarse_solver, Val{b.blocksize}; b.kwargs...)
    return (aspreconditioner(ml), I)
end
