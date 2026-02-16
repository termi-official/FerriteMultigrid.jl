struct PMultigridPreconBuilder{Tk, CS}
    fe_space::FESpace
    pgrid_config::PMultigridConfiguration
    pcoarse_solver::CS
    blocksize::Int
    kwargs::Tk
end

function PMultigridPreconBuilder(fe_space::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(); B = nothing, pcoarse_solver = SmoothedAggregationCoarseSolver(; B), blocksize = 1, kwargs...)
    return PMultigridPreconBuilder(fe_space, pgrid_config , pcoarse_solver, blocksize, kwargs)
end

function (b::PMultigridPreconBuilder)(A::AbstractSparseMatrixCSC, p = nothing)
    ml = @timeit_debug "pmultigrid hierarchy" pmultigrid(SparseMatrixCSC(A), b.fe_space, b.pgrid_config, b.pcoarse_solver, Val{b.blocksize}; b.kwargs...)
    return (aspreconditioner(ml), I)
end
