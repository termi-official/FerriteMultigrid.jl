struct PMultigridPreconBuilder{Tk, CS, C}
    fe_space::FESpace
    pgrid_config::PMultigridConfiguration
    pcoarse_solver::CS
    blocksize::Int
    cycle::C
    kwargs::Tk
end

function PMultigridPreconBuilder(fe_space::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(); cycle = AMG.V(), pcoarse_solver = SmoothedAggregationCoarseSolver(), blocksize = 1, kwargs...)
    return PMultigridPreconBuilder(fe_space, pgrid_config , pcoarse_solver, blocksize, cycle, kwargs)
end

function (b::PMultigridPreconBuilder)(A::AbstractSparseMatrixCSC, p = nothing)
    return b(SparseMatrixCSC(A), p)
end

function (b::PMultigridPreconBuilder)(A::SparseMatrixCSC, p = nothing)
    ml = @timeit_debug "pmultigrid hierarchy" pmultigrid(A, b.fe_space, b.pgrid_config, b.pcoarse_solver, Val{b.blocksize}; p, b.kwargs...)
    return (aspreconditioner(ml, b.cycle), I)
end
