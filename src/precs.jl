struct PMultigridPreconBuilder{Tk, CS}
    fe_space::FESpace
    pgrid_config::PMultigridConfiguration
    pcoarse_solver::CS
    blocksize::Int
    kwargs::Tk
end

function PMultigridPreconBuilder(fe_space::FESpace, pgrid_config::PMultigridConfiguration = pmultigrid_config(), pcoarse_solvertype::Type{<:CoarseSolver} = SmoothedAggregationCoarseSolver; blocksize = 1, kwargs...)
    return PMultigridPreconBuilder(fe_space, pgrid_config , setup_coarse_solver(pcoarse_solvertype; kwargs...), blocksize, kwargs)
end

function (b::PMultigridPreconBuilder)(A::AbstractSparseMatrixCSC, p = nothing)
    return (aspreconditioner(pmultigrid(SparseMatrixCSC(A), b.fe_space, b.pgrid_config, b.pcoarse_solver, Val{b.blocksize}; b.kwargs...)), I)
end
