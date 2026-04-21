## defines how we obtain A operator for the coarse grid
abstract type AbstractCoarseningStrategy end

@doc raw"""
    Galerkin <: AbstractCoarseningStrategy
Galerkin coarsening operator can be defined as follows:
```math
A_{h,p-1} = \mathcal{I}_{p}^{p-1} A_{h,p} \mathcal{I}_{p-1}^p
```
and according to [tielen2020](@citet) $\mathcal{I}_{p-1}^p$ is the interpolation operator from the coarse space to the fine space
and is defined as follows:

```math
\mathcal{I}_{p-1}^p (\mathbf{v}_{p-1}) = (\mathbf{M}_p)^{-1} \mathbf{P}_{p-1}^p \, \mathbf{v}_{p-1}
```
"""
struct Galerkin <: AbstractCoarseningStrategy
    is_sym::Bool
end
Galerkin() = Galerkin(true)

"""
    Rediscretization{TI, TS} <: AbstractCoarseningStrategy

Coarsening strategy that re-assembles the operator on the coarse grid using FerriteOperators.

# Fields
- `integrator::TI` – an `AbstractBilinearIntegrator` (e.g. `DiffusionMultigrid`)
- `strategy::TS`   – an `AbstractAssemblyStrategy` (default: `SequentialAssemblyStrategy(SequentialCPUDevice())`)
- `is_sym::Bool`   – whether the operator is symmetric (determines R = Pᵀ vs separate assembly)
"""
struct Rediscretization{TI <: AbstractBilinearIntegrator, TS <: AbstractAssemblyStrategy} <: AbstractCoarseningStrategy
    integrator::TI
    strategy::TS
    is_sym::Bool
end
Rediscretization(integrator::AbstractBilinearIntegrator) =
    Rediscretization(integrator, SequentialAssemblyStrategy(SequentialCPUDevice()), true)
Rediscretization(integrator::AbstractBilinearIntegrator, is_sym::Bool) =
    Rediscretization(integrator, SequentialAssemblyStrategy(SequentialCPUDevice()), is_sym)

## defines how we project from fine to coarse grid - always one step to p=1

"""
    PMultigridConfiguration{TC<:AbstractCoarseningStrategy}
This struct represents the configuration for the polynomial multigrid method.
"""
struct PMultigridConfiguration{TC<:AbstractCoarseningStrategy}
    coarse_strategy::TC # coarsening strategy
end


"""
    pmultigrid_config(;coarse_strategy = Galerkin())
This function is the main api to instantiate [`PMultigridConfiguration`](@ref).
"""
pmultigrid_config(;coarse_strategy = Galerkin()) = PMultigridConfiguration(coarse_strategy)


"""
    pmultigrid(A, dhh, chh, pgrid_config, pcoarse_solver, [Val{bs}]; kwargs...)

Build a polynomial multigrid preconditioner from a pre-built `DofHandlerHierarchy` and
`ConstraintHandlerHierarchy`.  Index 1 of the hierarchies must be the coarsest level
and index `end` the finest.
"""
function pmultigrid(
    A::TA,
    dhh::DofHandlerHierarchy,
    chh::Union{ConstraintHandlerHierarchy, Nothing},
    pgrid_config::PMultigridConfiguration,
    pcoarse_solver,
    ::Type{Val{bs}} = Val{1};
    u = nothing,
    p = nothing,
    presmoother  = GaussSeidel(),
    postsmoother = GaussSeidel(),
    kwargs...,
    ) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    n_levels = length(dhh) - 1
    @assert n_levels >= 1 "DofHandlerHierarchy must have at least 2 levels"
    chh !== nothing && @assert length(dhh) == length(chh) "Dof and constraint handler hierarchies must match"

    levels = Vector{Level{TA,TA,Adjoint{T,TA}}}()
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    cs    = pgrid_config.coarse_strategy
    cur_A = A

    fine_ch = nothing
    coarse_ch = nothing
    # Iterate from finest (index end) down to coarsest (index 1)
    for k in n_levels:-1:1
        # Unpack dh pair
        fine_dh   = dhh[k+1]
        coarse_dh = dhh[k]

        # Unpack ch pair if available
        if chh !== nothing
            fine_ch   = chh[k+1]
            coarse_ch = chh[k]
        end

        @timeit_debug "extend_hierarchy!" cur_A = extend_hierarchy!(
            levels, fine_dh, fine_ch, coarse_dh, coarse_ch, cur_A, cs, u, p)

        coarse_x!(w, size(cur_A, 1))
        coarse_b!(w, size(cur_A, 1))
        residual!(w, size(cur_A, 1))
    end

    coarse_solver = @timeit_debug "coarse solver setup" pcoarse_solver(cur_A)
    return MultiLevel(levels, cur_A, coarse_solver, presmoother, postsmoother, w)
end

function extend_hierarchy!(levels, fine_dh, fine_ch, coarse_dh, coarse_ch, A, cs::Galerkin, u, p)
    P = @timeit_debug "build prolongator" build_prolongator(fine_dh, coarse_dh)
    R = @timeit_debug "build restriction" build_restriction(coarse_dh, fine_dh, P, cs.is_sym)
    push!(levels, Level(A, P, R))
    RAP = @timeit_debug "RAP" R * A * P # Galerkin projection
    return RAP
end

function extend_hierarchy!(levels, fine_dh, fine_ch, coarse_dh, coarse_ch, A, cs::Rediscretization, u, p)
    P = build_prolongator(fine_dh, coarse_dh)
    R = build_restriction(coarse_dh, fine_dh, P, cs.is_sym)
    push!(levels, Level(A, P, R))

    op = @timeit_debug "setup coarse operator" setup_operator(cs.strategy, cs.integrator, coarse_dh)
    @timeit_debug "assemble coarse operator" update_operator!(op, p) # TODO might call update_linearization! instead.
    coarse_ch !== nothing && apply!(op.A, coarse_ch)
    A = op.A
    return A
end



