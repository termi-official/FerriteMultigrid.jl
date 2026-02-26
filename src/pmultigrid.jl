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

## defines how we project from fine to coarse grid
abstract type AbstractProjectionStrategy end

@doc raw"""
    DirectProjection <: AbstractProjectionStrategy
This struct represents a direct projection from $\mathcal{V}_{h,p}$ to $\mathcal{V}_{h,p=1}$. 
"""
struct DirectProjection <: AbstractProjectionStrategy end
    
@doc raw"""
    StepProjection <: AbstractProjectionStrategy
This struct represents a projection from $\mathcal{V}_{h,p}$ to $\mathcal{V}_{h,p-step}$, where `step` is a positive integer.
It is used to reduce the polynomial order by a fixed step size until `p = 1`.
"""    
struct StepProjection <: AbstractProjectionStrategy 
    step::Int
    function StepProjection(step::Int)
        step < 1 && error("Step must be greater than or equal to 1")
        return new(step)
    end
end

"""
    PMultigridConfiguration{TC<:AbstractCoarseningStrategy, TP<:AbstractProjectionStrategy}
This struct represents the configuration for the polynomial multigrid method.
"""
struct PMultigridConfiguration{TC<:AbstractCoarseningStrategy, TP<:AbstractProjectionStrategy}
    coarse_strategy::TC # coarsening strategy
    proj_strategy::TP # projection strategy
end


"""
    pmultigrid_config(;coarse_strategy = Galerkin(), proj_strategy = DirectProjection())
This function is the main api to instantiate [`PMultigridConfiguration`](@ref).
"""
pmultigrid_config(;coarse_strategy = Galerkin(), proj_strategy = DirectProjection()) = PMultigridConfiguration(coarse_strategy, proj_strategy)

"""
    build_pmg_dofhandler_hierarchy(dh, ch, pgrid_config) -> (DofHandlerHierarchy, ConstraintHandlerHierarchy)

Build the full polynomial-order hierarchy for polynomial multigrid starting from the
fine-grid handler `dh` and constraint handler `ch`, using the projection strategy in
`pgrid_config`.

Returns a `(DofHandlerHierarchy, ConstraintHandlerHierarchy)` pair where index 1 is the
coarsest level (order 1) and index `end` is the finest level (`dh` and `ch`).

This is the standalone convenience function that underpins `pmultigrid(A, dh, ch, ...)`.
"""
function build_pmg_dofhandler_hierarchy(
        dh::AbstractDofHandler,
        ch::ConstraintHandler,
        pgrid_config::PMultigridConfiguration,
    )
    degree = order(dh)
    ps     = pgrid_config.proj_strategy
    step   = _calculate_step(ps, degree)

    # Build from finest to coarsest
    dhs = AbstractDofHandler[dh]
    chs = ConstraintHandler[ch]

    while degree > 1
        degree      = degree - step > 1 ? degree - step : 1
        fine_dh     = dhs[end]
        fine_ch     = chs[end]
        coarse_dh, coarse_ch = coarsen_order(fine_dh, fine_ch, degree)
        push!(dhs, coarse_dh)
        push!(chs, coarse_ch)
    end

    # Reverse so that index 1 = coarsest (consistent with GridHierarchy convention)
    return DofHandlerHierarchy(reverse(dhs)), ConstraintHandlerHierarchy(reverse(chs))
end

"""
    pmultigrid(A, dhh, chh, pgrid_config, pcoarse_solver, [Val{bs}]; kwargs...)

Build a polynomial multigrid preconditioner from a pre-built `DofHandlerHierarchy` and
`ConstraintHandlerHierarchy`.  Index 1 of the hierarchies must be the coarsest level
and index `end` the finest.

This is the primary dispatch; the convenience overload taking a single `DofHandler` calls
[`build_pmg_dofhandler_hierarchy`](@ref) and forwards here.
"""
function pmultigrid(
    A::TA,
    dhh::DofHandlerHierarchy,
    chh::ConstraintHandlerHierarchy,
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

    levels = Vector{Level{TA,TA,Adjoint{T,TA}}}()
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    cs    = pgrid_config.coarse_strategy
    cur_A = A

    # Iterate from finest (index end) down to coarsest (index 1)
    for k in n_levels:-1:1
        fine_dh   = dhh[k+1]
        fine_ch   = chh[k+1]
        coarse_dh = dhh[k]
        coarse_ch = chh[k]

        @timeit_debug "extend_hierarchy!" cur_A = extend_hierarchy!(
            levels, fine_dh, fine_ch, coarse_dh, coarse_ch, cur_A, cs, u, p)

        coarse_x!(w, size(cur_A, 1))
        coarse_b!(w, size(cur_A, 1))
        residual!(w, size(cur_A, 1))
    end

    coarse_solver = @timeit_debug "coarse solver setup" pcoarse_solver(cur_A)
    return MultiLevel(levels, cur_A, coarse_solver, presmoother, postsmoother, w)
end

"""
    pmultigrid(A, dh, ch, pgrid_config, pcoarse_solver, [Val{bs}]; kwargs...)

Convenience dispatch: builds the `DofHandlerHierarchy` / `ConstraintHandlerHierarchy`
automatically via [`build_pmg_dofhandler_hierarchy`](@ref) and then calls the
hierarchy-based `pmultigrid`.
"""
function pmultigrid(
    A::TA,
    dh::AbstractDofHandler,
    ch::ConstraintHandler,
    pgrid_config::PMultigridConfiguration,
    pcoarse_solver,
    ::Type{Val{bs}} = Val{1};
    kwargs...,
    ) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    dhh, chh = build_pmg_dofhandler_hierarchy(dh, ch, pgrid_config)
    return pmultigrid(A, dhh, chh, pgrid_config, pcoarse_solver, Val{bs}; kwargs...)
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
    @timeit_debug "assemble coarse operator" update_operator!(op, p)
    apply!(op.A, coarse_ch)
    A = op.A
    return A
end

function _calculate_step(ps::StepProjection, p::Int) 
    step = ps.step
    step ≥ p && error("Step must be less than the polynomial order $p")
    return step
end

_calculate_step(::DirectProjection, fine_p::Int) = fine_p - 1


