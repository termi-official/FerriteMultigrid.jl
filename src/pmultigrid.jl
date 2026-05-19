## defines how we obtain A operator for the coarse grid
abstract type AbstractCoarseningStrategy end

"""
    MultilevelGeometry{TP, TR, RW}

Stores the prolongation and restriction operators for all level transitions in a
multigrid hierarchy.  These operators depend only on the mesh/polynomial structure
and can therefore be built once and reused across repeated numeric phases (i.e.
across Newton iterations).

For Galerkin coarsening `RW` is a `Vector{RAPWorkspace}` that pre-allocates the
sparsity pattern of each coarse-grid matrix.  For Rediscretization `RW` is `Nothing`.

`levels[1]` corresponds to the finest → next-coarser transition,
`levels[end]` to the second-coarsest → coarsest transition.
"""
struct MultilevelGeometry{TP, TR, RW}
    levels::Vector{Tuple{TP, TR}}
    rap_workspaces::RW
end

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
    pmultigrid_symbolic(dhh, pgrid_config, A)

Build the [`MultilevelGeometry`](@ref) for a polynomial multigrid hierarchy with
the Galerkin coarsening strategy.

Assembles all prolongation/restriction operators and, using the fine-grid matrix
`A`, pre-allocates one [`RAPWorkspace`](@ref) per level (chaining coarse matrices
through the hierarchy).  The resulting geometry can be passed directly to
[`pmultigrid_numeric!`](@ref) for all subsequent Newton iterations without any
further allocation.

Index 1 of `dhh` must be the coarsest level and index `end` the finest.
"""
function pmultigrid_symbolic(
        dhh::DofHandlerHierarchy,
        pgrid_config::PMultigridConfiguration{<:Galerkin},
        A::SparseMatrixCSC,
    )
    n_levels = length(dhh) - 1
    @assert n_levels >= 1 "DofHandlerHierarchy must have at least 2 levels"

    cs     = pgrid_config.coarse_strategy
    is_sym = cs.is_sym

    P0 = @timeit_debug "build prolongator" build_prolongator(dhh[n_levels+1], dhh[n_levels])
    R0 = @timeit_debug "build restriction" build_restriction(dhh[n_levels], dhh[n_levels+1], P0, is_sym)

    pairs = Vector{Tuple{typeof(P0), typeof(R0)}}()
    push!(pairs, (P0, R0))

    for k in n_levels-1:-1:1
        P = @timeit_debug "build prolongator" build_prolongator(dhh[k+1], dhh[k])
        R = @timeit_debug "build restriction" build_restriction(dhh[k], dhh[k+1], P, is_sym)
        push!(pairs, (P, R))
    end

    workspaces = Vector{RAPWorkspace}(undef, n_levels)
    cur_A = A
    for (i, (P, R)) in enumerate(pairs)
        ws = @timeit_debug "RAP symbolic" rap_symbolic(R, cur_A, P)
        workspaces[i] = ws
        cur_A = ws.C
    end

    return MultilevelGeometry(pairs, workspaces)
end

"""
    pmultigrid_symbolic(dhh, pgrid_config)

Build the [`MultilevelGeometry`](@ref) for a polynomial multigrid hierarchy with
the Rediscretization coarsening strategy.

Only the prolongation/restriction operators are assembled; coarse matrices are
re-assembled from scratch on each numeric call.

Index 1 of `dhh` must be the coarsest level and index `end` the finest.
"""
function pmultigrid_symbolic(
        dhh::DofHandlerHierarchy,
        pgrid_config::PMultigridConfiguration{<:Rediscretization},
    )
    n_levels = length(dhh) - 1
    @assert n_levels >= 1 "DofHandlerHierarchy must have at least 2 levels"

    cs     = pgrid_config.coarse_strategy
    is_sym = cs.is_sym

    P0 = @timeit_debug "build prolongator" build_prolongator(dhh[n_levels+1], dhh[n_levels])
    R0 = @timeit_debug "build restriction" build_restriction(dhh[n_levels], dhh[n_levels+1], P0, is_sym)

    pairs = Vector{Tuple{typeof(P0), typeof(R0)}}()
    push!(pairs, (P0, R0))

    for k in n_levels-1:-1:1
        P = @timeit_debug "build prolongator" build_prolongator(dhh[k+1], dhh[k])
        R = @timeit_debug "build restriction" build_restriction(dhh[k], dhh[k+1], P, is_sym)
        push!(pairs, (P, R))
    end

    return MultilevelGeometry(pairs, nothing)
end

function _pmg_coarse_matrix(cs::Galerkin, R, A, P, coarse_dh, coarse_ch, u, p, rap_ws::RAPWorkspace)
    return @timeit_debug "RAP numeric" rap_numeric!(rap_ws, A)
end

function _pmg_coarse_matrix(cs::Rediscretization, R, A, P, coarse_dh, coarse_ch, u, p, ::Nothing)
    op = @timeit_debug "setup coarse operator" setup_operator(cs.strategy, cs.integrator, coarse_dh)
    @timeit_debug "assemble coarse operator" update_operator!(op, p)
    coarse_ch !== nothing && apply!(op.A, coarse_ch)
    return op.A
end

"""
    pmultigrid_numeric!(geo, A, dhh, chh, pgrid_config, pcoarse_solver, [Val{bs}]; kwargs...)

Build a polynomial multigrid `MultiLevel` using a pre-built [`MultilevelGeometry`](@ref).
This performs only the numeric phase (smoother setup, Galerkin projection or re-assembly)
and can be called repeatedly with the same `geo` to rebuild the hierarchy cheaply when
only the matrix `A` changes (e.g. across Newton iterations).
"""
function pmultigrid_numeric!(
    geo::MultilevelGeometry,
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
    symmetry = AMG.HermitianSymmetry(),
    kwargs...,
    ) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    n_levels = length(dhh) - 1
    @assert n_levels >= 1
    @assert length(geo.levels) == n_levels "Geometry has $(length(geo.levels)) levels but dhh implies $n_levels"
    chh !== nothing && @assert length(dhh) == length(chh)

    TP, TR = fieldtypes(eltype(geo.levels))
    levels = Vector{Level{TA, TP, TR}}()
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    cs    = pgrid_config.coarse_strategy
    cur_A = A

    # geo.levels[1] = finest transition (dhh[end] → dhh[end-1]), ...
    for (i, (P, R)) in enumerate(geo.levels)
        coarse_idx = n_levels + 1 - i   # index of coarse level in dhh
        coarse_dh  = dhh[coarse_idx]
        coarse_ch  = chh !== nothing ? chh[coarse_idx] : nothing

        @timeit_debug "smoother setup" begin
            pre  = AMG.setup_smoother(presmoother, cur_A, symmetry)
            post = AMG.setup_smoother(postsmoother, cur_A, symmetry)
            push!(levels, Level(cur_A, P, R, pre, post))
        end

        rap_ws = geo.rap_workspaces !== nothing ? geo.rap_workspaces[i] : nothing
        cur_A = _pmg_coarse_matrix(cs, R, cur_A, P, coarse_dh, coarse_ch, u, p, rap_ws)

        coarse_x!(w, size(cur_A, 1))
        coarse_b!(w, size(cur_A, 1))
        residual!(w, size(cur_A, 1))
    end

    coarse_solver = @timeit_debug "coarse solver setup" pcoarse_solver(cur_A)
    return MultiLevel(levels, cur_A, coarse_solver, presmoother, postsmoother, w)
end

"""
    pmultigrid(A, dhh, chh, pgrid_config, pcoarse_solver, [Val{bs}]; kwargs...)

Build a polynomial multigrid preconditioner from a pre-built `DofHandlerHierarchy` and
`ConstraintHandlerHierarchy`.  Index 1 of the hierarchies must be the coarsest level
and index `end` the finest.

This is a convenience wrapper that calls [`pmultigrid_symbolic`](@ref) followed by
[`pmultigrid_numeric!`](@ref).  When rebuilding the hierarchy across Newton iterations,
prefer caching the result of `pmultigrid_symbolic` and calling `pmultigrid_numeric!`
directly to avoid redundant prolongator/restrictor assembly.
"""
function pmultigrid(
    A::TA,
    dhh::DofHandlerHierarchy,
    chh::Union{ConstraintHandlerHierarchy, Nothing},
    pgrid_config::PMultigridConfiguration{<:Galerkin},
    pcoarse_solver,
    ::Type{Val{bs}} = Val{1};
    kwargs...,
    ) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    geo = @timeit_debug "pmultigrid symbolic" pmultigrid_symbolic(dhh, pgrid_config, SparseMatrixCSC(A))
    return @timeit_debug "pmultigrid numeric" pmultigrid_numeric!(geo, A, dhh, chh, pgrid_config, pcoarse_solver, Val{bs}; kwargs...)
end

function pmultigrid(
    A::TA,
    dhh::DofHandlerHierarchy,
    chh::Union{ConstraintHandlerHierarchy, Nothing},
    pgrid_config::PMultigridConfiguration{<:Rediscretization},
    pcoarse_solver,
    ::Type{Val{bs}} = Val{1};
    kwargs...,
    ) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    geo = @timeit_debug "pmultigrid symbolic" pmultigrid_symbolic(dhh, pgrid_config)
    return @timeit_debug "pmultigrid numeric" pmultigrid_numeric!(geo, A, dhh, chh, pgrid_config, pcoarse_solver, Val{bs}; kwargs...)
end
