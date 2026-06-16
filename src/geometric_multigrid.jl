## Geometric Multigrid on Nested Grids
##
## Provides uniform grid refinement and a full geometric multigrid method analogous
## to `pmultigrid` but driven by a hierarchy of grids produced by `uniform_refinement`.
##
## Key API:
##   uniform_refinement(grid)  → (fine_grid, fine2coarse, child_ref_coords)
##   GridHierarchy(grid, n_refinements)
##   gmultigrid_config(; coarse_strategy = Galerkin())
##   gmultigrid(A, gh, dh_hierarchy, ch_hierarchy, config, pcoarse_solver)

#######################################################################
## GridHierarchy                                                     ##
#######################################################################

"""
    GridHierarchy{G}

A hierarchy of nested grids produced by repeated `uniform_refinement`.

- `grids[1]`      is the coarsest grid
- `grids[end]`    is the finest grid
- `fine2coarse[k]`      maps fine cell ids on level `k+1` → coarse cell ids on level `k`
- `child_ref_coords[k]` stores the reference coordinates of fine-cell nodes in their
                        parent coarse reference element, for level transition `k+1 → k`
"""
struct GridHierarchy{G}
    grids::Vector{G}
    fine2coarse::Vector{Vector{Int}}
    child_ref_coords::Vector{Vector}   # child_ref_coords[k][fine_id] = Vector{Vec}
end

"""
    GridHierarchy(coarse_grid::Grid, n_refinements::Int)

Build a grid hierarchy with `n_refinements` levels of uniform refinement above
`coarse_grid`.  The resulting hierarchy has `n_refinements + 1` grids.
"""
function GridHierarchy(coarse_grid::Grid, n_refinements::Int)
    @assert n_refinements >= 1 "Need at least one refinement level"
    grids            = [coarse_grid]
    fine2coarse_maps = Vector{Int}[]
    crc_maps         = Vector[]

    for _ in 1:n_refinements
        fine_grid, f2c, crc = uniform_refinement(grids[end])
        push!(grids, fine_grid)
        push!(fine2coarse_maps, f2c)
        push!(crc_maps, crc)
    end

    return GridHierarchy(grids, fine2coarse_maps, crc_maps)
end

Base.length(gh::GridHierarchy) = length(gh.grids)

"""
    DofHandlerHierarchy(gh::GridHierarchy)

Allocate a `DofHandler` for each grid level in `gh` (coarsest first).
Fields must be added via `add!` and the hierarchy must be closed via `close!`
before use.
"""
DofHandlerHierarchy(gh::GridHierarchy) =
    DofHandlerHierarchy([DofHandler(g) for g in gh.grids])


#######################################################################
## Geometric multigrid configuration                                 ##
#######################################################################

"""
    GMultigridConfiguration{TC}

Configuration for the geometric multigrid method (analogous to `PMultigridConfiguration`).
"""
struct GMultigridConfiguration{TC<:AbstractCoarseningStrategy}
    coarse_strategy::TC
end

"""
    gmultigrid_config(; coarse_strategy = Galerkin())

Create a `GMultigridConfiguration`.
"""
gmultigrid_config(; coarse_strategy = Galerkin()) = GMultigridConfiguration(coarse_strategy)


#######################################################################
## Nodal prolongator for nested grids                               ##
#######################################################################

@doc raw"""
    NestedNodalProlongatorIntegrator

Integrator for assembling a prolongation operator between a fine and a coarse grid
that are **hierarchically nested** (geometric multigrid).

Unlike [`NestedMassProlongatorIntegrator`](@ref), this variant avoids the local mass
matrix and Cholesky solve entirely.  The coarse basis functions are evaluated directly
at the fine element's Lagrange node positions, which are first mapped to the coarse
reference frame using the same affine map as `NestedMassProlongatorIntegrator`:

```math
\xi_\text{coarse} = \sum_k N_k^{\text{geo,fine}}(\hat\xi_i)\;\hat{x}_k
```

where ``\hat\xi_i`` is the ``i``-th Lagrange node position in fine reference coordinates
and ``\hat{x}_k`` are the fine-cell corner positions in the parent (coarse) reference
element (`child_ref_coords`).  The element prolongation entry is then

```math
P_{ij} = \varphi_j^{\text{coarse}}(\xi_\text{coarse})
```

This is always well-posed regardless of how many quadrature points are available, and is
exact for Lagrange elements of any order on any reference shape.
"""
struct NestedNodalProlongatorIntegrator <: AbstractTransferIntegrator
    field_name::Symbol
end

"""
The element cache associated with [`NestedNodalProlongatorIntegrator`](@ref).
"""
struct NestedNodalProlongatorElementCache{IP_coarse, IP_geo, T} <: AbstractTransferElementCache
    ip_coarse::IP_coarse   # field interpolation of the coarse element
    ip_geo_fine::IP_geo    # scalar geometric interpolation of fine element (for ref-space map)
    positions::Vector{T}   # fine Lagrange node positions in fine reference coordinates
    vdim::Int
end

function FerriteOperators.duplicate_for_device(::Any, cache::NestedNodalProlongatorElementCache)
    return cache  # all fields are immutable; safe to share across threads
end

function FerriteOperators.setup_transfer_element_cache(
        integrator::NestedNodalProlongatorIntegrator,
        sdh_fine::SubDofHandler,
        sdh_coarse::SubDofHandler,
    )
    field_name  = integrator.field_name
    ip_fine     = Ferrite.getfieldinterpolation(sdh_fine,   field_name)
    ip_coarse   = Ferrite.getfieldinterpolation(sdh_coarse, field_name)
    ip_geo_fine = Ferrite.geometric_interpolation(typeof(FerriteOperators.get_first_cell(sdh_fine)))
    positions   = Ferrite.reference_coordinates(ip_fine)
    return NestedNodalProlongatorElementCache(ip_coarse, ip_geo_fine, positions, Ferrite.n_components(ip_fine))
end

function FerriteOperators.assemble_transfer_element!(
        Pₑ::AbstractMatrix,
        tc::NestedGridCellCache,
        cache::NestedNodalProlongatorElementCache,
        p,
    )
    (; ip_coarse, ip_geo_fine, positions, vdim) = cache
    child_nodes = get_child_ref_coords(tc)
    n_fine   = size(Pₑ, 1)
    n_coarse = size(Pₑ, 2)
    if vdim > 1
        @inbounds for i in 1:n_fine ÷ vdim
            ξ_fine = positions[i]
            # Map fine Lagrange node i to coarse reference coordinates via the affine map
            # defined by the fine element's geometric interpolation and child_ref_coords.
            ξ_coarse = sum(
                Ferrite.reference_shape_value(ip_geo_fine, ξ_fine, k) * child_nodes[k]
                for k in eachindex(child_nodes)
            )
            for j in 1:n_coarse
                val = Ferrite.reference_shape_value(ip_coarse, ξ_coarse, j)
                for k in 1:vdim
                    Pₑ[vdim*(i-1)+k, j] += val[k]
                end
            end
        end
    else
        @inbounds for i in 1:n_fine
            ξ_fine = positions[i]
            # Map fine Lagrange node i to coarse reference coordinates via the affine map
            # defined by the fine element's geometric interpolation and child_ref_coords.
            ξ_coarse = sum(
                Ferrite.reference_shape_value(ip_geo_fine, ξ_fine, k) * child_nodes[k]
                for k in eachindex(child_nodes)
            )
            for j in 1:n_coarse
                Pₑ[i, j] = Ferrite.reference_shape_value(ip_coarse, ξ_coarse, j)
            end
        end
    end
end


#######################################################################
## Prolongator assembly for nested grids                             ##
#######################################################################

"""
    build_geometric_prolongator(dh_fine, dh_coarse, fine2coarse, child_ref_coords)

Assemble the prolongation matrix P for a geometric level transition using
`NestedNodalProlongatorIntegrator` via FerriteOperators.

The prolongation is computed by direct nodal interpolation: for each fine Lagrange
node, the coarse basis functions are evaluated at the corresponding coarse reference
coordinate (obtained by mapping through `child_ref_coords`).  This avoids the local
mass-matrix solve of `NestedMassProlongatorIntegrator` and is well-posed for all
supported element types including Wedge.
"""
function build_geometric_prolongator(
        dh_fine::DofHandler,
        dh_coarse::DofHandler,
        fine2coarse::AbstractVector{Int},
        child_ref_coords::AbstractVector
    )
    field_name = first(Ferrite.getfieldnames(dh_fine))
    # FIXME multi-field support
    @assert length(dh_fine.field_names) == 1 "Multiple fields not yet supported"
    integrator = NestedNodalProlongatorIntegrator(field_name)
    strategy   = SequentialAssemblyStrategy(SequentialCPUDevice())

    op = setup_nested_transfer_operator(strategy, integrator,
                                        dh_fine, dh_coarse, fine2coarse, child_ref_coords)
    update_operator!(op, nothing)

    # Normalise rows: a fine dof shared by multiple fine cells accumulates multiple
    # element contributions; each row must be divided by the contribution count.
    row_contrib = zeros(Int, ndofs(dh_fine))
    for (fine_sdh, coarse_sdh) in zip(dh_fine.subdofhandlers, dh_coarse.subdofhandlers)
        for tc in NestedGridCellIterator(fine_sdh, coarse_sdh, fine2coarse, child_ref_coords)
            for rdof in getrowdofs(tc)
                row_contrib[rdof] += 1
            end
        end
    end
    normalize_rows!(op.P, row_contrib)

    return op.P
end


#######################################################################
## gmultigrid                                                        ##
#######################################################################

_gmg_coarse_matrix(A, P, R, ::Galerkin, dh_coarse, ch_coarse, u, p, rap_ws::RAPWorkspace) =
    rap_numeric!(rap_ws, A)

function _gmg_coarse_matrix(A, P, R, cs::Rediscretization, dh_coarse, ch_coarse, u, p, ::Nothing)
    op = setup_operator(cs.strategy, cs.integrator, dh_coarse)
    update_operator!(op, p)
    ch_coarse !== nothing && apply!(op.A, ch_coarse)
    return op.A
end

"""
    gmultigrid_symbolic(gh, dhh, config, A)

Build the [`MultilevelGeometry`](@ref) for a geometric multigrid hierarchy with
the Galerkin coarsening strategy.

Assembles all prolongation/restriction operators and, using the fine-grid matrix
`A`, pre-allocates one [`RAPWorkspace`](@ref) per level (chaining coarse matrices
through the hierarchy).

Index 1 of `dhh` (and `gh`) must be the coarsest level and index `end` the finest.
"""
function gmultigrid_symbolic(
        gh::GridHierarchy,
        dhh::DofHandlerHierarchy,
        config::GMultigridConfiguration{<:Galerkin},
        A::SparseMatrixCSC,
    )
    n_levels = length(gh) - 1
    @assert n_levels >= 1
    @assert length(dhh) == length(gh)

    P0 = @timeit_debug "build geometric prolongator" build_geometric_prolongator(
            dhh[n_levels+1], dhh[n_levels], gh.fine2coarse[n_levels], gh.child_ref_coords[n_levels])
    R0 = @timeit_debug "build geometric restriction" P0'

    pairs = Vector{Tuple{typeof(P0), typeof(R0)}}()
    push!(pairs, (P0, R0))

    for k in n_levels-1:-1:1
        P = @timeit_debug "build geometric prolongator" build_geometric_prolongator(
                dhh[k+1], dhh[k], gh.fine2coarse[k], gh.child_ref_coords[k])
        R = @timeit_debug "build geometric restriction" P'
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
    gmultigrid_symbolic(gh, dhh, config)

Build the [`MultilevelGeometry`](@ref) for a geometric multigrid hierarchy with
the Rediscretization coarsening strategy (no RAP workspaces).
"""
function gmultigrid_symbolic(
        gh::GridHierarchy,
        dhh::DofHandlerHierarchy,
        config::GMultigridConfiguration{<:Rediscretization},
    )
    n_levels = length(gh) - 1
    @assert n_levels >= 1
    @assert length(dhh) == length(gh)

    P0 = @timeit_debug "build geometric prolongator" build_geometric_prolongator(
            dhh[n_levels+1], dhh[n_levels], gh.fine2coarse[n_levels], gh.child_ref_coords[n_levels])
    R0 = @timeit_debug "build geometric restriction" P0'

    pairs = Vector{Tuple{typeof(P0), typeof(R0)}}()
    push!(pairs, (P0, R0))

    for k in n_levels-1:-1:1
        P = @timeit_debug "build geometric prolongator" build_geometric_prolongator(
                dhh[k+1], dhh[k], gh.fine2coarse[k], gh.child_ref_coords[k])
        R = @timeit_debug "build geometric restriction" P'
        push!(pairs, (P, R))
    end

    return MultilevelGeometry(pairs, nothing)
end

"""
    gmultigrid_numeric!(geo, A, gh, dhh, chh, config, pcoarse_solver; kwargs...)

Build a geometric multigrid `MultiLevel` using a pre-built [`MultilevelGeometry`](@ref).
Performs only the numeric phase (smoother setup + coarse-matrix computation).
"""
function gmultigrid_numeric!(
        geo::MultilevelGeometry,
        A::SparseMatrixCSC{T},
        gh::GridHierarchy,
        dhh::DofHandlerHierarchy,
        chh::Union{ConstraintHandlerHierarchy, Nothing},
        config::GMultigridConfiguration,
        pcoarse_solver;
        u          = nothing,
        p          = nothing,
        presmoother  = GaussSeidel(),
        postsmoother = GaussSeidel(),
        symmetry     = AMG.HermitianSymmetry(),
    ) where T

    n_levels = length(gh) - 1
    @assert n_levels >= 1
    @assert length(geo.levels) == n_levels
    @assert length(dhh) == length(gh)
    chh !== nothing && @assert length(chh) == length(gh)

    TP, TR = fieldtypes(eltype(geo.levels))
    TA = SparseMatrixCSC{T, Int}
    levels = Vector{Level{TA, TP, TR}}()
    w      = MultiLevelWorkspace(Val{1}, T)
    residual!(w, size(A, 1))

    cs    = config.coarse_strategy
    cur_A = A

    ch_coarse = nothing
    for (i, (P, R)) in enumerate(geo.levels)
        coarse_idx = n_levels + 1 - i
        dh_coarse  = dhh[coarse_idx]
        ch_coarse  = chh !== nothing ? chh[coarse_idx] : nothing

        @timeit_debug "smoother setup" begin
            pre  = AMG.setup_smoother(presmoother, cur_A, symmetry)
            post = AMG.setup_smoother(postsmoother, cur_A, symmetry)
            push!(levels, Level(cur_A, P, R, pre, post))
        end

        rap_ws = geo.rap_workspaces !== nothing ? geo.rap_workspaces[i] : nothing
        cur_A = @timeit_debug "coarse matrix" _gmg_coarse_matrix(
            cur_A, P, R, cs, dh_coarse, ch_coarse, u, p, rap_ws)

        coarse_x!(w, size(cur_A, 1))
        coarse_b!(w, size(cur_A, 1))
        residual!(w, size(cur_A, 1))
    end

    coarse_solver = @timeit_debug "coarse solver setup" pcoarse_solver(cur_A)
    return MultiLevel(levels, cur_A, coarse_solver, presmoother, postsmoother, w)
end

"""
    gmultigrid(A, gh, dhh, chh, config, pcoarse_solver; kwargs...)

Build a geometric multigrid preconditioner / solver for `Ax = b`.

This is a convenience wrapper that calls [`gmultigrid_symbolic`](@ref) followed by
[`gmultigrid_numeric!`](@ref).  When rebuilding across Newton iterations, prefer
caching the [`MultilevelGeometry`](@ref) from `gmultigrid_symbolic` and calling
`gmultigrid_numeric!` directly.

# Arguments
- `A`             – assembled fine-grid matrix
- `gh`            – [`GridHierarchy`](@ref) (from coarse to fine)
- `dhh`           – [`DofHandlerHierarchy`](@ref), one handler per grid level (index 1 = coarsest)
- `chh`           – [`ConstraintHandlerHierarchy`](@ref), one handler per grid level
- `config`        – [`GMultigridConfiguration`](@ref)
- `pcoarse_solver` – callable that returns a coarse-grid solver given the coarse matrix

# Keyword arguments
- `p`            – parameter passed to `update_operator!` (default `nothing`)
- `presmoother`  / `postsmoother` – AlgebraicMultigrid smoother (default `GaussSeidel()`)

# Coarsening strategies
- `Galerkin()` – coarse-grid matrix = R A P  (Galerkin projection)
- `Rediscretization(integrator)` – re-assembles the operator on each coarse grid
"""
function gmultigrid(
        A::SparseMatrixCSC{T},
        gh::GridHierarchy,
        dhh::DofHandlerHierarchy,
        chh::Union{ConstraintHandlerHierarchy, Nothing},
        config::GMultigridConfiguration{<:Galerkin},
        pcoarse_solver;
        kwargs...,
    ) where T

    geo = @timeit_debug "gmultigrid symbolic" gmultigrid_symbolic(gh, dhh, config, A)
    return @timeit_debug "gmultigrid numeric" gmultigrid_numeric!(geo, A, gh, dhh, chh, config, pcoarse_solver; kwargs...)
end

function gmultigrid(
        A::SparseMatrixCSC{T},
        gh::GridHierarchy,
        dhh::DofHandlerHierarchy,
        chh::Union{ConstraintHandlerHierarchy, Nothing},
        config::GMultigridConfiguration{<:Rediscretization},
        pcoarse_solver;
        kwargs...,
    ) where T

    geo = @timeit_debug "gmultigrid symbolic" gmultigrid_symbolic(gh, dhh, config)
    return @timeit_debug "gmultigrid numeric" gmultigrid_numeric!(geo, A, gh, dhh, chh, config, pcoarse_solver; kwargs...)
end

"""
    init(A, b, gh, dhh, chh, config; pcoarse_solver, kwargs...) -> MGSolver

Build a geometric multigrid solver and return an [`MGSolver`](@ref).
"""
function init(A, b,
              gh::GridHierarchy,
              dhh::DofHandlerHierarchy, chh::ConstraintHandlerHierarchy,
              config::GMultigridConfiguration;
              pcoarse_solver  = SmoothedAggregationCoarseSolver(),
              p               = nothing,
              presmoother     = GaussSeidel(),
              postsmoother    = GaussSeidel(),
              kwargs...)
    ml = gmultigrid(A, gh, dhh, chh, config, pcoarse_solver;
                    p, presmoother, postsmoother)
    return MGSolver(ml, b)
end

"""
    solve(A, b, gh, dhh, chh, config; pcoarse_solver, kwargs...)

High-level geometric multigrid solve for `Ax = b`.
`kwargs` are forwarded to the iterative solve (e.g. `maxiter`, `reltol`, `log`).
"""
function solve(A::AbstractMatrix, b::AbstractVector,
               gh::GridHierarchy,
               dhh::DofHandlerHierarchy, chh::ConstraintHandlerHierarchy,
               config::GMultigridConfiguration;
               pcoarse_solver = SmoothedAggregationCoarseSolver(), kwargs...)
    @timeit_debug "init"   solver = init(A, b, gh, dhh, chh, config; pcoarse_solver, kwargs...)
    @timeit_debug "solve!" solve!(solver; kwargs...)
end
