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
## Prolongator assembly for nested grids                             ##
#######################################################################

"""
    build_geometric_prolongator(dh_fine, dh_coarse, fine2coarse, child_ref_coords)

Assemble the prolongation matrix P for a geometric level transition using
`NestedMassProlongatorIntegrator` via FerriteOperators.
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
    qr_order   = 2 * getorder(dh_fine.subdofhandlers[1].field_interpolations[1])
    integrator = NestedMassProlongatorIntegrator(QuadratureRuleCollection(qr_order), field_name)
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

_gmg_coarse_matrix(A, P, R, ::Galerkin, dh_coarse, ch_coarse, u, p) = rap_threaded(R, A, P) # fused Galerkin projection

function _gmg_coarse_matrix(A, P, R, cs::Rediscretization, dh_coarse, ch_coarse, u, p)
    op = setup_operator(cs.strategy, cs.integrator, dh_coarse)
    update_operator!(op, p)
    ch_coarse !== nothing && apply!(op.A, ch_coarse)
    return op.A
end

"""
    gmultigrid_symbolic(gh, dhh)

Build the [`MultilevelGeometry`](@ref) for a geometric multigrid hierarchy.
Assembles all prolongation and restriction operators once from the grid hierarchy;
these can be reused across Newton iterations via [`gmultigrid_numeric!`](@ref).
"""
function gmultigrid_symbolic(
        gh::GridHierarchy,
        dhh::DofHandlerHierarchy,
    )
    n_levels = length(gh) - 1
    @assert n_levels >= 1
    @assert length(dhh) == length(gh)

    # Build first pair to establish concrete types.
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

    return MultilevelGeometry(pairs)
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

        cur_A = @timeit_debug "coarse matrix" _gmg_coarse_matrix(
            cur_A, P, R, cs, dh_coarse, ch_coarse, u, p)

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
        config::GMultigridConfiguration,
        pcoarse_solver;
        kwargs...,
    ) where T

    geo = @timeit_debug "gmultigrid symbolic" gmultigrid_symbolic(gh, dhh)
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
