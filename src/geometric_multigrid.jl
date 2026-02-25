## Geometric Multigrid on Nested Grids – 1D Diffusion Sketch
##
## This file provides the building blocks for a geometric multigrid method on
## hierarchically nested grids.  The implementation focuses on the 1D diffusion
## problem as an illustrative starting point.
##
## Key concepts:
##   - `NestedGrid1D` – builds a coarse/fine grid pair and the fine→coarse mapping.
##   - `build_geometric_prolongator_1d` – assembles P using `NestedGridTransferCellIterator`.
##   - `gmultigrid` – top-level geometric multigrid V-cycle solver (sketch).

#######################################################################
## Nested grid data structure (1D)                                   ##
#######################################################################

"""
    NestedGrid1D

Stores a coarse and fine 1D grid (Line elements) together with the mapping from each fine
cell to its parent coarse cell and the reference coordinates of the fine cell's nodes
inside the parent element.

# Fields
- `coarse_grid` / `fine_grid`     – Ferrite `Grid`s
- `coarse_dh` / `fine_dh`         – `DofHandler`s (scalar field `:u`, Lagrange P1)
- `fine2coarse :: Vector{Int}`     – `fine2coarse[fine_id] = coarse_id`
- `child_ref_coords`               – for each fine cell, the reference coords of its two
                                     nodes inside the parent coarse reference element [-1,1]
"""
struct NestedGrid1D
    coarse_grid
    fine_grid
    coarse_dh::DofHandler
    fine_dh::DofHandler
    fine2coarse::Vector{Int}
    child_ref_coords::Vector{Vector{Vec{1,Float64}}}
end

"""
    NestedGrid1D(N_coarse::Int; xₗ = 0.0, xᵣ = 1.0, order = 1)

Build a uniform 1D coarse grid with `N_coarse` cells and a fine grid with `2*N_coarse`
cells (uniform bisection).  The mapping and reference coordinates are computed
automatically.

`order` controls the Lagrange interpolation order for both grids.
"""
function NestedGrid1D(N_coarse::Int; xₗ::Float64 = 0.0, xᵣ::Float64 = 1.0, order::Int = 1)
    # Build grids
    coarse_grid = generate_grid(Line, (N_coarse,),        Vec((xₗ,)), Vec((xᵣ,)))
    fine_grid   = generate_grid(Line, (2 * N_coarse,),   Vec((xₗ,)), Vec((xᵣ,)))

    # DofHandlers (scalar `:u`, Lagrange Pk)
    coarse_dh = _make_1d_dh(coarse_grid, order)
    fine_dh   = _make_1d_dh(fine_grid,   order)

    # Build fine→coarse mapping.
    # Uniform bisection: fine cell i is child of coarse cell ceil(i/2).
    N_fine = 2 * N_coarse
    fine2coarse = [div(i - 1, 2) + 1 for i in 1:N_fine]

    # Reference coordinates of the fine cell's nodes inside the parent coarse element.
    # In the reference interval [-1, 1]:
    #   - odd fine cell (left child):  nodes at (-1, 0)
    #   - even fine cell (right child): nodes at (0, +1)
    child_ref_coords = Vector{Vector{Vec{1,Float64}}}(undef, N_fine)
    for i in 1:N_fine
        if isodd(i)
            child_ref_coords[i] = [Vec((-1.0,)), Vec((0.0,))]
        else
            child_ref_coords[i] = [Vec((0.0,)),  Vec((1.0,))]
        end
    end

    return NestedGrid1D(coarse_grid, fine_grid, coarse_dh, fine_dh, fine2coarse, child_ref_coords)
end

function _make_1d_dh(grid, order::Int)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, order}())
    close!(dh)
    return dh
end


#######################################################################
## Geometric prolongator assembly (1D)                               ##
#######################################################################

"""
    build_geometric_prolongator_1d(ng::NestedGrid1D)

Assemble the prolongation matrix P of size `(ndofs(ng.fine_dh) × ndofs(ng.coarse_dh))`
for a P1 geometric multigrid method on nested 1D grids.

The prolongation is computed via element-local L²-projection using
[`NestedGridTransferCellIterator`](@ref): for each fine cell the coarse basis functions are
evaluated at the fine quadrature points using the pre-computed reference coordinates
`child_ref_coords`.
"""
function build_geometric_prolongator_1d(ng::NestedGrid1D; qr_order::Int = 2)
    fine_dh   = ng.fine_dh
    coarse_dh = ng.coarse_dh
    fine_ndofs   = ndofs(fine_dh)
    coarse_ndofs = ndofs(coarse_dh)
    P = spzeros(fine_ndofs, coarse_ndofs)
    row_contrib = zeros(Int, fine_ndofs)

    # CellValues for fine and coarse spaces
    ip_fine   = Ferrite.getfieldinterpolation(fine_dh.subdofhandlers[1],   :u)
    ip_coarse = Ferrite.getfieldinterpolation(coarse_dh.subdofhandlers[1], :u)
    ip_geo    = Ferrite.geometric_interpolation(Line)
    qr        = QuadratureRule{RefLine}(qr_order)
    fine_cv   = CellValues(qr, ip_fine,   ip_geo)
    coarse_cv = CellValues(qr, ip_coarse, ip_geo)

    n_fine_basis   = getnbasefunctions(fine_cv)
    n_coarse_basis = getnbasefunctions(coarse_cv)
    Pe        = zeros(n_fine_basis, n_coarse_basis)
    Pe_buffer = zeros(n_fine_basis, n_coarse_basis)
    Me        = zeros(n_fine_basis, n_fine_basis)

    for tc in NestedGridTransferCellIterator(fine_dh, coarse_dh,
                                             ng.fine2coarse, ng.child_ref_coords)
        # Reinit fine CellValues at fine cell geometry
        reinit!(fine_cv, getcells(ng.fine_grid, tc.fine_cellid), get_fine_coordinates(tc))

        # For the coarse CellValues we need to evaluate coarse basis functions at the
        # physical positions of the fine quadrature points.  We do this by constructing
        # a custom CellValues reinit using the child reference coordinates mapped through
        # the coarse reference element.
        _reinit_coarse_cv_at_child!(coarse_cv, tc, ng.coarse_grid)

        # Element prolongator: P_e = M_e^{-1} * ∫ φ_fine ⊗ φ_coarse dΩ
        element_prolongator!(Pe, Me, fine_cv, coarse_cv, Pe_buffer)

        rdofs = getrowdofs(tc)
        cdofs = getcolumndofs(tc)
        for i in 1:n_fine_basis
            gi = rdofs[i]
            row_contrib[gi] += 1
            for j in 1:n_coarse_basis
                P[gi, cdofs[j]] += Pe[i, j]
            end
        end
    end
    normalize_rows!(P, row_contrib)
    return P
end

"""
    _reinit_coarse_cv_at_child!(coarse_cv, tc, coarse_grid)

Reinitialise `coarse_cv` so that its quadrature points coincide with those of the fine
child element.  The fine quadrature points are expressed in the coarse reference element
via `get_child_ref_coords(tc)` (a linear map for uniform bisection).

For the P1 / geometric-multigrid 1D case the fine cell's nodes span exactly one half of
the coarse reference element, so the quadrature rule on the fine cell maps to a sub-interval
of [-1,1] in the coarse reference frame.
"""
function _reinit_coarse_cv_at_child!(coarse_cv::CellValues, tc::NestedGridTransferCellCache, coarse_grid)
    child_nodes = get_child_ref_coords(tc)   # [ξ_left, ξ_right] in coarse reference coords
    # Build a custom quadrature rule at the mapped fine quadrature points
    fine_qr   = coarse_cv.qr  # same quadrature rule used for the fine cv
    ξ_left  = child_nodes[1][1]
    ξ_right = child_nodes[2][1]
    # Affine map: fine reference coord η ∈ [-1,1] → coarse reference coord ξ
    #   ξ = ξ_left + (ξ_right - ξ_left) / 2 * (η + 1)
    mapped_points  = [Vec(((ξ_left + ξ_right)/2 + (ξ_right - ξ_left)/2 * η[1],))
                      for η in fine_qr.points]
    scaled_weights = [(ξ_right - ξ_left)/2 * w for w in fine_qr.weights]
    mapped_qr = QuadratureRule{RefLine}(scaled_weights, mapped_points)
    # Temporarily build new CellValues with mapped quadrature (cheap for 1D)
    ip_coarse = coarse_cv.fun_values.ip
    ip_geo    = coarse_cv.geo_mapping.ip
    tmp_cv    = CellValues(mapped_qr, ip_coarse, ip_geo)
    reinit!(tmp_cv, getcells(coarse_grid, tc.coarse_cellid), get_coarse_coordinates(tc))
    # Copy evaluated values into coarse_cv's buffers
    # NOTE: This is a sketch; a production implementation would avoid the allocation by
    # pre-computing the coarse shape values at the mapped points during operator setup.
    copyto!(coarse_cv.fun_values.Nx,  tmp_cv.fun_values.Nx)
    copyto!(coarse_cv.fun_values.dNdx, tmp_cv.fun_values.dNdx)
    copyto!(coarse_cv.detJdV, tmp_cv.detJdV)
    return coarse_cv
end


#######################################################################
## Geometric multigrid V-cycle sketch                                ##
#######################################################################

"""
    gmultigrid(A, dh, ch, integrator; kwargs...)

Build a two-level geometric multigrid preconditioner/solver for the system `Ax = b`
assembled on a 1D uniform grid `dh`.

This is a **sketch** intended to demonstrate the use of `NestedGridTransferCellIterator`
and `build_geometric_prolongator_1d`.  It builds one coarse level by uniform bisection
and delegates the coarse-grid solve to an AMG coarse solver.

# Arguments
- `A`          – assembled fine-grid matrix
- `dh`         – fine-grid `DofHandler` (1D, scalar `:u`, P1)
- `ch`         – fine-grid `ConstraintHandler`
- `integrator` – an `AbstractBilinearIntegrator` (e.g. `DiffusionMultigrid(1.0)`)
                 used to assemble the coarse-grid matrix via `FerriteOperators`.

Returns a `MultiLevel` object compatible with `AlgebraicMultigrid._solve`.
"""
function gmultigrid(
        A::SparseMatrixCSC{T},
        ng::NestedGrid1D,
        integrator::AbstractBilinearIntegrator,
        coarse_solver_type = SmoothedAggregationCoarseSolver;
        presmoother  = GaussSeidel(),
        postsmoother = GaussSeidel(),
    ) where {T}

    # Build transfer operators
    P = build_geometric_prolongator_1d(ng)
    R = P'   # symmetric restriction

    # Assemble coarse-grid operator via FerriteOperators
    strategy  = SequentialAssemblyStrategy(SequentialCPUDevice())
    coarse_op = setup_operator(strategy, integrator, ng.coarse_dh)
    update_operator!(coarse_op, nothing)
    # Apply coarse Dirichlet BCs (build a ConstraintHandler matching the coarse grid)
    # NOTE: In a real implementation the user would supply coarse_ch.  Here we just use
    # the assembled matrix directly (BCs must have been applied to the fine problem).
    A_coarse = coarse_op.A

    # Wrap in AlgebraicMultigrid Level
    w = MultiLevelWorkspace(Val{1}, T)
    residual!(w, size(A, 1))
    coarse_x!(w, size(A_coarse, 1))
    coarse_b!(w, size(A_coarse, 1))

    cs = coarse_solver_type()(A_coarse)

    level = Level(A, P, R)
    return MultiLevel([level], A_coarse, cs, presmoother, postsmoother, w)
end
