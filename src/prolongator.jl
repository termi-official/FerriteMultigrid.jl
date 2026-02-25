## P_ij = M^{-1}_{ij} ∫ ϕ_i(x) ⋅ ϕc_j(x) dx (ϕc is the coarse basis function)
function element_prolongator!(
    Pe::AbstractMatrix,
    Me::AbstractMatrix,
    fine_cv::AbstractCellValues,
    coarse_cv::AbstractCellValues,
    Pe_buffer::AbstractMatrix,
)
    fill!(Pe_buffer, zero(eltype(Pe)))
    n_fine_basefuncs = getnbasefunctions(fine_cv)
    n_coarse_basefuncs = getnbasefunctions(coarse_cv)

    for q = 1:getnquadpoints(fine_cv)
        dΩ = getdetJdV(fine_cv, q)
        for i = 1:n_fine_basefuncs
            ϕᵢ = shape_value(fine_cv, q, i)
            for j = 1:n_coarse_basefuncs
                ϕⱼ = shape_value(coarse_cv, q, j)
                Pe_buffer[i, j] += (ϕᵢ ⋅ ϕⱼ) * dΩ
            end
        end
    end

    # Invert the mass matrix to get the prolongator
    _element_mass_matrix!(Me, fine_cv)
    lu_fact = qr(Me)
    ldiv!(Pe, lu_fact, Pe_buffer)

    return drop_small_entries!(Pe)
end


function drop_small_entries!(A::AbstractMatrix, tol::Float64 = 1e-10)
    for ij in eachindex(A)
        if abs(A[ij]) < tol
            A[ij] = 0.0
        end
    end
    return A
end

## M_{ij} = ∫ ϕ_i(x) ⋅ ϕ_j(x) dx
function _element_mass_matrix!(Me::AbstractMatrix, cv::AbstractCellValues)
    fill!(Me, zero(eltype(Me)))
    n_basefuncs = getnbasefunctions(cv)
    for q = 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q)
        for i = 1:n_basefuncs
            δu = shape_value(cv, q, i)
            for j = 1:n_basefuncs
                u = shape_value(cv, q, j)
                Me[i, j] += (δu ⋅ u) * dΩ
            end
        end
    end
    return Me
end

"""
    build_prolongator(fine_dh::DofHandler, coarse_dh::DofHandler)

Assemble the prolongation matrix P of size `(ndofs(fine_dh) × ndofs(coarse_dh))`.

Uses a `MassProlongatorIntegrator` via the FerriteOperators transfer operator infrastructure.
For each cell the local prolongator is computed via an element-local L²-projection
(fine mass matrix inverse times cross-mass matrix), then assembled into the global matrix.
Rows are normalised by the number of element contributions to handle shared dofs correctly.
"""
function build_prolongator(fine_dh::DofHandler, coarse_dh::DofHandler)
    field_name  = first(Ferrite.getfieldnames(fine_dh))
    integrator  = MassProlongatorIntegrator(QuadratureRuleCollection(2 * order(fine_dh)), field_name)
    strategy    = SequentialAssemblyStrategy(SequentialCPUDevice())

    op = @timeit_debug "setup transfer operator" setup_transfer_operator(strategy, integrator, fine_dh, coarse_dh)
    @timeit_debug "assemble transfer operator" update_operator!(op, nothing)

    # Count how many element contributions each fine dof accumulated, then normalise.
    row_contrib = zeros(Int, ndofs(fine_dh))
    for tc in SameGridTransferCellIterator(fine_dh, coarse_dh)
        for rdof in getrowdofs(tc)
            row_contrib[rdof] += 1
        end
    end
    @timeit_debug "row normalization" normalize_rows!(op.P, row_contrib)

    return op.P
end

## Normalize rows of a CSC sparse matrix by contribution count.
## Iterates over CSC nonzeros directly — O(nnz) instead of O(nrows × ncols).
function normalize_rows!(P::SparseMatrixCSC, row_contrib::Vector{Int})
    rows = rowvals(P)
    vals = nonzeros(P)
    for j in 1:size(P, 2)
        for idx in nzrange(P, j)
            row = rows[idx]
            if row_contrib[row] > 1
                vals[idx] /= row_contrib[row]
            end
        end
    end
    return P
end

function build_restriction(coarse_dh, fine_dh, P, is_sym)
    if !is_sym
        return build_prolongator(coarse_dh, fine_dh)
    else
        return P'
    end
end

