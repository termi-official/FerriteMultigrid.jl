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
    for (fine_sdh, coarse_sdh) in zip(fine_dh.subdofhandlers, coarse_dh.subdofhandlers)
        for tc in SameGridCellIterator(fine_sdh, coarse_sdh)
            for rdof in getrowdofs(tc)
                row_contrib[rdof] += 1
            end
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

