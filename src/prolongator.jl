struct PolynomialProlongationIntegrator <: AbstractTransferIntegrator
    field_name::Symbol
end

struct NodalPolynomialProlongationElementCache{CV <: CellValues} <: AbstractTransferElementCache
    cv::CV
    vdim::Int
end

function FerriteOperators.duplicate_for_device(device, cache::NodalPolynomialProlongationElementCache)
    return NodalPolynomialProlongationElementCache(
        FerriteOperators.duplicate_for_device(device, cache.cv),
        cache.vdim,
    )
end

function FerriteOperators.assemble_transfer_element!(Pₑ::AbstractMatrix, cell, element_cache::NodalPolynomialProlongationElementCache, p)
    (; cv, vdim) = element_cache

    if vdim > 1
        for i in 1:getnquadpoints(cv)
            for j in 1:getnbasefunctions(cv)
                val = shape_value(cv, i, j)
                for k in 1:vdim
                    Pₑ[vdim*(i-1)+k, j] += val[k]
                end
            end
        end
    else
        for i in 1:getnquadpoints(cv)
            for j in 1:getnbasefunctions(cv)
                Pₑ[i, j] += shape_value(cv, i, j)
            end
        end
    end
end

function FerriteOperators.setup_transfer_element_cache(element_model::PolynomialProlongationIntegrator, sdh_row::SubDofHandler, sdh_col::SubDofHandler)
    field_name = element_model.field_name
    ip1        = Ferrite.getfieldinterpolation(sdh_row, field_name)
    ip2        = Ferrite.getfieldinterpolation(sdh_col, field_name)
    ip_geo     = FerriteOperators.geometric_subdomain_interpolation(sdh_row)
    positions  = Ferrite.reference_coordinates(ip1)
    ref_shape  = Ferrite.getrefshape(ip1)
    qr         = QuadratureRule{ref_shape}([NaN for _ = 1:length(positions)], positions)
    return NodalPolynomialProlongationElementCache(CellValues(qr, ip2, ip_geo), Ferrite.n_components(ip2))
end

"""
    build_prolongator(fine_dh::DofHandler, coarse_dh::DofHandler)

Assemble the prolongation matrix P of size `(ndofs(fine_dh) × ndofs(coarse_dh))`.

Uses a `MassProlongatorIntegrator` via the FerriteOperators transfer operator infrastructure.
For each cell the local prolongator is computed via an element-local L²-projection
(fine mass matrix inverse times cross-mass matrix), then assembled into the global matrix.
Rows are normalised by the number of element contributions to handle shared dofs correctly.
"""
function build_prolongator(
        fine_dh::DofHandler,
        coarse_dh::DofHandler
    )
    field_name  = first(Ferrite.getfieldnames(fine_dh))
    # FIXME multi-field support
    @assert length(fine_dh.field_names) == 1 "Multiple fields not yet supported"
    qr_order    = 2 * getorder(fine_dh.subdofhandlers[1].field_interpolations[1])
    integrator  = PolynomialProlongationIntegrator(field_name)
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

