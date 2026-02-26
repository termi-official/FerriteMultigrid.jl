## Helpers to query DofHandler metadata ──────────────────────────────────────────────────

"""
    _first_field_ip(dh)

Return the interpolation of the first field in the first subdomain of `dh`.
This is a convenience function used when a single-field DofHandler is expected
(e.g. in the multigrid hierarchy builders).
"""
function _first_field_ip(dh::AbstractDofHandler)
    sdh = dh.subdofhandlers[1]
    return Ferrite.getfieldinterpolation(sdh, first(Ferrite.getfieldnames(sdh)))
end

"""
    order(dh::AbstractDofHandler)

Return the polynomial order of the interpolation of the first field in `dh`.
"""
order(dh::AbstractDofHandler) = getorder(_first_field_ip(dh))

"""
    interpolation(dh::AbstractDofHandler)

Return the interpolation of the first field in `dh`.
"""
interpolation(dh::AbstractDofHandler) = _first_field_ip(dh)

## Coarsening ─────────────────────────────────────────────────────────────────────────────

"""
    coarsen_order(dh::DofHandler, ch::ConstraintHandler, p::Int)
        -> (coarse_dh::DofHandler, coarse_ch::ConstraintHandler)

Create a coarser DofHandler / ConstraintHandler pair by replacing the polynomial
interpolation of every field with a version of order `p`.

The returned DofHandler lives on the **same** grid object as `dh`.
Only Dirichlet boundary conditions are transferred; affine inhomogeneities are not supported.
"""
function coarsen_order(dh::DofHandler, ch::ConstraintHandler, p::Int)
    @assert 1 ≤ p < order(dh) "Invalid coarsening order $p (current order is $(order(dh)))"
    @assert all(x -> x === nothing, ch.affine_inhomogeneities) "Affine constraints are not supported"

    coarse_dh = DofHandler(dh.grid)
    for sdh in dh.subdofhandlers
        coarse_sdh = SubDofHandler(coarse_dh, sdh.cellset)
        for fieldname in Ferrite.getfieldnames(sdh)
            add!(coarse_sdh, fieldname, _new_coarse_ip(Ferrite.getfieldinterpolation(sdh, fieldname), p))
        end
    end
    close!(coarse_dh)

    coarse_ch = ConstraintHandler(coarse_dh)
    for dbc in ch.dbcs
        add!(coarse_ch, dbc)
    end
    close!(coarse_ch)

    return coarse_dh, coarse_ch
end

function _new_coarse_ip(ip::ScalarInterpolation, p::Int)
    T = typeof(ip)
    BasisFunction = T.name.wrapper
    RefShape = T.parameters[1]
    return BasisFunction{RefShape, p}()
end

function _new_coarse_ip(::VectorizedInterpolation{vdim, refshape, order, SI}, p::Int) where {vdim, refshape, order, SI <: ScalarInterpolation{refshape, order}}
    BasisFunction = SI.name.wrapper
    return BasisFunction{refshape, p}()^vdim
end

