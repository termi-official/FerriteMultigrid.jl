"""
    FESpace{DH<:AbstractDofHandler, CV<:AbstractCellValues, CH<:ConstraintHandler}

A structure that encapsulates the finite element space.

# Fields
- `dh::DH`: [Degree-of-freedom handler](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/dofhandler/#Degrees-of-freedom) 
- `cv::CV`: [Cell values](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/fevalues/#Main-types)
- `ch::CH`: [Constraint handler](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/fevalues/#Main-types).
"""
struct FESpace{DH<:AbstractDofHandler,CV<:AbstractCellValues,CH<:ConstraintHandler}
    dh::DH
    cv::CV
    ch::CH
end

order(fe_space::FESpace) = fe_space.cv.fun_values.ip |> getorder
interpolation(fe_space::FESpace) = fe_space.cv.fun_values.ip
quadraturerule(fe_space::FESpace) = fe_space.cv.qr
Ferrite.ndofs(fe_space::FESpace) = ndofs(fe_space.dh)
Ferrite.getnbasefunctions(fe_space::FESpace) = getnbasefunctions(fe_space.cv)

function coarsen_order(fe_space::FESpace, p::Int)
    dh = fe_space.dh
    cv = fe_space.cv
    ch = fe_space.ch

    @assert 1 ≤ p < order(fe_space) "Invalid order $p for coarsening"

    # FIXME: more robust way to handle this?
    qr = fe_space |> quadraturerule
    ip = _new_coarse_ip(fe_space |> interpolation, p)
    coarse_cv = CellValues(qr, ip)
    coarse_dh = DofHandler(dh.grid)
    add!(coarse_dh, dh.field_names |> first, ip) # FIXME: better way to handle this?
    close!(coarse_dh)

    @assert all(ch.affine_inhomogeneities .== nothing) "Affine constraints are not supported"

    coarse_ch = ConstraintHandler(coarse_dh)
    for dbc in ch.dbcs
        add!(coarse_ch, dbc)
    end
    close!(coarse_ch)

    return FESpace(coarse_dh, coarse_cv,coarse_ch)
end

function _new_coarse_ip(ip::ScalarInterpolation, p::Int)
    T = typeof(ip)
    BasisFunction = T.name.wrapper  #TODO: all basis functions have the same construction structure?
    RefShape = T.parameters[1]
    return BasisFunction{RefShape,p}()
end

# TODO: more robust
function _new_coarse_ip(::VectorizedInterpolation{vdim, refshape, order, SI}, p::Int) where {vdim, refshape, order, SI <: ScalarInterpolation{refshape, order}}
    BasisFunction = SI.name.wrapper  #TODO: all basis functions have the same construction structure?
    return BasisFunction{refshape,p}()^vdim
end
