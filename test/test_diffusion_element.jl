using FerriteOperators

#######################
## Diffusion problem ##
#######################

"""
    DiffusionIntegrator{C, QRC} <: AbstractBilinearIntegrator

Multigrid problem for a scalar diffusion equation −∇⋅(K ∇u) = f.
Implements the FerriteOperators `AbstractBilinearIntegrator` interface so that
`setup_operator` / `update_operator!` can be used for assembly.

# Fields
- `coeff::C`
- `qrc::QuadratureRuleCollection`
"""
struct DiffusionIntegrator{C, QRC <: QuadratureRuleCollection} <: FerriteOperators.AbstractBilinearIntegrator
    coeff::C
    qrc::QRC
end

DiffusionIntegrator(coeff, qr_order::Int) = DiffusionIntegrator(coeff, QuadratureRuleCollection(qr_order))

struct DiffusionElementCache{CV} <: FerriteOperators.AbstractVolumetricElementCache
    cv::CV
    K::Float64  # scalar diffusion coefficient
end

function FerriteOperators.setup_element_cache(integrator::DiffusionIntegrator, sdh::SubDofHandler)
    qr      = getquadraturerule(integrator.qrc, sdh)
    ip      = Ferrite.getfieldinterpolation(sdh, first(Ferrite.getfieldnames(sdh)))
    first_cell = getcells(Ferrite.get_grid(sdh.dh), first(sdh.cellset))
    ip_geo  = Ferrite.geometric_interpolation(typeof(first_cell))
    cv      = CellValues(qr, ip, ip_geo)
    return DiffusionElementCache(cv, Float64(integrator.coeff))
end

function FerriteOperators.assemble_element!(Ke::AbstractMatrix, cell::CellCache, cache::DiffusionElementCache, p)
    Ferrite.reinit!(cache.cv, cell)
    fill!(Ke, 0.0)
    K_coeff = cache.K
    for q in 1:getnquadpoints(cache.cv)
        dΩ = getdetJdV(cache.cv, q)
        for i in 1:getnbasefunctions(cache.cv)
            ∇δu = shape_gradient(cache.cv, q, i)
            for j in 1:getnbasefunctions(cache.cv)
                ∇u = shape_gradient(cache.cv, q, j)
                Ke[i, j] += K_coeff * (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke
end
