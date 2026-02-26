# interface for multigrid problems
# there will be ready to use implementations for some common problems,
# however for new problems, this interface should be implemented

"""
    abstract type AbstractPMultigrid
This is an abstract type that can be extended to when `Rediscretization` strategy is used as coarsening strategy, otherwise it's not required.
"""
abstract type AbstractPMultigrid end

#######################
## Diffusion problem ##
#######################

abstract type AbstractCoefficient end
struct ConstantCoefficient{Tv <: Real} <: AbstractCoefficient
    K::Tv
end

function *(c::ConstantCoefficient, x::Real); return c.K * x; end
function *(x::Real, c::ConstantCoefficient); return x * c.K; end

"""
    DiffusionMultigrid{C, QRC} <: AbstractBilinearIntegrator

Multigrid problem for a scalar diffusion equation −∇⋅(K ∇u) = f.
Implements the FerriteOperators `AbstractBilinearIntegrator` interface so that
`setup_operator` / `update_operator!` can be used for assembly.

# Fields
- `coeff::C` – diffusion coefficient (supports `ConstantCoefficient`)
- `qrc::QuadratureRuleCollection` – quadrature rule collection (default order 2)
"""
struct DiffusionMultigrid{C, QRC} <: AbstractBilinearIntegrator
    coeff::C
    qrc::QRC
end

DiffusionMultigrid(coeff::Real) = DiffusionMultigrid(ConstantCoefficient(coeff), QuadratureRuleCollection(2))
DiffusionMultigrid(coeff::Real, qr_order::Int) = DiffusionMultigrid(ConstantCoefficient(coeff), QuadratureRuleCollection(qr_order))
DiffusionMultigrid(coeff::AbstractCoefficient) = DiffusionMultigrid(coeff, QuadratureRuleCollection(2))

## Element cache ─────────────────────────────────────────────────────────────────────────

struct DiffusionElementCache{CV} <: AbstractVolumetricElementCache
    cv::CV
    K::Float64  # scalar diffusion coefficient
end

function setup_element_cache(problem::DiffusionMultigrid, sdh::SubDofHandler)
    qr      = getquadraturerule(problem.qrc, sdh)
    ip      = Ferrite.getfieldinterpolation(sdh, first(Ferrite.getfieldnames(sdh)))
    first_cell = getcells(get_grid(sdh.dh), first(sdh.cellset))
    ip_geo  = Ferrite.geometric_interpolation(typeof(first_cell))
    cv      = CellValues(qr, ip, ip_geo)
    return DiffusionElementCache(cv, Float64(problem.coeff.K))
end

function assemble_element!(Ke::AbstractMatrix, cell::CellCache, cache::DiffusionElementCache, p)
    reinit!(cache.cv, cell)
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

###############################
## Linear Elasticity problem ##
###############################

"""
    LinearElasticityMultigrid{TC, QRC} <: AbstractBilinearIntegrator

Multigrid problem for linear elasticity.
Implements the FerriteOperators `AbstractBilinearIntegrator` interface.
"""
struct LinearElasticityMultigrid{TC <: SymmetricTensor, QRC} <: AbstractBilinearIntegrator
    ℂ::TC  # material stiffness tensor (4th order)
    qrc::QRC
end

function LinearElasticityMultigrid(dim::Int, E::Tv, ν::Tv) where {Tv <: Real}
    @assert 1 ≤ dim ≤ 3 "Invalid dimension $dim for linear elasticity problem"
    @assert E > 0 "Young's modulus E must be positive"
    @assert 0 ≤ ν < 0.5 "Poisson's ratio ν must be in the range [0, 0.5)"
    G = E / (2(1 + ν))
    K = E / (3(1 - 2ν))
    ℂ = gradient(ϵ -> 2 * G * dev(ϵ) + 3 * K * vol(ϵ), zero(SymmetricTensor{2, dim}))
    return LinearElasticityMultigrid(ℂ, QuadratureRuleCollection(2))
end

LinearElasticityMultigrid(ℂ::SymmetricTensor) = LinearElasticityMultigrid(ℂ, QuadratureRuleCollection(2))

## Element cache ─────────────────────────────────────────────────────────────────────────

struct LinearElasticityElementCache{CV, TC} <: AbstractVolumetricElementCache
    cv::CV
    ℂ::TC
end

function setup_element_cache(problem::LinearElasticityMultigrid, sdh::SubDofHandler)
    qr     = getquadraturerule(problem.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, first(Ferrite.getfieldnames(sdh)))
    first_cell = getcells(get_grid(sdh.dh), first(sdh.cellset))
    ip_geo = Ferrite.geometric_interpolation(typeof(first_cell))
    cv     = CellValues(qr, ip, ip_geo)
    return LinearElasticityElementCache(cv, problem.ℂ)
end

function assemble_element!(Ke::AbstractMatrix, cell::CellCache, cache::LinearElasticityElementCache, p)
    reinit!(cache.cv, cell)
    fill!(Ke, 0.0)
    ℂ = cache.ℂ
    for q_point in 1:getnquadpoints(cache.cv)
        dΩ = getdetJdV(cache.cv, q_point)
        for i in 1:getnbasefunctions(cache.cv)
            ∇Nᵢ = shape_gradient(cache.cv, q_point, i)
            for j in 1:getnbasefunctions(cache.cv)
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cache.cv, q_point, j)
                Ke[i, j] += (∇Nᵢ ⊡ ℂ ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return Ke
end

