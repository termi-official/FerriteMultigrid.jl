# interface for multigrid problems
# there will be ready to use implementations for some common problems,
# however for new problems, this interface should be implemented

"""
    abstract type AbstractPMultigrid
This is an abstract type that can be extended to when `Rediscretization` strategy is used as coarsening strategy, otherwise it's not required.
"""
abstract type AbstractPMultigrid end

"""
    assemble(problem::AbstractPMultigrid, fe_space::FESpace)
The interface that needs to be implemented to define the `Rediscretization` strategy for a specific problem type.
"""
function assemble(problem::AbstractPMultigrid, ::FESpace)
    # this is an interface that should be defined for each specific problem type
    # This function should be implemented in the specific problem type
    error("assemble not implemented for $(typeof(problem))")
end

#######################
## Diffusion problem ##
#######################

#TODO: more coefficient types
abstract type AbstractCoefficient end
struct ConstantCoefficient{Tv <: Real} <: AbstractCoefficient
    K::Tv
end

function *(c::ConstantCoefficient, x::Real)
    return c.K * x
end

function *(x::Real, c::ConstantCoefficient)
    return x * c.K
end


"""
    DiffusionMultigrid{C} <: AbstractPMultigrid
This struct represents a multigrid problem for diffusion equations with a coefficient `C`.
"""
struct DiffusionMultigrid{C} <: AbstractPMultigrid
    coeff::C
end

function DiffusionMultigrid(coeff::Real)
    return DiffusionMultigrid(ConstantCoefficient(coeff))
end


function assemble(problem::DiffusionMultigrid, fe_space::FESpace)
    dh = fe_space.dh
    cv = fe_space.cv
    ch = fe_space.ch

    K = allocate_matrix(dh)
    _assemble_global!(K, dh, cv, problem)
    apply!(K, ch)

    return K
end


function _assemble_cell!(Ke, cellvalues,problem::DiffusionMultigrid)
    fill!(Ke, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q)
        for i in 1:n_basefuncs
            ∇δu = shape_gradient(cellvalues, q, i)
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q, j)
                Ke[i, j] += problem.coeff * (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
end

function _assemble_global!(K, dh, cellvalues, problem::DiffusionMultigrid)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        _assemble_cell!(Ke, cellvalues, problem)
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K
end

###############################
## Linear Elasticity problem ##
###############################

"""
    LinearElasticityMultigrid{TC} <: AbstractPMultigrid
This struct represents a multigrid problem for linear elasticity equations with a material stiffness tensor `TC
"""
struct LinearElasticityMultigrid{TC<: SymmetricTensor} <: AbstractPMultigrid
    ℂ:: TC # material stiffness tensor (4th order tensor)
end

function LinearElasticityMultigrid(dim::Int, E::Tv, ν::Tv) where {Tv <: Real}
    # dim has to be either 1, 2 or 3
    @assert 1 ≤ dim ≤ 3 "Invalid dimension $dim for linear elasticity problem"
    @assert E > 0 "Young's modulus E must be positive"
    @assert 0 ≤ ν < 0.5 "Poisson's ratio ν must be in the range [0, 0.5)" #TODO: revisit this range

    G = E / (2(1 + ν))  # Shear modulus
    K = E / (3(1 - 2ν)) # Bulk modulus
    ℂ = gradient(ϵ -> 2 * G * dev(ϵ) + 3 * K * vol(ϵ), zero(SymmetricTensor{2, dim}));
    return LinearElasticityMultigrid(ℂ)
end

function assemble(problem::LinearElasticityMultigrid, fe_space::FESpace)
    dh = fe_space.dh
    cv = fe_space.cv
    ch = fe_space.ch

    K = allocate_matrix(dh)
    _assemble_global!(K, dh, cv, problem)
    apply!(K, ch)

    return K
end

function _assemble_cell!(ke, cellvalues, problem::LinearElasticityMultigrid)
    ℂ = problem.ℂ
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the test function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ ℂ ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function _assemble_global!(K, dh, cellvalues, problem::LinearElasticityMultigrid)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cellvalues, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        # Compute element contribution
        _assemble_cell!(ke, cellvalues, problem)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

###############################
## Hyperelasticity problem ##
###############################

"""
    HyperelasticityMultigrid <: AbstractPMultigrid
This struct represents a multigrid problem for hyperelasticity equations with a material model.
"""
struct HyperelasticityMultigrid{TM} <: AbstractPMultigrid
    material::TM # material model (e.g., NeoHooke)
end

function assemble(problem::HyperelasticityMultigrid, fe_space::FESpace)
    dh = fe_space.dh
    cv = fe_space.cv
    ch = fe_space.ch

    K = allocate_matrix(dh)
    _assemble_global_linearized!(K, dh, cv, problem)
    apply!(K, ch)

    return K
end

function _assemble_cell_linearized!(ke, cellvalues, problem::HyperelasticityMultigrid, ue)
    fill!(ke, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)

    for qp in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)

        ## Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cellvalues, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F

        ## Compute stress and tangent using automatic differentiation
        ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, problem.material), C, :all)
        S = 2.0 * ∂Ψ∂C
        ∂S∂C = 2.0 * ∂²Ψ∂C²

        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)

        for i in 1:n_basefuncs
            ∇δui = shape_gradient(cellvalues, qp, i)
            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
            for j in 1:n_basefuncs
                ∇δuj = shape_gradient(cellvalues, qp, j)
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
    return ke
end

function _assemble_global_linearized!(K, dh, cellvalues, problem::HyperelasticityMultigrid)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)

    ## For linearization around zero displacement
    ue = zeros(n_basefuncs)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        _assemble_cell_linearized!(ke, cellvalues, problem, ue)
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

## Helper function for potential energy (needs to be defined in hyperelasticity context)
function Ψ(C, mp)
    ## This will be overloaded by specific material models
    error("Ψ not implemented for material type $(typeof(mp))")
end
