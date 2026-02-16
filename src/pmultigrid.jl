## defines how we obtain A operator for the coarse grid
abstract type AbstractCoarseningStrategy end

@doc raw"""
    Galerkin <: AbstractCoarseningStrategy
Galerkin coarsening operator can be defined as follows:
```math
A_{h,p-1} = \mathcal{I}_{p}^{p-1} A_{h,p} \mathcal{I}_{p-1}^p
```
and according to [tielen2020](@citet) $\mathcal{I}_{p-1}^p$ is the interpolation operator from the coarse space to the fine space
and is defined as follows:

```math
\mathcal{I}_{p-1}^p (\mathbf{v}_{p-1}) = (\mathbf{M}_p)^{-1} \mathbf{P}_{p-1}^p \, \mathbf{v}_{p-1}
```
"""
struct Galerkin <: AbstractCoarseningStrategy
    is_sym::Bool
end
Galerkin() = Galerkin(true)

"""
    Rediscretization{TP <: AbstractPMultigrid} <: AbstractCoarseningStrategy
This struct represents a coarsening strategy that uses the `assemble` function to obtain the coarse grid operator.
It is used when the `Rediscretization` strategy is specified in the `pmultigrid_config`. It requires the problem type `TP` to be a subtype of [`AbstractPMultigrid`](@ref).
"""
struct Rediscretization{TP <: AbstractPMultigrid} <: AbstractCoarseningStrategy
    problem::TP
    is_sym::Bool
end
Rediscretization(problem) = Rediscretization(problem, true)

## defines how we project from fine to coarse grid
abstract type AbstractProjectionStrategy end

@doc raw"""
    DirectProjection <: AbstractProjectionStrategy
This struct represents a direct projection from $\mathcal{V}_{h,p}$ to $\mathcal{V}_{h,p=1}$. 
"""
struct DirectProjection <: AbstractProjectionStrategy end
    
@doc raw"""
    StepProjection <: AbstractProjectionStrategy
This struct represents a projection from $\mathcal{V}_{h,p}$ to $\mathcal{V}_{h,p-step}$, where `step` is a positive integer.
It is used to reduce the polynomial order by a fixed step size until `p = 1`.
"""    
struct StepProjection <: AbstractProjectionStrategy 
    step::Int
    function StepProjection(step::Int)
        step < 1 && error("Step must be greater than or equal to 1")
        return new(step)
    end
end

"""
    PMultigridConfiguration{TC<:AbstractCoarseningStrategy, TP<:AbstractProjectionStrategy}
This struct represents the configuration for the polynomial multigrid method.
"""
struct PMultigridConfiguration{TC<:AbstractCoarseningStrategy, TP<:AbstractProjectionStrategy}
    coarse_strategy::TC # coarsening strategy
    proj_strategy::TP # projection strategy
end


"""
    pmultigrid_config(;coarse_strategy = Galerkin(), proj_strategy = DirectProjection())
This function is the main api to instantiate [`PMultigridConfiguration`](@ref).
"""
pmultigrid_config(;coarse_strategy = Galerkin(), proj_strategy = DirectProjection()) = PMultigridConfiguration(coarse_strategy, proj_strategy)

function pmultigrid(
    A::TA,
    fe_space::FESpace,
    pgrid_config::PMultigridConfiguration,
    pcoarse_solver, 
    u = nothing,
    p = nothing,
    ::Type{Val{bs}} = Val{1};
    presmoother = GaussSeidel(),
    postsmoother = GaussSeidel(),
    kwargs...) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    levels = Vector{Level{TA,TA,Adjoint{T,TA}}}()
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    degree = fe_space |> order
    fespaces = Vector{FESpace}()
    push!(fespaces, fe_space)

    ps = pgrid_config.proj_strategy
    cs = pgrid_config.coarse_strategy
    step  = _calculate_step(ps, degree)

    while degree > 1
        # reduce the polynomial order
        degree = degree - step > 1 ? degree - step : 1

        fine_fespace = fespaces[end]
        coarse_fespace = coarsen_order(fine_fespace, degree)
        push!(fespaces, coarse_fespace)

        A = extend_hierarchy!(levels, fine_fespace, coarse_fespace, A, cs, u, p)

        coarse_x!(w, size(A, 1))
        coarse_b!(w, size(A, 1))
        residual!(w, size(A, 1))
    end
    coarse_solver = @timeit_debug "coarse solver setup" pcoarse_solver(A)
    return MultiLevel(levels, A, coarse_solver, presmoother, postsmoother, w)
end

function extend_hierarchy!(levels, fine_fespace::FESpace, coarse_fespace::FESpace, A, cs::Galerkin, u, p)
    P = @timeit_debug "build prolongator" build_prolongator(fine_fespace, coarse_fespace)
    R = @timeit_debug "build restriction" build_restriction(coarse_fespace, fine_fespace, P, cs.is_sym)
    push!(levels, Level(A, P, R))
    ## Galerkin projection: A_coarse = R * A * P
    RA = @timeit_debug "R * A" R * A
    A = @timeit_debug "RA * P" RA * P
    return A
end

function extend_hierarchy!(levels, fine_fespace::FESpace, coarse_fespace::FESpace, A, cs::Rediscretization, u, p)
    P = build_prolongator(fine_fespace, coarse_fespace)
    R = build_restriction(coarse_fespace, fine_fespace, P, cs.is_sym)
    push!(levels, Level(A, P, R))

    problem = cs.problem
    # TODO use FerriteOperators
    A = assemble(problem, coarse_fespace, u, p)
    return A
end

function _calculate_step(ps::StepProjection, p::Int) 
    step = ps.step
    step ≥ p && error("Step must be less than the polynomial order $p")
    return step
end

_calculate_step(::DirectProjection, fine_p::Int) = fine_p - 1

