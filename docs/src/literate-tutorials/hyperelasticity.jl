# # [Hyperelasticity](@id tutorial-hyperelasticity)
#
# **Keywords**: *hyperelasticity*, *finite strain*, *large deformations*, *Newton's method*,
# *multigrid*, *automatic differentiation*, *p-multigrid*
#
# ![hyperelasticity.png](hyperelasticity.png)
#
# *Figure 1*: Cube loaded in torsion modeled with a hyperelastic material model and
# finite strain.
#
#md # !!! note
#md #     This example demonstrates how to solve a hyperelastic problem using FerriteMultigrid with second-order elements and compares different solver strategies.
#
# ## Introduction
#
# In this example we will solve a problem in a finite strain setting using a
# hyperelastic material model. We will use second-order Lagrange elements and solve
# the nonlinear system using Newton's method. For the linear system within each Newton
# iteration, we will compare three different approaches:
#
# 1. **P-Multigrid** (Galerkin coarsening strategy)
# 2. **Algebraic Multigrid (AMG)** (vanilla smoothed aggregation)
# 3. **Conjugate Gradient (CG)** (iterative solver)
#
# The weak form is expressed in terms of the first Piola-Kirchoff stress ``\mathbf{P}``
# as follows: Find ``\mathbf{u} \in \mathbb{U}`` such that
#
# ```math
# \int_{\Omega} [\nabla_{\mathbf{X}} \delta \mathbf{u}] : \mathbf{P}(\mathbf{u})\ \mathrm{d}\Omega =
# \int_{\Omega} \delta \mathbf{u} \cdot \mathbf{b}\ \mathrm{d}\Omega + \int_{\Gamma_\mathrm{N}}
# \delta \mathbf{u} \cdot \mathbf{t}\ \mathrm{d}\Gamma
# \quad \forall \delta \mathbf{u} \in \mathbb{U}^0,
# ```
#
# where ``\mathbf{u}`` is the unknown displacement field, ``\mathbf{b}`` is the body force,
# and ``\mathbf{t}`` is the traction on the Neumann boundary.

using Ferrite, Tensors, SparseArrays
using LinearAlgebra, IterativeSolvers
using Statistics, Printf

# ## Hyperelastic material model
#
# We use the compressible neo-Hookean model with the potential
#
# ```math
# \Psi(\mathbf{C}) = \frac{\mu}{2} (I_1 - 3) - {\mu} \ln(J) + \frac{\lambda}{2} (J - 1)^2,
# ```
#
# where ``I_1 = \mathrm{tr}(\mathbf{C})`` is the first invariant, ``J = \sqrt{\det(\mathbf{C})}``,
# and ``\mu`` and ``\lambda`` are material parameters.

struct NeoHooke
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

function constitutive_driver(C, mp::NeoHooke)
    ## Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

# ## Finite element assembly
#
# The element routine for assembling the residual and tangent stiffness.

function assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
    ## Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    b = Vec{3}((0.0, -0.5, 0.0)) # Body force
    tn = 0.1 # Traction
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ## Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F
        ## Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)

        ## Loop over test functions
        for i in 1:ndofs
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            ## Add contribution to the residual
            ge[i] += (∇δui ⊡ P - δui ⋅ b) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                ## Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end

    ## Surface integral for the traction
    for facet in 1:nfacets(cell)
        if (cellid(cell), facet) in ΓN
            reinit!(fv, cell, facet)
            for q_point in 1:getnquadpoints(fv)
                t = tn * getnormal(fv, q_point)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ t) * dΓ
                end
            end
        end
    end
    return
end;

function assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    assembler = start_assemble(K, g)

    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs]
        assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
        assemble!(assembler, global_dofs, ke, ge)
    end
    return
end;

# ## Problem setup
#
# Create the hyperelasticity problem with second-order elements.

function setup_hyperelasticity(; order=2, N=6)
    ## Generate a grid
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Tetrahedron, (N, N, N), left, right)

    ## Material parameters
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    ## Finite element base (second-order elements)
    ip = Lagrange{RefTetrahedron, order}()^3
    qr = QuadratureRule{RefTetrahedron}(order + 1)
    qr_facet = FacetQuadratureRule{RefTetrahedron}(order)
    cv = CellValues(qr, ip)
    fv = FacetValues(qr_facet, ip)

    ## DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ## Boundary conditions
    function rotation(X, t)
        θ = pi / 3 # 60°
        x, y, z = X
        return t * Vec{3}(
            (
                0.0,
                L / 2 - y + (y - L / 2) * cos(θ) - (z - L / 2) * sin(θ),
                L / 2 - z + (y - L / 2) * sin(θ) + (z - L / 2) * cos(θ),
            )
        )
    end

    dbcs = ConstraintHandler(dh)
    dbc = Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> rotation(x, t), [1, 2, 3])
    add!(dbcs, dbc)
    close!(dbcs)
    t = 0.5
    Ferrite.update!(dbcs, t)

    ## Neumann boundary
    ΓN = union(
        getfacetset(grid, "top"),
        getfacetset(grid, "bottom"),
        getfacetset(grid, "front"),
        getfacetset(grid, "back"),
    )

    return dh, cv, fv, mp, dbcs, ΓN
end;

# ### Near Null Space (NNS) for 3D
#
# For 3D hyperelasticity problems, the rigid body modes are:
# 1. Translation in x, y, z directions (3 modes)
# 2. Rotation about x, y, z axes (3 modes)
#
# The function `create_nns_3d` constructs the NNS matrix `B ∈ ℝ^{n × 6}` for the case of p=1
# (linear interpolation), as `B` is only relevant for AMG.

function create_nns_3d(dh)
    grid = dh.grid
    Ndof = 3 * (grid.nodes |> length) # nns at p = 1 for AMG
    B = zeros(Float64, Ndof, 6)

    ## Translations
    B[1:3:end, 1] .= 1.0  # x-translation
    B[2:3:end, 2] .= 1.0  # y-translation
    B[3:3:end, 3] .= 1.0  # z-translation

    ## Rotations: For rotation about axis i, the displacement is r × e_i
    coords = reduce(hcat, grid.nodes .|> (n -> n.x |> collect))' # Nx3 array
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]

    ## Rotation about x-axis: (x,y,z) → (x, -z, y)
    B[1:3:end, 4] .= 0.0
    B[2:3:end, 4] .= -z
    B[3:3:end, 4] .= y

    ## Rotation about y-axis: (x,y,z) → (z, y, -x)
    B[1:3:end, 5] .= z
    B[2:3:end, 5] .= 0.0
    B[3:3:end, 5] .= -x

    ## Rotation about z-axis: (x,y,z) → (-y, x, z)
    B[1:3:end, 6] .= -y
    B[2:3:end, 6] .= x
    B[3:3:end, 6] .= 0.0

    return B
end;

# ## Solver comparison
#
# We implement Newton's method with different linear solvers for comparison.

struct SolverStats
    method::String
    newton_iters::Int
    linear_iters::Vector{Int}
    linear_times::Vector{Float64}
    residuals::Vector{Float64}
end

function solve_nonlinear(solver_func, dh, cv, fv, mp, dbcs, ΓN; max_iter=30, tol=1e-8, verbose=true)
    _ndofs = ndofs(dh)
    un = zeros(_ndofs)
    u = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)
    apply!(un, dbcs)

    K = allocate_matrix(dh)
    g = zeros(_ndofs)

    iterations = Int[]
    residuals = Float64[]
    linear_solve_times = Float64[]

    for newton_itr in 1:max_iter
        u .= un .+ Δu
        assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
        apply_zero!(K, g, dbcs)

        normg = norm(g)
        push!(residuals, normg)

        if verbose
            @printf("Newton iter %2d: residual = %.3e\n", newton_itr, normg)
        end

        if normg < tol
            if verbose
                println("Converged in $newton_itr Newton iterations")
            end
            return un .+ Δu, SolverStats("", newton_itr, iterations, linear_solve_times, residuals)
        end

        ## Solve using provided solver
        fill!(ΔΔu, 0.0)
        t_start = time()
        n_iters = solver_func(ΔΔu, K, g)
        t_elapsed = time() - t_start

        push!(linear_solve_times, t_elapsed)
        push!(iterations, n_iters)

        apply_zero!(ΔΔu, dbcs)
        Δu .-= ΔΔu
    end

    error("Newton did not converge in $max_iter iterations")
end;

# ## Setup and solve with different methods

println("Setting up hyperelasticity problem...")
dh, cv, fv, mp, dbcs, ΓN = setup_hyperelasticity(order=2, N=6);
println("DOFs: ", ndofs(dh))
println("Cells: ", getncells(dh.grid))

## Create near null space
B = create_nns_3d(dh)

## Load FerriteMultigrid
using FerriteMultigrid

## Create FE space for p-multigrid
fe_space = FESpace(dh, cv, dbcs)

## 1. P-Multigrid with Galerkin coarsening
println("\n" * "="^60)
println("Solving with P-Multigrid (Galerkin)")
println("="^60)
config_gal = pmultigrid_config(coarse_strategy = Galerkin())

solver_pmg = function(x, K, b)
    x_sol, lin_res = solve(K, b, fe_space, config_gal; B=B, log=true, rtol=1e-6, maxiter=100)
    x .= x_sol
    return length(lin_res)
end

u_pmg, stats_pmg = solve_nonlinear(solver_pmg, dh, cv, fv, mp, dbcs, ΓN)
stats_pmg = SolverStats("P-MG (Galerkin)", stats_pmg.newton_iters, stats_pmg.linear_iters,
                         stats_pmg.linear_times, stats_pmg.residuals)

## 2. Vanilla AMG using AlgebraicMultigrid (smoothed aggregation)
println("\n" * "="^60)
println("Solving with Algebraic Multigrid (AMG)")
println("="^60)

solver_amg = function(x, K, b)
    # Use smoothed aggregation AMG with near null space
    ml_amg = smoothed_aggregation(K, B)

    # Use iterative solving with AMG as preconditioner
    ch = IterativeSolvers.cg!(x, K, b, Pl=ml_amg, maxiter=1000, log=true, reltol=1e-6)

    return ch[2].iters
end

u_amg, stats_amg = solve_nonlinear(solver_amg, dh, cv, fv, mp, dbcs, ΓN)
stats_amg = SolverStats("AMG", stats_amg.newton_iters, stats_amg.linear_iters,
                         stats_amg.linear_times, stats_amg.residuals)

## 3. Conjugate Gradient
println("\n" * "="^60)
println("Solving with Conjugate Gradient (CG)")
println("="^60)

solver_cg = function(x, K, b)
    cg_log = IterativeSolvers.cg!(x, K, b; maxiter=1000, log=true, reltol=1e-6)
    return cg_log[2].iters
end

u_cg, stats_cg = solve_nonlinear(solver_cg, dh, cv, fv, mp, dbcs, ΓN)
stats_cg = SolverStats("CG", stats_cg.newton_iters, stats_cg.linear_iters,
                        stats_cg.linear_times, stats_cg.residuals)

# ## Performance comparison and plotting

using Plots

## Plot 1: Newton convergence history
p1 = plot(
    title="Newton Convergence",
    xlabel="Newton Iteration",
    ylabel="Residual Norm",
    yscale=:log10,
    legend=:topright,
    grid=true,
    minorgrid=true
)
plot!(p1, 1:length(stats_pmg.residuals), stats_pmg.residuals,
      label="P-MG (Galerkin)", marker=:circle, lw=2, ms=4)
plot!(p1, 1:length(stats_amg.residuals), stats_amg.residuals,
      label="AMG", marker=:diamond, lw=2, ms=4)
plot!(p1, 1:length(stats_cg.residuals), stats_cg.residuals,
      label="CG", marker=:cross, lw=2, ms=5)

## Plot 2: Average linear solver iterations per Newton step
p2 = bar(
    ["P-MG\n(Galerkin)", "AMG", "CG"],
    [mean(stats_pmg.linear_iters), mean(stats_amg.linear_iters), mean(stats_cg.linear_iters)],
    title="Avg. Linear Solver Iterations",
    ylabel="Iterations",
    legend=false,
    color=[:blue, :red, :orange],
    bar_width=0.6
)

## Plot 3: Total linear solver time
p3 = bar(
    ["P-MG\n(Galerkin)", "AMG", "CG"],
    [sum(stats_pmg.linear_times), sum(stats_amg.linear_times), sum(stats_cg.linear_times)],
    title="Total Linear Solver Time",
    ylabel="Time (s)",
    legend=false,
    color=[:blue, :red, :orange],
    bar_width=0.6
)

## Combine plots
plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
savefig("hyperelasticity_comparison.png")

# ## Summary statistics

println("\n" * "="^70)
println("SOLVER PERFORMANCE SUMMARY")
println("="^70)
println("Method                | Newton Its | Avg Lin Its | Total Time (s)")
println("-"^70)
@printf("%-20s | %10d | %11.1f | %14.3f\n",
    stats_pmg.method, stats_pmg.newton_iters, mean(stats_pmg.linear_iters), sum(stats_pmg.linear_times))
@printf("%-20s | %10d | %11.1f | %14.3f\n",
    stats_amg.method, stats_amg.newton_iters, mean(stats_amg.linear_iters), sum(stats_amg.linear_times))
@printf("%-20s | %10d | %11.1f | %14.3f\n",
    stats_cg.method, stats_cg.newton_iters, mean(stats_cg.linear_iters), sum(stats_cg.linear_times))
println("="^70)

# ## Test the solutions

using Test
@testset "Hyperelasticity Solver Comparison" begin
    ## Assemble final system to check residuals
    K_final = allocate_matrix(dh)
    g_final = zeros(ndofs(dh))

    ## Test P-MG
    assemble_global!(K_final, g_final, dh, cv, fv, mp, u_pmg, ΓN)
    apply_zero!(K_final, g_final, dbcs)
    res_pmg = norm(g_final)
    println("\nFinal residual (P-MG): ", res_pmg)
    @test res_pmg < 1e-7

    ## Test AMG
    fill!(g_final, 0.0)
    assemble_global!(K_final, g_final, dh, cv, fv, mp, u_amg, ΓN)
    apply_zero!(K_final, g_final, dbcs)
    res_amg = norm(g_final)
    println("Final residual (AMG): ", res_amg)
    @test res_amg < 1e-7

    ## Test CG
    fill!(g_final, 0.0)
    assemble_global!(K_final, g_final, dh, cv, fv, mp, u_cg, ΓN)
    apply_zero!(K_final, g_final, dbcs)
    res_cg = norm(g_final)
    println("Final residual (CG): ", res_cg)
    @test res_cg < 1e-7

    ## All solutions should be similar
    @test u_pmg ≈ u_amg atol=1e-4
    @test u_pmg ≈ u_cg atol=1e-4
end

#md # ## [Plain program](@id hyperelasticity-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`hyperelasticity.jl`](hyperelasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
