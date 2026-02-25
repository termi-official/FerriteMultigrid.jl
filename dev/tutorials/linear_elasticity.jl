using Ferrite, FerriteGmsh, FerriteMultigrid, AlgebraicMultigrid
using Downloads: download
using IterativeSolvers
using TimerOutputs

TimerOutputs.enable_debug_timings(AlgebraicMultigrid)
TimerOutputs.enable_debug_timings(FerriteMultigrid)

Emod = 200.0e3 # Young's modulus [MPa]
ν = 0.3        # Poisson's ratio [-]

Gmod = Emod / (2(1 + ν))  # Shear modulus
Kmod = Emod / (3(1 - 2ν)) # Bulk modulus

C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2,2}))

function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    # Create a temporary array for the facet's local contributions to the external force vector
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for facet in FacetIterator(dh, facetset)
        # Update the facetvalues to the correct facet number
        reinit!(facetvalues, facet)
        # Reset the temporary array for the next facet
        fill!(fe_ext, 0.0)
        # Access the cell's coordinates
        cell_coordinates = getcoordinates(facet)
        for qp in 1:getnquadpoints(facetvalues)
            # Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(facetvalues, qp, cell_coordinates)
            tₚ = prescribed_traction(x)
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                Nᵢ = shape_value(facetvalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(f_ext, celldofs(facet), fe_ext)
    end
    return f_ext
end

function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the test function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_global!(K, dh, cellvalues, C)
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
        assemble_cell!(ke, cellvalues, C)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

function linear_elasticity_2d(C)
    logo_mesh = "logo.geo"
    asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
    isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)

    grid = togrid(logo_mesh)
    addfacetset!(grid, "top", x -> x[2] ≈ 1.0) # facets for which x[2] ≈ 1.0 for all nodes
    addfacetset!(grid, "left", x -> abs(x[1]) < 1.0e-6)
    addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6)

    dim = 2
    order = 4
    ip = Lagrange{RefTriangle,order}()^dim # vector valued interpolation
    ip_coarse = Lagrange{RefTriangle,1}()^dim

    qr = QuadratureRule{RefTriangle}(8)
    qr_face = FacetQuadratureRule{RefTriangle}(6)

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    dh_coarse = DofHandler(grid)
    add!(dh_coarse, :u, ip_coarse)
    close!(dh_coarse)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
    close!(ch)

    traction(x) = Vec(0.0, 20.0e3 * x[1])

    A = allocate_matrix(dh)
    assemble_global!(A, dh, cellvalues, C)

    b = zeros(ndofs(dh))
    assemble_external_forces!(b, dh, getfacetset(grid, "top"), facetvalues, traction)
    apply!(A, b, ch)

    return A, b, dh, dh_coarse, cellvalues, ch
end

function create_nns(dh, fieldname = first(dh.field_names))
    @assert length(dh.field_names) == 1 "Only a single field is supported for now."

    coords_flat = zeros(ndofs(dh))
    apply_analytical!(coords_flat, dh, fieldname, x -> x)
    coords = reshape(coords_flat, (length(coords_flat) ÷ 2, 2))

    grid = dh.grid
    B = zeros(Float64, ndofs(dh), 3)
    B[1:2:end, 1] .= 1 # x - translation
    B[2:2:end, 2] .= 1 # y - translation

    # in-plane rotation (x,y) → (-y,x)
    x = coords[:, 1]
    y = coords[:, 2]
    B[1:2:end, 3] .= -y
    B[2:2:end, 3] .= x

    return B
end

using FerriteMultigrid

A, b, dh, dh_coarse, cellvalues, ch = linear_elasticity_2d(C);

B = create_nns(dh_coarse)

fe_space = FESpace(dh, cellvalues, ch)

reset_timer!()

pcoarse_solver = SmoothedAggregationCoarseSolver(; B)

@timeit "CG" x_cg = IterativeSolvers.cg(A, b; maxiter = 1000, verbose=false)

config_gal = pmultigrid_config(coarse_strategy = Galerkin())
@timeit "Galerkin only" x_gal, res_gal = FerriteMultigrid.solve(A, b,fe_space, config_gal; pcoarse_solver, verbose=false, log=true, rtol = 1e-10)

builder_gal = PMultigridPreconBuilder(fe_space, config_gal; pcoarse_solver)
@timeit "Build preconditioner" Pl_gal = builder_gal(A)[1]
@timeit "Galerkin CG" IterativeSolvers.cg(A, b; Pl = Pl_gal, maxiter = 1000, verbose=false)

# Rediscretization Coarsening Strategy
config_red = pmultigrid_config(coarse_strategy = Rediscretization(LinearElasticityMultigrid(C)))
@timeit "Rediscretization only" x_red, res_red = solve(A, b, fe_space, config_red; pcoarse_solver, log=true, rtol = 1e-10)

builder_red = PMultigridPreconBuilder(fe_space, config_red; pcoarse_solver)
@timeit "Build preconditioner" Pl_red = builder_red(A)[1]
@timeit "Rediscretization CG" IterativeSolvers.cg(A, b; Pl = Pl_red, maxiter = 1000, verbose=false)

print_timer(title = "Analysis with $(getncells(dh.grid)) elements", linechars = :ascii)

using Test
@testset "Linear Elasticity Example" begin
    println("Final residual with Galerkin coarsening: ", res_gal[end])
    @test A * x_gal ≈ b atol=1e-4
    println("Final residual with Rediscretization coarsening: ", res_red[end])
    @test A * x_red ≈ b atol=1e-4
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
