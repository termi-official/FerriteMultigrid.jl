## Tests for geometric multigrid: uniform_refinement, GridHierarchy, gmultigrid

using FerriteMultigrid, Ferrite, Test, SparseArrays
import LinearAlgebra: norm, det
import FerriteMultigrid: assemble_poisson

# ─────────────────────────────────────────────────────────────────────────────
# uniform_refinement tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "uniform_refinement – Line (1D)" begin
    coarse = generate_grid(Line, (4,))
    fine, f2c, crc = uniform_refinement(coarse)

    @test getncells(fine) == 2 * getncells(coarse)
    @test length(f2c)    == getncells(fine)
    @test length(crc)    == getncells(fine)
    @test all(1 .<= f2c .<= getncells(coarse))

    # Every fine cell has 2 reference coordinate nodes in [-1,1]
    for (k, coords) in enumerate(crc)
        @test length(coords) == 2
        for ξ in coords
            @test -1.0 - 1e-12 <= ξ[1] <= 1.0 + 1e-12
        end
    end

    # Boundary sets propagated
    @test haskey(Ferrite.getfacetsets(fine), "left")
    @test haskey(Ferrite.getfacetsets(fine), "right")
    @test !isempty(getfacetset(fine, "left"))
    @test !isempty(getfacetset(fine, "right"))
end

@testset "uniform_refinement – Triangle (2D)" begin
    coarse = generate_grid(Triangle, (3, 3))
    fine, f2c, crc = uniform_refinement(coarse)

    @test getncells(fine) == 4 * getncells(coarse)
    @test length(f2c) == getncells(fine)
    # Each triangle has 3 nodes
    for coords in crc
        @test length(coords) == 3
        for ξ in coords
            @test ξ[1] >= -1e-12
            @test ξ[2] >= -1e-12
            @test ξ[1] + ξ[2] <= 1.0 + 1e-12
        end
    end
    # Boundary sets
    @test haskey(Ferrite.getfacetsets(fine), "left")
end

@testset "uniform_refinement – Quadrilateral (2D)" begin
    coarse = generate_grid(Quadrilateral, (3, 3))
    fine, f2c, crc = uniform_refinement(coarse)

    @test getncells(fine) == 4 * getncells(coarse)
    @test length(f2c) == getncells(fine)
    # Boundary sets
    for name in ("left", "right", "bottom", "top")
        @test haskey(Ferrite.getfacetsets(fine), name)
        @test length(getfacetset(fine, name)) == 2 * length(getfacetset(coarse, name))
    end
end

@testset "uniform_refinement – Tetrahedron (3D)" begin
    coarse = generate_grid(Tetrahedron, (2, 2, 2))
    fine, f2c, crc = uniform_refinement(coarse)

    @test getncells(fine) == 8 * getncells(coarse)
    # Each tet has 4 nodes; all ref coords inside RefTetrahedron
    for coords in crc
        @test length(coords) == 4
        for ξ in coords
            @test ξ[1] >= -1e-12
            @test ξ[2] >= -1e-12
            @test ξ[3] >= -1e-12
            @test ξ[1] + ξ[2] + ξ[3] <= 1.0 + 1e-12
        end
    end
    @test haskey(Ferrite.getfacetsets(fine), "left")
end

@testset "uniform_refinement – Hexahedron (3D)" begin
    coarse = generate_grid(Hexahedron, (2, 2, 2))
    fine, f2c, crc = uniform_refinement(coarse)

    @test getncells(fine) == 8 * getncells(coarse)
    for coords in crc
        @test length(coords) == 8
        for ξ in coords
            @test all(-1.0 - 1e-12 .<= ξ .<= 1.0 + 1e-12)
        end
    end
    # Boundary sets: each coarse face → 4 fine faces
    for name in ("left", "right", "bottom", "top", "front", "back")
        @test haskey(Ferrite.getfacetsets(fine), name)
        @test length(getfacetset(fine, name)) == 4 * length(getfacetset(coarse, name))
    end
end

@testset "GridHierarchy construction" begin
    coarse = generate_grid(Line, (4,))
    gh = GridHierarchy(coarse, 2)
    @test length(gh) == 3
    @test getncells(gh.grids[1]) == 4
    @test getncells(gh.grids[2]) == 8
    @test getncells(gh.grids[3]) == 16
    @test length(gh.fine2coarse) == 2
    @test length(gh.child_ref_coords) == 2

    # 2D quad hierarchy
    coarse2d = generate_grid(Quadrilateral, (4, 4))
    gh2d = GridHierarchy(coarse2d, 2)
    @test length(gh2d) == 3
    @test getncells(gh2d.grids[2]) == 4 * getncells(coarse2d)
    @test getncells(gh2d.grids[3]) == 16 * getncells(coarse2d)
end


# ─────────────────────────────────────────────────────────────────────────────
# Prolongator tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "build_geometric_prolongator – 1D P1 (2 refinements)" begin
    coarse_grid = generate_grid(Line, (4,))
    gh = GridHierarchy(coarse_grid, 2)

    for k in 1:2
        dh_fine   = DofHandler(gh.grids[k+1]); add!(dh_fine,   :u, Lagrange{RefLine,1}()); close!(dh_fine)
        dh_coarse = DofHandler(gh.grids[k]);   add!(dh_coarse, :u, Lagrange{RefLine,1}()); close!(dh_coarse)

        P = build_geometric_prolongator(dh_fine, dh_coarse, gh.fine2coarse[k], gh.child_ref_coords[k])

        @test size(P, 1) == ndofs(dh_fine)
        @test size(P, 2) == ndofs(dh_coarse)
        # Partition-of-unity: each row of P sums to 1 (prolongation preserves constants)
        row_sums = vec(sum(P; dims=2))
        @test row_sums ≈ ones(ndofs(dh_fine)) atol=1e-10
    end
end

@testset "build_geometric_prolongator – 2D Quad P1" begin
    coarse_grid = generate_grid(Quadrilateral, (4, 4))
    fine_grid, f2c, crc = uniform_refinement(coarse_grid)

    dh_fine   = DofHandler(fine_grid);   add!(dh_fine,   :u, Lagrange{RefQuadrilateral,1}()); close!(dh_fine)
    dh_coarse = DofHandler(coarse_grid); add!(dh_coarse, :u, Lagrange{RefQuadrilateral,1}()); close!(dh_coarse)

    P = build_geometric_prolongator(dh_fine, dh_coarse, f2c, crc)
    @test size(P) == (ndofs(dh_fine), ndofs(dh_coarse))
    row_sums = vec(sum(P; dims=2))
    @test row_sums ≈ ones(ndofs(dh_fine)) atol=1e-10
end


# ─────────────────────────────────────────────────────────────────────────────
# gmultigrid solver tests (at most 2 refinement levels)
# ─────────────────────────────────────────────────────────────────────────────

@testset "gmultigrid – 1D Poisson, Galerkin, 1 level" begin
    N = 20
    coarse_grid = generate_grid(Line, (N ÷ 2,))
    gh = GridHierarchy(coarse_grid, 1)

    dhh = DofHandlerHierarchy(gh)
    add!(dhh, :u, Lagrange{RefLine, 1}())
    close!(dhh)

    chh = ConstraintHandlerHierarchy(dhh)
    add!(chh, dh -> Dirichlet(:u,
        union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right")), (x, t) -> 0.0))
    close!(chh)

    K, f = assemble_poisson(dhh[end], chh[end])
    config = gmultigrid_config()

    x, res = solve(K, f, gh, dhh, chh, config;
                   pcoarse_solver = SmoothedAggregationCoarseSolver(),
                   maxiter = 50, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end

@testset "gmultigrid – 1D Poisson, Rediscretization, 1 level" begin
    N = 20
    coarse_grid = generate_grid(Line, (N ÷ 2,))
    gh = GridHierarchy(coarse_grid, 1)

    dhh = DofHandlerHierarchy(gh)
    add!(dhh, :u, Lagrange{RefLine, 1}())
    close!(dhh)

    chh = ConstraintHandlerHierarchy(dhh)
    add!(chh, dh -> Dirichlet(:u,
        union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right")), (x, t) -> 0.0))
    close!(chh)

    K, f = assemble_poisson(dhh[end], chh[end])
    config = gmultigrid_config(coarse_strategy = Rediscretization(DiffusionMultigrid(1.0)))

    x, res = solve(K, f, gh, dhh, chh, config;
                   pcoarse_solver = SmoothedAggregationCoarseSolver(),
                   maxiter = 50, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end

@testset "gmultigrid – 1D Poisson, Galerkin, 2 levels" begin
    N = 40
    coarse_grid = generate_grid(Line, (N ÷ 4,))
    gh = GridHierarchy(coarse_grid, 2)

    dhh = DofHandlerHierarchy(gh)
    add!(dhh, :u, Lagrange{RefLine, 1}())
    close!(dhh)

    chh = ConstraintHandlerHierarchy(dhh)
    add!(chh, dh -> Dirichlet(:u,
        union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right")), (x, t) -> 0.0))
    close!(chh)

    K, f = assemble_poisson(dhh[end], chh[end])
    config = gmultigrid_config()

    x, res = solve(K, f, gh, dhh, chh, config;
                   pcoarse_solver = SmoothedAggregationCoarseSolver(),
                   maxiter = 100, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end

@testset "gmultigrid – 2D Quad Poisson, Galerkin, 3 level" begin
    coarse_grid = generate_grid(Quadrilateral, (4, 4))
    gh = GridHierarchy(coarse_grid, 3)

    dhh = DofHandlerHierarchy(gh)
    add!(dhh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dhh)

    chh = ConstraintHandlerHierarchy(dhh)
    add!(chh, dh -> begin
        ∂Ω = union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right"),
                   getfacetset(dh.grid, "bottom"), getfacetset(dh.grid, "top"))
        Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
    end)
    close!(chh)

    K, f = assemble_poisson(dhh[end], chh[end])
    config = gmultigrid_config()

    x, res = solve(K, f, gh, dhh, chh, config;
                   pcoarse_solver = SmoothedAggregationCoarseSolver(),
                   maxiter = 100, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: two-subdomain setup and assembly (SubDofHandler)
# ─────────────────────────────────────────────────────────────────────────────

## Split grid cells into "left_domain" (centroid x < split_x) and "right_domain"
## (centroid x ≥ split_x).  Both sets cover all cells exactly once.
function _add_subdomain_cellsets!(grid, split_x = 0.0)
    left = Set{Int}()
    right = Set{Int}()
    nodes = getnodes(grid)
    for (cid, cell) in enumerate(getcells(grid))
        cx = sum(get_node_coordinate(nodes[n])[1] for n in cell.nodes) / length(cell.nodes)
        cx < split_x ? push!(left, cid) : push!(right, cid)
    end
    addcellset!(grid, "left_domain",  left)
    addcellset!(grid, "right_domain", right)
end

# ─────────────────────────────────────────────────────────────────────────────
# SubDofHandler tests: polynomial multigrid on a two-subdomain problem
# ─────────────────────────────────────────────────────────────────────────────

@testset "pmultigrid – 2D two-subdomain Poisson via SubDofHandler, P3→P1" begin
    # Build a 4×4 grid split into left (x<0) / right (x≥0) subdomains
    grid = generate_grid(Quadrilateral, (4, 4))
    _add_subdomain_cellsets!(grid)

    dh = DofHandler(grid)
    sdh_l = SubDofHandler(dh, getcellset(grid, "left_domain"))
    add!(sdh_l, :u, Lagrange{RefQuadrilateral, 3}())
    sdh_r = SubDofHandler(dh, getcellset(grid, "right_domain"))
    add!(sdh_r, :u, Lagrange{RefQuadrilateral, 3}())
    close!(dh)

    ∂Ω = union(getfacetset(grid, "left"), getfacetset(grid, "right"),
               getfacetset(grid, "bottom"), getfacetset(grid, "top"))
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> 0.0))
    close!(ch)

    K, f = assemble_poisson(dh, ch)
    config = pmultigrid_config()

    x, res = solve(K, f, dh, ch, config;
                   pcoarse_solver = SmoothedAggregationCoarseSolver(),
                   maxiter = 100, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end

@testset "pmultigrid – 2D two-subdomain Poisson via DofHandlerHierarchy, P2→P1" begin
    # Same problem but using the hierarchy API explicitly
    grid = generate_grid(Quadrilateral, (4, 4))
    _add_subdomain_cellsets!(grid)

    dh = DofHandler(grid)
    sdh_l = SubDofHandler(dh, getcellset(grid, "left_domain"))
    add!(sdh_l, :u, Lagrange{RefQuadrilateral, 2}())
    sdh_r = SubDofHandler(dh, getcellset(grid, "right_domain"))
    add!(sdh_r, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh)

    ∂Ω = union(getfacetset(grid, "left"), getfacetset(grid, "right"),
               getfacetset(grid, "bottom"), getfacetset(grid, "top"))
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> 0.0))
    close!(ch)

    K, f = assemble_poisson(dh, ch)
    config = pmultigrid_config()

    dhh, chh = build_pmg_dofhandler_hierarchy(dh, ch, config)
    @test length(dhh) == 2  # P2 → P1: two levels
    @test length(dhh[1].subdofhandlers) == 2   # coarsest level preserves 2 subdomains
    @test length(dhh[end].subdofhandlers) == 2  # finest level preserves 2 subdomains

    x, res = solve(K, f, dhh, chh, config;
                   pcoarse_solver = SmoothedAggregationCoarseSolver(),
                   maxiter = 100, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end


# ─────────────────────────────────────────────────────────────────────────────
# SubDofHandler tests: geometric multigrid on a two-subdomain problem
# ─────────────────────────────────────────────────────────────────────────────

@testset "gmultigrid – 2D two-subdomain Poisson via SubDofHandlerHierarchy, Galerkin" begin
    coarse_grid = generate_grid(Quadrilateral, (4, 4))
    _add_subdomain_cellsets!(coarse_grid)

    gh = GridHierarchy(coarse_grid, 1)

    # Cellsets must be propagated to the fine grid
    @test haskey(Ferrite.getcellsets(gh.grids[2]), "left_domain")
    @test haskey(Ferrite.getcellsets(gh.grids[2]), "right_domain")
    @test length(getcellset(gh.grids[2], "left_domain"))  == 4 * length(getcellset(coarse_grid, "left_domain"))
    @test length(getcellset(gh.grids[2], "right_domain")) == 4 * length(getcellset(coarse_grid, "right_domain"))

    dhh = DofHandlerHierarchy(gh)
    sdhh_l = SubDofHandlerHierarchy(dhh, dh -> getcellset(dh.grid, "left_domain"))
    add!(sdhh_l, :u, Lagrange{RefQuadrilateral, 1}())
    sdhh_r = SubDofHandlerHierarchy(dhh, dh -> getcellset(dh.grid, "right_domain"))
    add!(sdhh_r, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dhh)

    chh = ConstraintHandlerHierarchy(dhh)
    add!(chh, dh -> begin
        ∂Ω = union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right"),
                   getfacetset(dh.grid, "bottom"), getfacetset(dh.grid, "top"))
        Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
    end)
    close!(chh)

    # Both levels must have 2 SubDofHandlers
    @test length(dhh[1].subdofhandlers) == 2
    @test length(dhh[2].subdofhandlers) == 2

    K, f = assemble_poisson(dhh[end], chh[end])
    config = gmultigrid_config()

    x, res = solve(K, f, gh, dhh, chh, config;
                   pcoarse_solver = SmoothedAggregationCoarseSolver(),
                   maxiter = 100, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end
