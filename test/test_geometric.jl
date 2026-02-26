## Tests for geometric multigrid: uniform_refinement, GridHierarchy, gmultigrid

using FerriteMultigrid, Ferrite, Test, SparseArrays
import LinearAlgebra: norm, det
import AlgebraicMultigrid as AMG

# ─────────────────────────────────────────────────────────────────────────────
# Helper: assemble 1D Poisson problem on an arbitrary grid
# ─────────────────────────────────────────────────────────────────────────────

function _make_1d_dh_ch(grid, order)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, order}())
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, union(getfacetset(grid, "left"), getfacetset(grid, "right")), (x, t) -> 0.0))
    close!(ch)
    return dh, ch
end

function _assemble_poisson(dh, ch, order)
    qr = QuadratureRule{RefLine}(order + 1)
    ip = Lagrange{RefLine, order}()
    cv = CellValues(qr, ip)
    n  = getnbasefunctions(cv)
    K  = allocate_matrix(dh)
    f  = zeros(ndofs(dh))
    asm = start_assemble(K, f)
    Ke  = zeros(n, n); fe = zeros(n)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0.0); fill!(fe, 0.0)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for i in 1:n
                fe[i] += shape_value(cv, q, i) * dΩ
                for j in 1:n
                    Ke[i,j] += shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j) * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cell), Ke, fe)
    end
    apply!(K, f, ch)
    return K, f
end

function _make_2d_quad_dh_ch(grid, order)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, order}())
    close!(dh)
    ch = ConstraintHandler(dh)
    ∂Ω = union(getfacetset(grid, "left"), getfacetset(grid, "right"),
               getfacetset(grid, "bottom"), getfacetset(grid, "top"))
    add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> 0.0))
    close!(ch)
    return dh, ch
end

function _assemble_poisson_2d(dh, ch, order)
    qr = QuadratureRule{RefQuadrilateral}(order + 1)
    ip = Lagrange{RefQuadrilateral, order}()
    cv = CellValues(qr, ip)
    n  = getnbasefunctions(cv)
    K  = allocate_matrix(dh)
    f  = zeros(ndofs(dh))
    asm = start_assemble(K, f)
    Ke  = zeros(n, n); fe = zeros(n)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0.0); fill!(fe, 0.0)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for i in 1:n
                fe[i] += shape_value(cv, q, i) * dΩ
                for j in 1:n
                    Ke[i,j] += shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j) * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cell), Ke, fe)
    end
    apply!(K, f, ch)
    return K, f
end


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

    dh_coarse, ch_coarse = _make_1d_dh_ch(gh.grids[1], 1)
    dh_fine,   ch_fine   = _make_1d_dh_ch(gh.grids[2], 1)
    dh_hierarchy = [dh_coarse, dh_fine]
    ch_hierarchy = [ch_coarse, ch_fine]

    K, f = _assemble_poisson(dh_fine, ch_fine, 1)
    config = gmultigrid_config()

    ml = gmultigrid(K, gh, dh_hierarchy, ch_hierarchy, config, SmoothedAggregationCoarseSolver())
    x, res = AMG._solve(ml, f; maxiter = 50, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end

@testset "gmultigrid – 1D Poisson, Rediscretization, 1 level" begin
    N = 20
    coarse_grid = generate_grid(Line, (N ÷ 2,))
    gh = GridHierarchy(coarse_grid, 1)

    dh_coarse, ch_coarse = _make_1d_dh_ch(gh.grids[1], 1)
    dh_fine,   ch_fine   = _make_1d_dh_ch(gh.grids[2], 1)
    dh_hierarchy = [dh_coarse, dh_fine]
    ch_hierarchy = [ch_coarse, ch_fine]

    K, f = _assemble_poisson(dh_fine, ch_fine, 1)
    config = gmultigrid_config(coarse_strategy = Rediscretization(DiffusionMultigrid(1.0)))

    ml = gmultigrid(K, gh, dh_hierarchy, ch_hierarchy, config, SmoothedAggregationCoarseSolver())
    x, res = AMG._solve(ml, f; maxiter = 50, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end

@testset "gmultigrid – 1D Poisson, Galerkin, 2 levels" begin
    N = 40
    coarse_grid = generate_grid(Line, (N ÷ 4,))
    gh = GridHierarchy(coarse_grid, 2)

    dh_hierarchy = map(g -> (dh = DofHandler(g); add!(dh, :u, Lagrange{RefLine,1}()); close!(dh); dh), gh.grids)
    ch_hierarchy = map(dh -> begin
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right")), (x,t)->0.0))
        close!(ch); ch
    end, dh_hierarchy)

    K, f = _assemble_poisson(dh_hierarchy[end], ch_hierarchy[end], 1)
    config = gmultigrid_config()

    ml = gmultigrid(K, gh, dh_hierarchy, ch_hierarchy, config, SmoothedAggregationCoarseSolver())
    x, res = AMG._solve(ml, f; maxiter = 100, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end

@testset "gmultigrid – 2D Quad Poisson, Galerkin, 1 level" begin
    coarse_grid = generate_grid(Quadrilateral, (8, 8))
    gh = GridHierarchy(coarse_grid, 1)

    dh_coarse, ch_coarse = _make_2d_quad_dh_ch(gh.grids[1], 1)
    dh_fine,   ch_fine   = _make_2d_quad_dh_ch(gh.grids[2], 1)
    dh_hierarchy = [dh_coarse, dh_fine]
    ch_hierarchy = [ch_coarse, ch_fine]

    K, f = _assemble_poisson_2d(dh_fine, ch_fine, 1)
    config = gmultigrid_config()

    ml = gmultigrid(K, gh, dh_hierarchy, ch_hierarchy, config, SmoothedAggregationCoarseSolver())
    x, res = AMG._solve(ml, f; maxiter = 100, reltol = 1e-10, log = true)
    @test K * x ≈ f atol=1e-6
end
