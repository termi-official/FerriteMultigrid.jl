using FerriteMultigrid, LinearSolve, Test
import Ferrite: generate_grid, Line, Lagrange, RefLine, DofHandler, add!, close!,
                ConstraintHandler, Dirichlet, getfacetset
import IterativeSolvers
import LinearAlgebra: norm

@testset "LinearSolvePrecs" begin
    A, b, dhh, chh = poisson(3, [1, 2], 3)

    prob = LinearProblem(A, b)
    prec_builder = PMultigridPreconBuilder(dhh, chh, pmultigrid_config(), pcoarse_solver = Pinv)
    strategy = KrylovJL_CG(precs=prec_builder)
    @test A*LinearSolve.solve(prob, strategy, atol=1.0e-14) ≈ b rtol = 1.0e-8
end

@testset "Combined PMG + GMG + AMG preconditioner with IterativeSolvers.cg" begin
    # Problem: P2 Poisson on a fine 1D grid
    # Preconditioner chain: PMG (P2 → P1, same fine grid)
    #                     → GMG (P1 fine → P1 coarse, 1 uniform refinement)
    #                     → AMG coarse solver
    N_coarse = 125
    coarse_grid = generate_grid(Line, (N_coarse,))
    fine_grid, _, _ = uniform_refinement(coarse_grid)
    gh = GridHierarchy(coarse_grid, 1)   # 1 refinement: coarse → fine

    # ── P2 problem on fine grid ───────────────────────────────────────────────
    A, b, dhh, chh = poisson((2 * N_coarse,), [1, 2], 3, Line, RefLine,
                                   g -> union(getfacetset(g, "left"),
                                              getfacetset(g, "right")))

    # ── P1 hierarchy for GMG (coarse=index 1, fine=index 2) ──────────────────
    dhh_p1 = DofHandlerHierarchy(gh)
    add!(dhh_p1, :u, Lagrange{RefLine, 1}())
    close!(dhh_p1)

    chh_p1 = ConstraintHandlerHierarchy(dhh_p1)
    add!(chh_p1, dh -> Dirichlet(:u,
        union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right")),
        (x, t) -> 0.0))
    close!(chh_p1)

    # ── Build combined preconditioner: PMG → GMG → AMG ───────────────────────
    gmg_builder = GMultigridCoarseSolverBuilder(gh, dhh_p1, chh_p1)
    ml = pmultigrid(A, dhh, chh, pmultigrid_config(), gmg_builder)
    prec = aspreconditioner(ml)

    # ── Solve with IterativeSolvers.cg ────────────────────────────────────────
    x = zeros(size(A, 1))
    IterativeSolvers.cg!(x, A, b; Pl = prec, maxiter = 100, reltol = 1e-8)
    @test norm(A * x - b) / norm(b) < 1e-6
end
