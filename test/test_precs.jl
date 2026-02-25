using FerriteMultigrid, LinearSolve, Test

@testset "LinearSolvePrecs" begin
    # TODO: more tests
    A, b, fe_space = poisson(3, 2, 3)

    prob = LinearProblem(A, b)
    prec_builder = PMultigridPreconBuilder(fe_space, pmultigrid_config(); coarse_solver = Pinv)
    strategy = KrylovJL_CG(precs=prec_builder)
    @test A*LinearSolve.solve(prob, strategy, atol=1.0e-14) ≈ b rtol = 1.0e-8

end
