import FerriteMultigrid: init, AMGCoarseSolver

@testset "Configuration Parameters" begin
    # config: Galerkin (default)
    config = pmultigrid_config()
    @test config.coarse_strategy isa Galerkin

    # config: Rediscretization
    config = pmultigrid_config(coarse_strategy=Rediscretization(DiffusionMultigrid(1.0)))
    @test config.coarse_strategy isa Rediscretization
end


@testset "MultiLevel Parameters" begin
    K, f, dh, ch = poisson(3, 2, 3)
    ## SA-AMG as coarse solver
    pmgsolver = init(K, f, dh, ch, pmultigrid_config(), pcoarse_solver=SmoothedAggregationCoarseSolver(), presmoother=GaussSeidel(; iter=4), postsmoother=GaussSeidel(; iter=2))
    ml = pmgsolver.ml
    @test ml.coarse_solver isa AMGCoarseSolver
    @test ml.levels |> length == 1
    @test ml.presmoother isa GaussSeidel
    @test ml.presmoother.iter == 4
    @test ml.postsmoother isa GaussSeidel
    @test ml.postsmoother.iter == 2

    ## RS-AMG as coarse solver
    pmgsolver = init(K, f, dh, ch, pmultigrid_config(), pcoarse_solver=RugeStubenCoarseSolver(), presmoother=GaussSeidel(; iter=2), postsmoother=GaussSeidel(; iter=4))
    ml = pmgsolver.ml
    @test ml.coarse_solver isa AMGCoarseSolver
    @test ml.levels |> length == 1
    @test ml.presmoother isa GaussSeidel
    @test ml.presmoother.iter == 2
    @test ml.postsmoother isa GaussSeidel
    @test ml.postsmoother.iter == 4

    ## Direct Solver as coarse solver (e.g. Pinv)
    pmgsolver = init(K, f, dh, ch, pmultigrid_config(), pcoarse_solver=Pinv)
    ml = pmgsolver.ml
    @test ml.coarse_solver isa Pinv

end
