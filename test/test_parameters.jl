import FerriteMultigrid: init, AMGCoarseSolver

@testset "Configuration Parameters" begin
    # config: Galerikn and DirectProjection
    config = pmultigrid_config()
    @test config.coarse_strategy isa Galerkin
    @test config.proj_strategy isa DirectProjection

    # config: Rediscretization and DirectProjection
    cofig = pmultigrid_config(coarse_strategy=Rediscretization(DiffusionMultigrid(1.0)))
    @test cofig.coarse_strategy isa Rediscretization
    @test cofig.proj_strategy isa DirectProjection

    # config: Galerikn and StepProjection
    config = pmultigrid_config(proj_strategy=StepProjection(1))
    @test config.coarse_strategy isa Galerkin
    @test config.proj_strategy isa StepProjection

    # config: Rediscretization and StepProjection
    config = pmultigrid_config(coarse_strategy=Rediscretization(DiffusionMultigrid(1.0)), proj_strategy=StepProjection(1))
    @test config.coarse_strategy isa Rediscretization
    @test config.proj_strategy isa StepProjection

end


@testset "MultiLevel Parameters" begin
    K, f, fe_space = poisson(3, 2, 3)
    ## SA-AMG as coarse solver
    pmgsolver = init(K, f, fe_space, pmultigrid_config(), pcoarse_solver=SmoothedAggregationCoarseSolver(), presmoother=GaussSeidel(; iter=4), postsmoother=GaussSeidel(; iter=2))
    ml = pmgsolver.ml
    @test ml.coarse_solver isa AMGCoarseSolver
    @test ml.levels |> length == 1
    @test ml.presmoother isa GaussSeidel
    @test ml.presmoother.iter == 4
    @test ml.postsmoother isa GaussSeidel
    @test ml.postsmoother.iter == 2

    ## RS-AMG as coarse solver
    pmgsolver = init(K, f, fe_space, pmultigrid_config(), pcoarse_solver=RugeStubenCoarseSolver(), presmoother=GaussSeidel(; iter=2), postsmoother=GaussSeidel(; iter=4))
    ml = pmgsolver.ml
    @test ml.coarse_solver isa AMGCoarseSolver
    @test ml.levels |> length == 1
    @test ml.presmoother isa GaussSeidel
    @test ml.presmoother.iter == 2
    @test ml.postsmoother isa GaussSeidel
    @test ml.postsmoother.iter == 4

    ## Direct Solver as coarse solver (e.g. Pinv)
    pmgsolver = init(K, f, fe_space, pmultigrid_config(), pcoarse_solver=Pinv)
    ml = pmgsolver.ml
    @test ml.coarse_solver isa Pinv

end
