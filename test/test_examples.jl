@testset "Poisson Equation Example" begin
    configs = [
        pmultigrid_config(), 
        pmultigrid_config(coarse_strategy = Rediscretization(DiffusionMultigrid(1.0))),
        pmultigrid_config(proj_strategy = StepProjection(1)),
        pmultigrid_config(coarse_strategy = Rediscretization(DiffusionMultigrid(1.0)), proj_strategy = StepProjection(1)),
    ]
    ## 1D Poisson equation example ##
    for config in configs
        K, f, fe_space = poisson(1000, 2, 3)
        # 1. default configuration
        x, res = solve(K, f, fe_space,config; log=true, rtol = 1e-10)
        println("final residual at iteration ", length(res), ": ", res[end])
        @test K * x ≈ f
    end

    # 2D Poisson equation example ##
    for config in configs
        K, f, fe_space = poisson((100,100), 2, 3)
        # 1. default configuration
        x, res = solve(K, f, fe_space, config; log=true, rtol = 1e-10)
        println("final residual at iteration ", length(res), ": ", res[end])
        @test K * x ≈ f
    end
end

module TestLinearElasticityExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate-tutorials/linear_elasticity.jl"))
        end
    end
end

module TestHyperelasticityExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate-tutorials/hyperelasticity.jl"))
        end
    end
end
