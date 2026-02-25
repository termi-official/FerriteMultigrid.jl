@testset "Poisson Equation Example with $config" for config in [
    pmultigrid_config(), 
    pmultigrid_config(coarse_strategy = Rediscretization(DiffusionMultigrid(1.0))),
    pmultigrid_config(proj_strategy = StepProjection(1)),
    pmultigrid_config(coarse_strategy = Rediscretization(DiffusionMultigrid(1.0)), proj_strategy = StepProjection(1)),
]
    ## 1D Poisson equation example ##
    K, f, dh, ch = poisson(1000, 2, 3)
    x, res = solve(K, f, dh, ch, config; log=true, rtol = 1e-10)
    println("final residual at iteration ", length(res), ": ", res[end])
    @test K * x ≈ f

    K, f, dh, ch = poisson((100,100), 2, 3)
    x, res = solve(K, f, dh, ch, config; log=true, rtol = 1e-10)
    println("final residual at iteration ", length(res), ": ", res[end])
    @test K * x ≈ f
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
