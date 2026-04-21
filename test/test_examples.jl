using FerriteMultigrid, Test

@testset "Poisson Equation Example with $config" for config in [
    pmultigrid_config(), 
    pmultigrid_config(coarse_strategy = Rediscretization(DiffusionIntegrator(1.0, 2))),
]
    ## 1D Poisson equation example ##
    K, f, dhh, chh = poisson(1000, [1, 2], 3)
    x, res = solve(K, f, dhh, chh, config; log=true, rtol = 1e-10)
    println("final residual at iteration ", length(res), ": ", res[end])
    @test K * x ≈ f atol=1e-7

    K, f, dhh, chh = poisson((100,100), [1, 2], 3)
    x, res = solve(K, f, dhh, chh, config; log=true, rtol = 1e-10)
    println("final residual at iteration ", length(res), ": ", res[end])
    @test K * x ≈ f atol=1e-7
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
