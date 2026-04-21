# FerriteMultigrid.jl

[![Build Status](https://github.com/termi-official/FerriteMultigrid.jl/workflows/CI/badge.svg)](https://github.com/termi-official/FerriteMultigrid.jl/actions)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://termi-official.github.io/FerriteMultigrid.jl/dev/)


**FerriteMultigrid.jl** is a lightweight, flexible polynomial and geometric **multigrid framework** designed for high-order finite element problems in Julia.
It is built on top of [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) and leverages [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) as the potential coarse-grid solver once the approximation is reduced to \( p = 1 \). Users can also combine polynomial and geometric multigrid methods.


## Example Usage

```julia
using FerriteMultigrid

# Define a 1D diffusion problem with p = 2 for the fine grid and p = 1 for the coarse grid and 3 quadrature points.
K, f, dhh, chh = poisson(1000, [1, 2], 3)

# Define a p-multigrid configuration
config = pmultigrid_config()

# Solve using the p-multigrid solver
x, res = solve(K, f, dhh, chh, config; log = true, rtol = 1e-10)
```

## Acknowledgement

This framework is primarily developed by [Abdelrahman Fathy](https://github.com/Abdelrahman912) at the [chair of continuum mechanics at Ruhr University Bochum](https://www.lkm.ruhr-uni-bochum.de/) under 
the supervision of [Dennis Ogiermann](https://github.com/termi-official).
