module FerriteMultigrid

using Reexport
using LinearAlgebra
import LinearSolve
using SparseArrays
import SparseArrays: AbstractSparseMatrixCSC
#import CommonSolve: solve, solve!, init
import Base: *

using Ferrite
import Ferrite: getorder, AbstractDofHandler, reinit!, AbstractCell, AbstractRefShape
@reexport using AlgebraicMultigrid
import AlgebraicMultigrid as AMG
import AlgebraicMultigrid:
    AMGAlg,
    Level,
    CoarseSolver,
    MultiLevel,
    residual!,
    coarse_x!,
    coarse_b!,
    Pinv,
    _solve,
    MultiLevelWorkspace

include("fe.jl")
include("multigrid_problems.jl")
include("prolongator.jl")
include("pmultigrid.jl")
include("multilevel.jl")
include("gallery.jl")
include("precs.jl")

export
    FESpace,
    AbstractPMultigrid,
    assemble,
    DiffusionMultigrid,
    LinearElasticityMultigrid,
    HyperelasticityMultigrid,
    ConstantCoefficient,
    Galerkin,
    Rediscretization,
    DirectProjection,
    StepProjection,
    SmoothedAggregationCoarseSolver,
    RugeStubenCoarseSolver,
    pmultigrid_config,
    Pinv,
    PMultigridPreconBuilder,
    AbstractCoefficient,
    solve
end
