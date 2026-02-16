module FerriteMultigrid

using Reexport
using LinearAlgebra
using TimerOutputs
import LinearSolve
using SparseArrays
import SparseArrays: AbstractSparseMatrixCSC
import Base: *

using TimerOutputs

using Ferrite
import Ferrite: getorder, AbstractDofHandler, reinit!, AbstractCell, AbstractRefShape
@reexport using AlgebraicMultigrid
import AlgebraicMultigrid as AMG
import AlgebraicMultigrid:
    init,
    solve,
    solve!,
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
