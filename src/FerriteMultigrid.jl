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

using FerriteOperators
import FerriteOperators:
    AbstractBilinearIntegrator, AbstractVolumetricElementCache,
    AbstractAssemblyStrategy,
    setup_operator, update_operator!,
    setup_element_cache, assemble_element!,
    setup_transfer_operator,
    MassProlongatorIntegrator,
    SameGridTransferCellIterator, NestedGridTransferCellIterator,
    getrowdofs, getcolumndofs,
    QuadratureRuleCollection, getquadraturerule,
    SequentialAssemblyStrategy, SequentialCPUDevice
import Ferrite: get_grid

include("fe.jl")
include("multigrid_problems.jl")
include("prolongator.jl")
include("pmultigrid.jl")
include("multilevel.jl")
include("gallery.jl")
include("precs.jl")
include("geometric_multigrid.jl")

export 
    AbstractPMultigrid,
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
    solve,
    # Geometric multigrid
    NestedGrid1D,
    build_geometric_prolongator_1d,
    gmultigrid
end
