using FerriteMultigrid
import FerriteMultigrid: _element_mass_matrix!, element_prolongator!, build_prolongator
using Test
using Ferrite
using SparseArrays


include("test_prolongator.jl")
include("test_parameters.jl")
include("test_examples.jl")
include("test_precs.jl")
