## DofHandlerHierarchy, ConstraintHandlerHierarchy, SubDofHandlerHierarchy
##
## Provide a uniform hierarchy API for DofHandlers and ConstraintHandlers,
## mirroring the GridHierarchy design. Usable with both geometric multigrid
## (driven by GridHierarchy) and polynomial multigrid.
##
## Convention: index 1 = coarsest level, index end = finest level.


#######################################################################
## DofHandlerHierarchy                                               ##
#######################################################################

"""
    DofHandlerHierarchy{DH <: AbstractDofHandler}

A hierarchy of `DofHandler`s, ordered from coarsest (index 1) to finest (index `end`).

Construct from a `GridHierarchy` via `DofHandlerHierarchy(gh::GridHierarchy)`
(which allocates one `DofHandler` per grid level without any fields), or wrap a
pre-built vector of handlers directly.

# Example
```julia
gh  = GridHierarchy(coarse_grid, 2)
dhh = DofHandlerHierarchy(gh)
add!(dhh, :u, Lagrange{RefLine, 1}())
close!(dhh)
```
"""
struct DofHandlerHierarchy{DH <: AbstractDofHandler}
    handlers::Vector{DH}
end

Base.getindex(dhh::DofHandlerHierarchy, i::Int) = dhh.handlers[i]
Base.length(dhh::DofHandlerHierarchy) = length(dhh.handlers)
Base.lastindex(dhh::DofHandlerHierarchy) = length(dhh.handlers)

"""
    add!(dhh::DofHandlerHierarchy, field_name::Symbol, ip)

Add a field `field_name` with interpolation `ip` to every `DofHandler` in the hierarchy.
"""
function Ferrite.add!(dhh::DofHandlerHierarchy, field_name::Symbol, ip)
    for dh in dhh.handlers
        add!(dh, field_name, ip)
    end
    return dhh
end

"""
    close!(dhh::DofHandlerHierarchy)

Close every `DofHandler` in the hierarchy.
"""
function Ferrite.close!(dhh::DofHandlerHierarchy)
    for dh in dhh.handlers
        close!(dh)
    end
    return dhh
end


#######################################################################
## ConstraintHandlerHierarchy                                        ##
#######################################################################

"""
    ConstraintHandlerHierarchy{CH}

A hierarchy of `ConstraintHandler`s, ordered from coarsest (index 1) to finest (index `end`).

Construct from a `DofHandlerHierarchy` via `ConstraintHandlerHierarchy(dhh)`,
which allocates one `ConstraintHandler` per level.

# Example
```julia
chh = ConstraintHandlerHierarchy(dhh)
add!(chh, dh -> Dirichlet(:u,
        union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right")),
        (x, t) -> 0.0))
close!(chh)
```
"""
struct ConstraintHandlerHierarchy{CH}
    handlers::Vector{CH}
end

"""
    ConstraintHandlerHierarchy(dhh::DofHandlerHierarchy)

Allocate a `ConstraintHandler` for each `DofHandler` in `dhh` (no constraints added yet).
"""
function ConstraintHandlerHierarchy(dhh::DofHandlerHierarchy)
    handlers = [ConstraintHandler(dh) for dh in dhh.handlers]
    return ConstraintHandlerHierarchy(handlers)
end

Base.getindex(chh::ConstraintHandlerHierarchy, i::Int) = chh.handlers[i]
Base.length(chh::ConstraintHandlerHierarchy) = length(chh.handlers)
Base.lastindex(chh::ConstraintHandlerHierarchy) = length(chh.handlers)

"""
    add!(chh::ConstraintHandlerHierarchy, constraint_factory)

Add a constraint to every `ConstraintHandler` in the hierarchy.

`constraint_factory` is a callable with signature `(dh::AbstractDofHandler) -> constraint`
that produces a level-specific constraint for the given dof handler.  This allows
each level to reference its own grid's facetsets while sharing the same constraint
specification.

# Example
```julia
add!(chh, dh -> Dirichlet(:u,
        union(getfacetset(dh.grid, "left"), getfacetset(dh.grid, "right")),
        (x, t) -> 0.0))
```
"""
function Ferrite.add!(chh::ConstraintHandlerHierarchy, constraint_factory)
    for ch in chh.handlers
        add!(ch, constraint_factory(ch.dh))
    end
    return chh
end

"""
    close!(chh::ConstraintHandlerHierarchy)

Close every `ConstraintHandler` in the hierarchy.
"""
function Ferrite.close!(chh::ConstraintHandlerHierarchy)
    for ch in chh.handlers
        close!(ch)
    end
    return chh
end


#######################################################################
## SubDofHandlerHierarchy                                            ##
#######################################################################

"""
    SubDofHandlerHierarchy{SDH <: AbstractDofHandler}

A hierarchy of `SubDofHandler`s for subdomain management, one per grid level,
ordered from coarsest (index 1) to finest (index `end`).

Construct from a `DofHandlerHierarchy` with a callable that maps each `DofHandler`
to the cell set for that level's subdomain:

```julia
sdhh = SubDofHandlerHierarchy(dhh, dh -> getcellset(dh.grid, "subdomain"))
add!(sdhh, :u, Lagrange{RefTriangle, 2}())
```
"""
struct SubDofHandlerHierarchy{SDH <: AbstractDofHandler}
    handlers::Vector{SDH}
end

"""
    SubDofHandlerHierarchy(dhh::DofHandlerHierarchy, cellset_fn)

Create a `SubDofHandler` for each level in `dhh`.  `cellset_fn(dh)` must return
the set of cell indices (e.g. a `Set{Int}`) that belong to the subdomain at that level.
"""
function SubDofHandlerHierarchy(dhh::DofHandlerHierarchy, cellset_fn)
    handlers = [SubDofHandler(dh, cellset_fn(dh)) for dh in dhh.handlers]
    return SubDofHandlerHierarchy(handlers)
end

Base.getindex(sdhh::SubDofHandlerHierarchy, i::Int) = sdhh.handlers[i]
Base.length(sdhh::SubDofHandlerHierarchy) = length(sdhh.handlers)
Base.lastindex(sdhh::SubDofHandlerHierarchy) = length(sdhh.handlers)

"""
    add!(sdhh::SubDofHandlerHierarchy, field_name::Symbol, ip)

Add a field to every `SubDofHandler` in the hierarchy.
"""
function Ferrite.add!(sdhh::SubDofHandlerHierarchy, field_name::Symbol, ip)
    for sdh in sdhh.handlers
        add!(sdh, field_name, ip)
    end
    return sdhh
end
