## Geometric Multigrid on Nested Grids
##
## Provides uniform grid refinement and a full geometric multigrid method analogous
## to `pmultigrid` but driven by a hierarchy of grids produced by `uniform_refinement`.
##
## Key API:
##   uniform_refinement(grid)  → (fine_grid, fine2coarse, child_ref_coords)
##   GridHierarchy(grid, n_refinements)
##   gmultigrid_config(; coarse_strategy = Galerkin())
##   gmultigrid(A, gh, dh_hierarchy, ch_hierarchy, config, pcoarse_solver)

#######################################################################
## Node-creation helpers                                             ##
#######################################################################

## Return the midpoint node id for edge (na, nb), creating it if not yet present.
function _edge_midpoint!(fine_nodes::Vector, edge_dict, node_parents, na, nb, coarse_nodes)
    key = minmax(na, nb)
    if !haskey(edge_dict, key)
        ca = get_node_coordinate(coarse_nodes[na])
        cb = get_node_coordinate(coarse_nodes[nb])
        push!(fine_nodes, Node((ca + cb) / 2))
        new_id = length(fine_nodes)
        edge_dict[key] = new_id
        node_parents[new_id] = [na, nb]
    end
    return edge_dict[key]
end

## Return the face-center node id for a face defined by the given node ids.
function _face_center!(fine_nodes::Vector, face_dict, node_parents, ns, coarse_nodes)
    key = Tuple(sort(collect(ns)))
    if !haskey(face_dict, key)
        c = sum(get_node_coordinate(coarse_nodes[n]) for n in ns) / length(ns)
        push!(fine_nodes, Node(c))
        new_id = length(fine_nodes)
        face_dict[key] = new_id
        node_parents[new_id] = collect(ns)
    end
    return face_dict[key]
end

## Create a unique body-center node for a coarse cell (not shared with neighbours).
function _body_center!(fine_nodes::Vector, node_parents, ns, coarse_nodes)
    c = sum(get_node_coordinate(coarse_nodes[n]) for n in ns) / length(ns)
    push!(fine_nodes, Node(c))
    new_id = length(fine_nodes)
    node_parents[new_id] = collect(ns)
    return new_id
end


#######################################################################
## Boundary-set propagation                                          ##
#######################################################################

## Propagate facetsets from the coarse grid to a fine grid.
## A fine facet is in the set if all its nodes are "reachable" from the coarse facetset:
##   - original coarse corner nodes → in boundary cluster
##   - new node → in cluster if ALL its parent nodes are in the cluster
function _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    coarse_facetsets = Ferrite.getfacetsets(coarse_grid)
    fine_facetsets   = Dict{String, Set{Ferrite.FacetIndex}}()
    coarse_cells     = getcells(coarse_grid)

    for (name, coarse_fs) in coarse_facetsets
        # Seed boundary cluster with nodes from the coarse facetset
        boundary_nodes = Set{Int}()
        for fi in coarse_fs
            cell_id, facet_idx = fi
            for n in Ferrite.facets(coarse_cells[cell_id])[facet_idx]
                push!(boundary_nodes, n)
            end
        end

        # Propagate to new (midpoint / face-center / body-center) nodes
        changed = true
        while changed
            changed = false
            for (new_id, parents) in node_parents
                if new_id ∉ boundary_nodes && all(p ∈ boundary_nodes for p in parents)
                    push!(boundary_nodes, new_id)
                    changed = true
                end
            end
        end

        # Collect fine facets whose every node is in the boundary cluster
        fine_fs = Set{Ferrite.FacetIndex}()
        for (fine_id, fine_cell) in enumerate(fine_cells)
            for (facet_idx, facet_nodes) in enumerate(Ferrite.facets(fine_cell))
                if all(n ∈ boundary_nodes for n in facet_nodes)
                    push!(fine_fs, Ferrite.FacetIndex(fine_id, facet_idx))
                end
            end
        end

        isempty(fine_fs) || (fine_facetsets[name] = fine_fs)
    end

    return fine_facetsets
end

## Propagate nodesets: a fine node is in the set if it is an original coarse node in the
## set, or it is an edge midpoint whose both parents are in the set.
function _propagate_nodesets(coarse_grid, node_parents)
    fine_nodesets = Dict{String, Set{Int}}()
    for (name, coarse_ns) in Ferrite.getnodesets(coarse_grid)
        fine_ns = Set{Int}(coarse_ns)
        for (new_id, parents) in node_parents
            if all(p ∈ coarse_ns for p in parents)
                push!(fine_ns, new_id)
            end
        end
        fine_nodesets[name] = fine_ns
    end
    return fine_nodesets
end


#######################################################################
## uniform_refinement  (one refinement level)                        ##
#######################################################################

"""
    uniform_refinement(grid::Grid) → (fine_grid, fine2coarse, child_ref_coords)

Uniformly refine `grid` by splitting each cell into smaller cells of the same type:

| Cell type       | Children |
|-----------------|----------|
| `Line`          | 2        |
| `Triangle`      | 4        |
| `Quadrilateral` | 4        |
| `Tetrahedron`   | 8        |
| `Hexahedron`    | 8        |

Returns:
- `fine_grid`        – refined `Grid`
- `fine2coarse`      – `fine2coarse[fine_id] = coarse_id`
- `child_ref_coords` – `child_ref_coords[fine_id]` = positions of that fine cell's nodes
                       in the *parent* coarse reference element

Facetsets and nodesets from `grid` are propagated to `fine_grid`.
"""
function uniform_refinement end

# ──────────────────────────────── Line (1D) ─────────────────────────────────

function uniform_refinement(coarse_grid::Grid{1, Line, T}) where T
    coarse_nodes = getnodes(coarse_grid)
    coarse_cells = getcells(coarse_grid)
    n_coarse     = length(coarse_cells)

    fine_nodes   = collect(coarse_nodes)
    edge_mid     = Dict{NTuple{2,Int}, Int}()
    node_parents = Dict{Int, Vector{Int}}()

    fine_cells        = Vector{Line}(undef, 2 * n_coarse)
    fine2coarse       = Vector{Int}(undef,  2 * n_coarse)
    child_ref_coords  = Vector{Vector{Vec{1,T}}}(undef, 2 * n_coarse)

    for (c, cell) in enumerate(coarse_cells)
        n1, n2 = cell.nodes
        m = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
        k = 2 * (c - 1)
        fine_cells[k+1] = Line((n1, m));  fine2coarse[k+1] = c
        fine_cells[k+2] = Line((m, n2));  fine2coarse[k+2] = c
        child_ref_coords[k+1] = [Vec((-1.0,)), Vec((0.0,))]
        child_ref_coords[k+2] = [Vec(( 0.0,)), Vec((1.0,))]
    end

    fine_facetsets = _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    fine_nodesets  = _propagate_nodesets(coarse_grid, node_parents)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets)
    return fine_grid, fine2coarse, child_ref_coords
end

# ──────────────────────────────── Triangle (2D) ──────────────────────────────
# Each triangle → 4 children (3 corner + 1 central)
# RefTriangle nodes: v1=(0,0), v2=(1,0), v3=(0,1)
# Edges: (1,2),(2,3),(3,1)  → midpoints e12, e23, e13

function uniform_refinement(coarse_grid::Grid{2, Triangle, T}) where T
    coarse_nodes = getnodes(coarse_grid)
    coarse_cells = getcells(coarse_grid)
    n_coarse     = length(coarse_cells)

    fine_nodes   = collect(coarse_nodes)
    edge_mid     = Dict{NTuple{2,Int}, Int}()
    node_parents = Dict{Int, Vector{Int}}()

    fine_cells       = Vector{Triangle}(undef, 4 * n_coarse)
    fine2coarse      = Vector{Int}(undef,  4 * n_coarse)
    child_ref_coords = Vector{Vector{Vec{2,T}}}(undef, 4 * n_coarse)

    # Reference coordinates of children (RefTriangle nodes: (0,0),(1,0),(0,1))
    crc = [
        [Vec((0.0,0.0)), Vec((0.5,0.0)), Vec((0.0,0.5))],  # child 1: corner v1
        [Vec((0.5,0.0)), Vec((1.0,0.0)), Vec((0.5,0.5))],  # child 2: corner v2
        [Vec((0.0,0.5)), Vec((0.5,0.5)), Vec((0.0,1.0))],  # child 3: corner v3
        [Vec((0.5,0.0)), Vec((0.5,0.5)), Vec((0.0,0.5))],  # child 4: central
    ]

    for (c, cell) in enumerate(coarse_cells)
        n1, n2, n3 = cell.nodes
        e12 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
        e23 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
        e13 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n3, coarse_nodes)
        k = 4 * (c - 1)
        fine_cells[k+1] = Triangle((n1, e12, e13)); fine2coarse[k+1] = c; child_ref_coords[k+1] = crc[1]
        fine_cells[k+2] = Triangle((e12, n2, e23)); fine2coarse[k+2] = c; child_ref_coords[k+2] = crc[2]
        fine_cells[k+3] = Triangle((e13, e23, n3)); fine2coarse[k+3] = c; child_ref_coords[k+3] = crc[3]
        fine_cells[k+4] = Triangle((e12, e23, e13)); fine2coarse[k+4] = c; child_ref_coords[k+4] = crc[4]
    end

    fine_facetsets = _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    fine_nodesets  = _propagate_nodesets(coarse_grid, node_parents)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets)
    return fine_grid, fine2coarse, child_ref_coords
end

# ──────────────────────────── Quadrilateral (2D) ─────────────────────────────
# Each quad → 4 children (one per quadrant)
# RefQuadrilateral nodes: v1=(-1,-1), v2=(1,-1), v3=(1,1), v4=(-1,1)
# Edges: (1,2)[bottom], (2,3)[right], (3,4)[top], (4,1)[left]

function uniform_refinement(coarse_grid::Grid{2, Quadrilateral, T}) where T
    coarse_nodes = getnodes(coarse_grid)
    coarse_cells = getcells(coarse_grid)
    n_coarse     = length(coarse_cells)

    fine_nodes   = collect(coarse_nodes)
    edge_mid     = Dict{NTuple{2,Int}, Int}()
    face_ctr     = Dict{NTuple{4,Int}, Int}()  # quad face center (only 1 per quad here)
    node_parents = Dict{Int, Vector{Int}}()

    fine_cells       = Vector{Quadrilateral}(undef, 4 * n_coarse)
    fine2coarse      = Vector{Int}(undef,  4 * n_coarse)
    child_ref_coords = Vector{Vector{Vec{2,T}}}(undef, 4 * n_coarse)

    # Reference coordinates for 4 child quads (following Ferrite CCW node order)
    # Child 1 (BL): nodes [v1, e12, fc, e41]
    # Child 2 (BR): nodes [e12, v2, e23, fc]
    # Child 3 (TR): nodes [fc, e23, v3, e34]
    # Child 4 (TL): nodes [e41, fc, e34, v4]
    crc = [
        [Vec((-1.0,-1.0)), Vec((0.0,-1.0)), Vec((0.0,0.0)), Vec((-1.0,0.0))],
        [Vec(( 0.0,-1.0)), Vec((1.0,-1.0)), Vec((1.0,0.0)), Vec(( 0.0,0.0))],
        [Vec(( 0.0, 0.0)), Vec((1.0, 0.0)), Vec((1.0,1.0)), Vec(( 0.0,1.0))],
        [Vec((-1.0, 0.0)), Vec((0.0, 0.0)), Vec((0.0,1.0)), Vec((-1.0,1.0))],
    ]

    for (c, cell) in enumerate(coarse_cells)
        n1, n2, n3, n4 = cell.nodes
        e12 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
        e23 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
        e34 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n4, coarse_nodes)
        e41 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n1, coarse_nodes)
        fc  = _face_center!(fine_nodes, face_ctr, node_parents, (n1, n2, n3, n4), coarse_nodes)
        k = 4 * (c - 1)
        fine_cells[k+1] = Quadrilateral((n1, e12, fc, e41)); fine2coarse[k+1] = c; child_ref_coords[k+1] = crc[1]
        fine_cells[k+2] = Quadrilateral((e12, n2, e23, fc)); fine2coarse[k+2] = c; child_ref_coords[k+2] = crc[2]
        fine_cells[k+3] = Quadrilateral((fc, e23, n3, e34)); fine2coarse[k+3] = c; child_ref_coords[k+3] = crc[3]
        fine_cells[k+4] = Quadrilateral((e41, fc, e34, n4)); fine2coarse[k+4] = c; child_ref_coords[k+4] = crc[4]
    end

    fine_facetsets = _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    fine_nodesets  = _propagate_nodesets(coarse_grid, node_parents)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets)
    return fine_grid, fine2coarse, child_ref_coords
end

# ──────────────────────────── Tetrahedron (3D) ───────────────────────────────
# Each tet → 8 children (4 corner + 4 inner octahedron)
# RefTetrahedron nodes: v1=(0,0,0), v2=(1,0,0), v3=(0,1,0), v4=(0,0,1)
# Edges: (1,2),(2,3),(3,1),(1,4),(2,4),(3,4) → midpoints e12,e23,e13,e14,e24,e34

function uniform_refinement(coarse_grid::Grid{3, Tetrahedron, T}) where T
    coarse_nodes = getnodes(coarse_grid)
    coarse_cells = getcells(coarse_grid)
    n_coarse     = length(coarse_cells)

    fine_nodes   = collect(coarse_nodes)
    edge_mid     = Dict{NTuple{2,Int}, Int}()
    node_parents = Dict{Int, Vector{Int}}()

    fine_cells       = Vector{Tetrahedron}(undef, 8 * n_coarse)
    fine2coarse      = Vector{Int}(undef,  8 * n_coarse)
    child_ref_coords = Vector{Vector{Vec{3,T}}}(undef, 8 * n_coarse)

    # Reference coordinates for 8 child tets
    # All 8 have Jacobian = 0.125 (verified).
    crc = [
        # Corner tets (mirror the parent's structure at 1/2 scale)
        [Vec((0.0,0.0,0.0)), Vec((0.5,0.0,0.0)), Vec((0.0,0.5,0.0)), Vec((0.0,0.0,0.5))], # c1: v1
        [Vec((0.5,0.0,0.0)), Vec((1.0,0.0,0.0)), Vec((0.5,0.5,0.0)), Vec((0.5,0.0,0.5))], # c2: v2
        [Vec((0.0,0.5,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,1.0,0.0)), Vec((0.0,0.5,0.5))], # c3: v3
        [Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5)), Vec((0.0,0.0,1.0))], # c4: v4
        # Inner octahedron tets (sharing the e13–e24 diagonal)
        [Vec((0.0,0.5,0.0)), Vec((0.5,0.0,0.0)), Vec((0.5,0.5,0.0)), Vec((0.5,0.0,0.5))], # c5
        [Vec((0.0,0.5,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,0.5,0.5)), Vec((0.5,0.0,0.5))], # c6
        [Vec((0.0,0.5,0.0)), Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5))], # c7
        [Vec((0.5,0.0,0.0)), Vec((0.0,0.5,0.0)), Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5))], # c8
    ]

    for (c, cell) in enumerate(coarse_cells)
        n1, n2, n3, n4 = cell.nodes
        e12 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
        e23 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
        e13 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n3, coarse_nodes)
        e14 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n4, coarse_nodes)
        e24 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n4, coarse_nodes)
        e34 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n4, coarse_nodes)
        k = 8 * (c - 1)
        fine_cells[k+1] = Tetrahedron((n1,  e12, e13, e14)); fine2coarse[k+1] = c; child_ref_coords[k+1] = crc[1]
        fine_cells[k+2] = Tetrahedron((e12, n2,  e23, e24)); fine2coarse[k+2] = c; child_ref_coords[k+2] = crc[2]
        fine_cells[k+3] = Tetrahedron((e13, e23, n3,  e34)); fine2coarse[k+3] = c; child_ref_coords[k+3] = crc[3]
        fine_cells[k+4] = Tetrahedron((e14, e24, e34, n4 )); fine2coarse[k+4] = c; child_ref_coords[k+4] = crc[4]
        fine_cells[k+5] = Tetrahedron((e13, e12, e23, e24)); fine2coarse[k+5] = c; child_ref_coords[k+5] = crc[5]
        fine_cells[k+6] = Tetrahedron((e13, e23, e34, e24)); fine2coarse[k+6] = c; child_ref_coords[k+6] = crc[6]
        fine_cells[k+7] = Tetrahedron((e13, e14, e24, e34)); fine2coarse[k+7] = c; child_ref_coords[k+7] = crc[7]
        fine_cells[k+8] = Tetrahedron((e12, e13, e14, e24)); fine2coarse[k+8] = c; child_ref_coords[k+8] = crc[8]
    end

    fine_facetsets = _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    fine_nodesets  = _propagate_nodesets(coarse_grid, node_parents)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets)
    return fine_grid, fine2coarse, child_ref_coords
end

# ──────────────────────────── Hexahedron (3D) ────────────────────────────────
# Each hex → 8 children (one per octant of [-1,1]³)
# RefHexahedron nodes: v1=(-1,-1,-1), v2=(1,-1,-1), v3=(1,1,-1), v4=(-1,1,-1),
#                      v5=(-1,-1,1),  v6=(1,-1,1),  v7=(1,1,1),  v8=(-1,1,1)
# Ferrite Hex face ordering: f1=(1,4,3,2)[bot], f2=(1,2,6,5)[frt], f3=(2,3,7,6)[rht],
#                            f4=(3,4,8,7)[bck], f5=(1,5,8,4)[lft], f6=(5,6,7,8)[top]

function uniform_refinement(coarse_grid::Grid{3, Hexahedron, T}) where T
    coarse_nodes = getnodes(coarse_grid)
    coarse_cells = getcells(coarse_grid)
    n_coarse     = length(coarse_cells)

    fine_nodes   = collect(coarse_nodes)
    edge_mid     = Dict{NTuple{2,Int}, Int}()
    face_ctr     = Dict{NTuple{4,Int}, Int}()
    node_parents = Dict{Int, Vector{Int}}()

    fine_cells       = Vector{Hexahedron}(undef, 8 * n_coarse)
    fine2coarse      = Vector{Int}(undef,  8 * n_coarse)
    child_ref_coords = Vector{Vector{Vec{3,T}}}(undef, 8 * n_coarse)

    # Reference coordinates for the 8 child hexes.
    # Each child occupies one octant of [-1,1]³; nodes follow Ferrite Hex ordering:
    # [BL-bot, BR-bot, TR-bot, TL-bot, BL-top, BR-top, TR-top, TL-top]
    crc = [
        # c1: x∈[-1,0], y∈[-1,0], z∈[-1,0]
        [Vec((-1.0,-1.0,-1.0)), Vec((0.0,-1.0,-1.0)), Vec((0.0,0.0,-1.0)), Vec((-1.0,0.0,-1.0)),
         Vec((-1.0,-1.0, 0.0)), Vec((0.0,-1.0, 0.0)), Vec((0.0,0.0, 0.0)), Vec((-1.0,0.0, 0.0))],
        # c2: x∈[0,1], y∈[-1,0], z∈[-1,0]
        [Vec((0.0,-1.0,-1.0)), Vec((1.0,-1.0,-1.0)), Vec((1.0,0.0,-1.0)), Vec((0.0,0.0,-1.0)),
         Vec((0.0,-1.0, 0.0)), Vec((1.0,-1.0, 0.0)), Vec((1.0,0.0, 0.0)), Vec((0.0,0.0, 0.0))],
        # c3: x∈[0,1], y∈[0,1], z∈[-1,0]
        [Vec((0.0,0.0,-1.0)), Vec((1.0,0.0,-1.0)), Vec((1.0,1.0,-1.0)), Vec((0.0,1.0,-1.0)),
         Vec((0.0,0.0, 0.0)), Vec((1.0,0.0, 0.0)), Vec((1.0,1.0, 0.0)), Vec((0.0,1.0, 0.0))],
        # c4: x∈[-1,0], y∈[0,1], z∈[-1,0]
        [Vec((-1.0,0.0,-1.0)), Vec((0.0,0.0,-1.0)), Vec((0.0,1.0,-1.0)), Vec((-1.0,1.0,-1.0)),
         Vec((-1.0,0.0, 0.0)), Vec((0.0,0.0, 0.0)), Vec((0.0,1.0, 0.0)), Vec((-1.0,1.0, 0.0))],
        # c5: x∈[-1,0], y∈[-1,0], z∈[0,1]
        [Vec((-1.0,-1.0,0.0)), Vec((0.0,-1.0,0.0)), Vec((0.0,0.0,0.0)), Vec((-1.0,0.0,0.0)),
         Vec((-1.0,-1.0,1.0)), Vec((0.0,-1.0,1.0)), Vec((0.0,0.0,1.0)), Vec((-1.0,0.0,1.0))],
        # c6: x∈[0,1], y∈[-1,0], z∈[0,1]
        [Vec((0.0,-1.0,0.0)), Vec((1.0,-1.0,0.0)), Vec((1.0,0.0,0.0)), Vec((0.0,0.0,0.0)),
         Vec((0.0,-1.0,1.0)), Vec((1.0,-1.0,1.0)), Vec((1.0,0.0,1.0)), Vec((0.0,0.0,1.0))],
        # c7: x∈[0,1], y∈[0,1], z∈[0,1]
        [Vec((0.0,0.0,0.0)), Vec((1.0,0.0,0.0)), Vec((1.0,1.0,0.0)), Vec((0.0,1.0,0.0)),
         Vec((0.0,0.0,1.0)), Vec((1.0,0.0,1.0)), Vec((1.0,1.0,1.0)), Vec((0.0,1.0,1.0))],
        # c8: x∈[-1,0], y∈[0,1], z∈[0,1]
        [Vec((-1.0,0.0,0.0)), Vec((0.0,0.0,0.0)), Vec((0.0,1.0,0.0)), Vec((-1.0,1.0,0.0)),
         Vec((-1.0,0.0,1.0)), Vec((0.0,0.0,1.0)), Vec((0.0,1.0,1.0)), Vec((-1.0,1.0,1.0))],
    ]

    for (c, cell) in enumerate(coarse_cells)
        n1,n2,n3,n4,n5,n6,n7,n8 = cell.nodes
        # 12 edge midpoints
        me12 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
        me23 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
        me34 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n4, coarse_nodes)
        me41 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n1, coarse_nodes)
        me56 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n5, n6, coarse_nodes)
        me67 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n6, n7, coarse_nodes)
        me78 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n7, n8, coarse_nodes)
        me85 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n8, n5, coarse_nodes)
        me15 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n5, coarse_nodes)
        me26 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n6, coarse_nodes)
        me37 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n7, coarse_nodes)
        me48 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n8, coarse_nodes)
        # 6 face centers  (face node ordering from reference_faces(RefHexahedron))
        fc_bot = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n2,n3,n4), coarse_nodes)
        fc_frt = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n2,n6,n5), coarse_nodes)
        fc_rht = _face_center!(fine_nodes, face_ctr, node_parents, (n2,n3,n7,n6), coarse_nodes)
        fc_bck = _face_center!(fine_nodes, face_ctr, node_parents, (n3,n4,n8,n7), coarse_nodes)
        fc_lft = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n5,n8,n4), coarse_nodes)
        fc_top = _face_center!(fine_nodes, face_ctr, node_parents, (n5,n6,n7,n8), coarse_nodes)
        # 1 body center (unique per cell)
        body   = _body_center!(fine_nodes, node_parents, (n1,n2,n3,n4,n5,n6,n7,n8), coarse_nodes)

        k = 8 * (c - 1)
        fine_cells[k+1] = Hexahedron((n1,me12,fc_bot,me41,me15,fc_frt,body,  fc_lft)); fine2coarse[k+1] = c; child_ref_coords[k+1] = crc[1]
        fine_cells[k+2] = Hexahedron((me12,n2,me23,fc_bot,fc_frt,me26,fc_rht,body  )); fine2coarse[k+2] = c; child_ref_coords[k+2] = crc[2]
        fine_cells[k+3] = Hexahedron((fc_bot,me23,n3,me34,body,  fc_rht,me37,fc_bck)); fine2coarse[k+3] = c; child_ref_coords[k+3] = crc[3]
        fine_cells[k+4] = Hexahedron((me41,fc_bot,me34,n4,fc_lft,body,  fc_bck,me48)); fine2coarse[k+4] = c; child_ref_coords[k+4] = crc[4]
        fine_cells[k+5] = Hexahedron((me15,fc_frt,body,  fc_lft,n5,me56,fc_top,me85)); fine2coarse[k+5] = c; child_ref_coords[k+5] = crc[5]
        fine_cells[k+6] = Hexahedron((fc_frt,me26,fc_rht,body,  me56,n6,me67,fc_top)); fine2coarse[k+6] = c; child_ref_coords[k+6] = crc[6]
        fine_cells[k+7] = Hexahedron((body,  fc_rht,me37,fc_bck,fc_top,me67,n7,me78)); fine2coarse[k+7] = c; child_ref_coords[k+7] = crc[7]
        fine_cells[k+8] = Hexahedron((fc_lft,body,  fc_bck,me48,me85,fc_top,me78,n8 )); fine2coarse[k+8] = c; child_ref_coords[k+8] = crc[8]
    end

    fine_facetsets = _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    fine_nodesets  = _propagate_nodesets(coarse_grid, node_parents)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets)
    return fine_grid, fine2coarse, child_ref_coords
end


#######################################################################
## GridHierarchy                                                     ##
#######################################################################

"""
    GridHierarchy{G}

A hierarchy of nested grids produced by repeated `uniform_refinement`.

- `grids[1]`      is the coarsest grid
- `grids[end]`    is the finest grid
- `fine2coarse[k]`      maps fine cell ids on level `k+1` → coarse cell ids on level `k`
- `child_ref_coords[k]` stores the reference coordinates of fine-cell nodes in their
                        parent coarse reference element, for level transition `k+1 → k`
"""
struct GridHierarchy{G}
    grids::Vector{G}
    fine2coarse::Vector{Vector{Int}}
    child_ref_coords::Vector{Vector}   # child_ref_coords[k][fine_id] = Vector{Vec}
end

"""
    GridHierarchy(coarse_grid::Grid, n_refinements::Int)

Build a grid hierarchy with `n_refinements` levels of uniform refinement above
`coarse_grid`.  The resulting hierarchy has `n_refinements + 1` grids.
"""
function GridHierarchy(coarse_grid::Grid, n_refinements::Int)
    @assert n_refinements >= 1 "Need at least one refinement level"
    grids            = [coarse_grid]
    fine2coarse_maps = Vector{Int}[]
    crc_maps         = Vector[]

    for _ in 1:n_refinements
        fine_grid, f2c, crc = uniform_refinement(grids[end])
        push!(grids, fine_grid)
        push!(fine2coarse_maps, f2c)
        push!(crc_maps, crc)
    end

    return GridHierarchy(grids, fine2coarse_maps, crc_maps)
end

Base.length(gh::GridHierarchy) = length(gh.grids)


#######################################################################
## Geometric multigrid configuration                                 ##
#######################################################################

"""
    GMultigridConfiguration{TC}

Configuration for the geometric multigrid method (analogous to `PMultigridConfiguration`).
"""
struct GMultigridConfiguration{TC<:AbstractCoarseningStrategy}
    coarse_strategy::TC
end

"""
    gmultigrid_config(; coarse_strategy = Galerkin())

Create a `GMultigridConfiguration`.
"""
gmultigrid_config(; coarse_strategy = Galerkin()) = GMultigridConfiguration(coarse_strategy)


#######################################################################
## Prolongator assembly for nested grids                             ##
#######################################################################

"""
    build_geometric_prolongator(dh_fine, dh_coarse, fine2coarse, child_ref_coords)

Assemble the prolongation matrix P for a geometric level transition using
`NestedMassProlongatorIntegrator` via FerriteOperators.
"""
function build_geometric_prolongator(
        dh_fine::DofHandler,
        dh_coarse::DofHandler,
        fine2coarse::AbstractVector{Int},
        child_ref_coords::AbstractVector;
        qr_order::Int = 2 * order(dh_fine),
    )
    field_name = first(Ferrite.getfieldnames(dh_fine))
    integrator = NestedMassProlongatorIntegrator(QuadratureRuleCollection(qr_order), field_name)
    strategy   = SequentialAssemblyStrategy(SequentialCPUDevice())

    op = setup_nested_transfer_operator(strategy, integrator,
                                        dh_fine, dh_coarse, fine2coarse, child_ref_coords)
    update_operator!(op, nothing)

    # Normalise rows: a fine dof shared by multiple fine cells accumulates multiple
    # element contributions; each row must be divided by the contribution count.
    row_contrib = zeros(Int, ndofs(dh_fine))
    for tc in NestedGridTransferCellIterator(dh_fine, dh_coarse, fine2coarse, child_ref_coords)
        for rdof in getrowdofs(tc)
            row_contrib[rdof] += 1
        end
    end
    normalize_rows!(op.P, row_contrib)

    return op.P
end


#######################################################################
## gmultigrid                                                        ##
#######################################################################

"""
    gmultigrid(A, gh, dh_hierarchy, ch_hierarchy, config, pcoarse_solver; kwargs...)

Build a geometric multigrid preconditioner / solver for `Ax = b`.

# Arguments
- `A`             – assembled fine-grid matrix
- `gh`            – [`GridHierarchy`](@ref) (from coarse to fine)
- `dh_hierarchy`  – `AbstractVector{DofHandler}`, one per grid level (index 1 = coarsest)
- `ch_hierarchy`  – `AbstractVector{ConstraintHandler}`, one per grid level
- `config`        – [`GMultigridConfiguration`](@ref)
- `pcoarse_solver` – callable that returns a coarse-grid solver given the coarse matrix

# Keyword arguments
- `p`            – parameter passed to `update_operator!` (default `nothing`)
- `presmoother`  / `postsmoother` – AlgebraicMultigrid smoother (default `GaussSeidel()`)

# Coarsening strategies
- `Galerkin()` – coarse-grid matrix = R A P  (Galerkin projection)
- `Rediscretization(integrator)` – re-assembles the operator on each coarse grid
"""
function gmultigrid(
        A::SparseMatrixCSC{T},
        gh::GridHierarchy,
        dh_hierarchy::AbstractVector,
        ch_hierarchy::AbstractVector,
        config::GMultigridConfiguration,
        pcoarse_solver;
        p          = nothing,
        presmoother  = GaussSeidel(),
        postsmoother = GaussSeidel(),
    ) where T

    n_levels = length(gh) - 1  # number of level transitions (1 = one coarsening step)
    @assert n_levels >= 1
    @assert length(dh_hierarchy) == length(gh) "dh_hierarchy must have length $(length(gh))"
    @assert length(ch_hierarchy) == length(gh) "ch_hierarchy must have length $(length(gh))"

    # AlgebraicMultigrid level list: levels[1] = finest, levels[end] = one above coarsest
    levels = Level{SparseMatrixCSC{T,Int}, SparseMatrixCSC{T,Int}, Adjoint{T, SparseMatrixCSC{T,Int}}}[]
    w      = MultiLevelWorkspace(Val{1}, T)
    residual!(w, size(A, 1))

    cur_A = A

    # Iterate from finest → coarsest
    for k in n_levels:-1:1
        dh_fine   = dh_hierarchy[k+1]   # fine level   (gh.grids[k+1])
        dh_coarse = dh_hierarchy[k]     # coarse level (gh.grids[k])
        ch_coarse = ch_hierarchy[k]
        f2c  = gh.fine2coarse[k]
        crc  = gh.child_ref_coords[k]

        P = @timeit_debug "build geometric prolongator" build_geometric_prolongator(
                dh_fine, dh_coarse, f2c, crc)
        R = @timeit_debug "build geometric restriction" P'

        push!(levels, Level(cur_A, P, R))

        cs = config.coarse_strategy
        if cs isa Galerkin
            cur_A = @timeit_debug "RAP" R * cur_A * P
        elseif cs isa Rediscretization
            coarse_op = @timeit_debug "setup coarse operator" setup_operator(cs.strategy, cs.integrator, dh_coarse)
            @timeit_debug "assemble coarse operator" update_operator!(coarse_op, p)
            apply!(coarse_op.A, ch_coarse)
            cur_A = coarse_op.A
        else
            error("Unknown coarsening strategy: $cs")
        end

        coarse_x!(w, size(cur_A, 1))
        coarse_b!(w, size(cur_A, 1))
        residual!(w, size(cur_A, 1))
    end

    coarse_solver = @timeit_debug "coarse solver setup" pcoarse_solver(cur_A)
    return MultiLevel(levels, cur_A, coarse_solver, presmoother, postsmoother, w)
end

#######################################################################
## Nested grid data structure (1D)                                   ##
#######################################################################

"""
    NestedGrid1D

Stores a coarse and fine 1D grid (Line elements) together with the mapping from each fine
cell to its parent coarse cell and the reference coordinates of the fine cell's nodes
inside the parent element.

# Fields
- `coarse_grid` / `fine_grid`     – Ferrite `Grid`s
- `coarse_dh` / `fine_dh`         – `DofHandler`s (scalar field `:u`, Lagrange P1)
- `fine2coarse :: Vector{Int}`     – `fine2coarse[fine_id] = coarse_id`
- `child_ref_coords`               – for each fine cell, the reference coords of its two
                                     nodes inside the parent coarse reference element [-1,1]
"""
struct NestedGrid1D
    coarse_grid
    fine_grid
    coarse_dh::DofHandler
    fine_dh::DofHandler
    fine2coarse::Vector{Int}
    child_ref_coords::Vector{Vector{Vec{1,Float64}}}
end

"""
    NestedGrid1D(N_coarse::Int; xₗ = 0.0, xᵣ = 1.0, order = 1)

Build a uniform 1D coarse grid with `N_coarse` cells and a fine grid with `2*N_coarse`
cells (uniform bisection).  The mapping and reference coordinates are computed
automatically.

`order` controls the Lagrange interpolation order for both grids.
"""
function NestedGrid1D(N_coarse::Int; xₗ::Float64 = 0.0, xᵣ::Float64 = 1.0, order::Int = 1)
    # Build grids
    coarse_grid = generate_grid(Line, (N_coarse,),        Vec((xₗ,)), Vec((xᵣ,)))
    fine_grid   = generate_grid(Line, (2 * N_coarse,),   Vec((xₗ,)), Vec((xᵣ,)))

    # DofHandlers (scalar `:u`, Lagrange Pk)
    coarse_dh = _make_1d_dh(coarse_grid, order)
    fine_dh   = _make_1d_dh(fine_grid,   order)

    # Build fine→coarse mapping.
    # Uniform bisection: fine cell i is child of coarse cell ceil(i/2).
    N_fine = 2 * N_coarse
    fine2coarse = [div(i - 1, 2) + 1 for i in 1:N_fine]

    # Reference coordinates of the fine cell's nodes inside the parent coarse element.
    # In the reference interval [-1, 1]:
    #   - odd fine cell (left child):  nodes at (-1, 0)
    #   - even fine cell (right child): nodes at (0, +1)
    child_ref_coords = Vector{Vector{Vec{1,Float64}}}(undef, N_fine)
    for i in 1:N_fine
        if isodd(i)
            child_ref_coords[i] = [Vec((-1.0,)), Vec((0.0,))]
        else
            child_ref_coords[i] = [Vec((0.0,)),  Vec((1.0,))]
        end
    end

    return NestedGrid1D(coarse_grid, fine_grid, coarse_dh, fine_dh, fine2coarse, child_ref_coords)
end

function _make_1d_dh(grid, order::Int)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, order}())
    close!(dh)
    return dh
end


#######################################################################
## Geometric prolongator assembly (1D)                               ##
#######################################################################

"""
    build_geometric_prolongator_1d(ng::NestedGrid1D)

Assemble the prolongation matrix P of size `(ndofs(ng.fine_dh) × ndofs(ng.coarse_dh))`
for a P1 geometric multigrid method on nested 1D grids.

The prolongation is computed via element-local L²-projection using
[`NestedGridTransferCellIterator`](@ref): for each fine cell the coarse basis functions are
evaluated at the fine quadrature points using the pre-computed reference coordinates
`child_ref_coords`.
"""
function build_geometric_prolongator_1d(ng::NestedGrid1D; qr_order::Int = 2)
    fine_dh   = ng.fine_dh
    coarse_dh = ng.coarse_dh
    fine_ndofs   = ndofs(fine_dh)
    coarse_ndofs = ndofs(coarse_dh)
    P = spzeros(fine_ndofs, coarse_ndofs)
    row_contrib = zeros(Int, fine_ndofs)

    # CellValues for fine and coarse spaces
    ip_fine   = Ferrite.getfieldinterpolation(fine_dh.subdofhandlers[1],   :u)
    ip_coarse = Ferrite.getfieldinterpolation(coarse_dh.subdofhandlers[1], :u)
    ip_geo    = Ferrite.geometric_interpolation(Line)
    qr        = QuadratureRule{RefLine}(qr_order)
    fine_cv   = CellValues(qr, ip_fine,   ip_geo)
    coarse_cv = CellValues(qr, ip_coarse, ip_geo)

    n_fine_basis   = getnbasefunctions(fine_cv)
    n_coarse_basis = getnbasefunctions(coarse_cv)
    Pe        = zeros(n_fine_basis, n_coarse_basis)
    Pe_buffer = zeros(n_fine_basis, n_coarse_basis)
    Me        = zeros(n_fine_basis, n_fine_basis)

    for tc in NestedGridTransferCellIterator(fine_dh, coarse_dh,
                                             ng.fine2coarse, ng.child_ref_coords)
        # Reinit fine CellValues at fine cell geometry
        reinit!(fine_cv, getcells(ng.fine_grid, tc.fine_cellid), get_fine_coordinates(tc))

        # For the coarse CellValues we need to evaluate coarse basis functions at the
        # physical positions of the fine quadrature points.  We do this by constructing
        # a custom CellValues reinit using the child reference coordinates mapped through
        # the coarse reference element.
        _reinit_coarse_cv_at_child!(coarse_cv, tc, ng.coarse_grid)

        # Element prolongator: P_e = M_e^{-1} * ∫ φ_fine ⊗ φ_coarse dΩ
        element_prolongator!(Pe, Me, fine_cv, coarse_cv, Pe_buffer)

        rdofs = getrowdofs(tc)
        cdofs = getcolumndofs(tc)
        for i in 1:n_fine_basis
            gi = rdofs[i]
            row_contrib[gi] += 1
            for j in 1:n_coarse_basis
                P[gi, cdofs[j]] += Pe[i, j]
            end
        end
    end
    normalize_rows!(P, row_contrib)
    return P
end

"""
    _reinit_coarse_cv_at_child!(coarse_cv, tc, coarse_grid)

Reinitialise `coarse_cv` so that its quadrature points coincide with those of the fine
child element.  The fine quadrature points are expressed in the coarse reference element
via `get_child_ref_coords(tc)` (a linear map for uniform bisection).

For the P1 / geometric-multigrid 1D case the fine cell's nodes span exactly one half of
the coarse reference element, so the quadrature rule on the fine cell maps to a sub-interval
of [-1,1] in the coarse reference frame.
"""
function _reinit_coarse_cv_at_child!(coarse_cv::CellValues, tc::NestedGridTransferCellCache, coarse_grid)
    child_nodes = get_child_ref_coords(tc)   # [ξ_left, ξ_right] in coarse reference coords
    # Build a custom quadrature rule at the mapped fine quadrature points
    fine_qr   = coarse_cv.qr  # same quadrature rule used for the fine cv
    ξ_left  = child_nodes[1][1]
    ξ_right = child_nodes[2][1]
    # Affine map: fine reference coord η ∈ [-1,1] → coarse reference coord ξ
    #   ξ = ξ_left + (ξ_right - ξ_left) / 2 * (η + 1)
    mapped_points  = [Vec(((ξ_left + ξ_right)/2 + (ξ_right - ξ_left)/2 * η[1],))
                      for η in fine_qr.points]
    scaled_weights = [(ξ_right - ξ_left)/2 * w for w in fine_qr.weights]
    mapped_qr = QuadratureRule{RefLine}(scaled_weights, mapped_points)
    # Temporarily build new CellValues with mapped quadrature (cheap for 1D)
    ip_coarse = coarse_cv.fun_values.ip
    ip_geo    = coarse_cv.geo_mapping.ip
    tmp_cv    = CellValues(mapped_qr, ip_coarse, ip_geo)
    reinit!(tmp_cv, getcells(coarse_grid, tc.coarse_cellid), get_coarse_coordinates(tc))
    # Copy evaluated values into coarse_cv's buffers
    # NOTE: This is a sketch; a production implementation would avoid the allocation by
    # pre-computing the coarse shape values at the mapped points during operator setup.
    copyto!(coarse_cv.fun_values.Nx,  tmp_cv.fun_values.Nx)
    copyto!(coarse_cv.fun_values.dNdx, tmp_cv.fun_values.dNdx)
    copyto!(coarse_cv.detJdV, tmp_cv.detJdV)
    return coarse_cv
end


#######################################################################
## Geometric multigrid V-cycle sketch                                ##
#######################################################################

"""
    gmultigrid(A, dh, ch, integrator; kwargs...)

Build a two-level geometric multigrid preconditioner/solver for the system `Ax = b`
assembled on a 1D uniform grid `dh`.

This is a **sketch** intended to demonstrate the use of `NestedGridTransferCellIterator`
and `build_geometric_prolongator_1d`.  It builds one coarse level by uniform bisection
and delegates the coarse-grid solve to an AMG coarse solver.

# Arguments
- `A`          – assembled fine-grid matrix
- `dh`         – fine-grid `DofHandler` (1D, scalar `:u`, P1)
- `ch`         – fine-grid `ConstraintHandler`
- `integrator` – an `AbstractBilinearIntegrator` (e.g. `DiffusionMultigrid(1.0)`)
                 used to assemble the coarse-grid matrix via `FerriteOperators`.

Returns a `MultiLevel` object compatible with `AlgebraicMultigrid._solve`.
"""
function gmultigrid(
        A::SparseMatrixCSC{T},
        ng::NestedGrid1D,
        integrator::AbstractBilinearIntegrator,
        coarse_solver_type = SmoothedAggregationCoarseSolver;
        presmoother  = GaussSeidel(),
        postsmoother = GaussSeidel(),
    ) where {T}

    # Build transfer operators
    P = build_geometric_prolongator_1d(ng)
    R = P'   # symmetric restriction

    # Assemble coarse-grid operator via FerriteOperators
    strategy  = SequentialAssemblyStrategy(SequentialCPUDevice())
    coarse_op = setup_operator(strategy, integrator, ng.coarse_dh)
    update_operator!(coarse_op, nothing)
    # Apply coarse Dirichlet BCs (build a ConstraintHandler matching the coarse grid)
    # NOTE: In a real implementation the user would supply coarse_ch.  Here we just use
    # the assembled matrix directly (BCs must have been applied to the fine problem).
    A_coarse = coarse_op.A

    # Wrap in AlgebraicMultigrid Level
    w = MultiLevelWorkspace(Val{1}, T)
    residual!(w, size(A, 1))
    coarse_x!(w, size(A_coarse, 1))
    coarse_b!(w, size(A_coarse, 1))

    cs = coarse_solver_type()(A_coarse)

    level = Level(A, P, R)
    return MultiLevel([level], A_coarse, cs, presmoother, postsmoother, w)
end
