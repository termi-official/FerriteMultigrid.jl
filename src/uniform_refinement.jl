
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
| `Wedge`         | 8        |

Mixed-cell grids (e.g. `Grid{3, Union{Tetrahedron,Wedge}, T}`) are also supported;
each cell is refined according to its own type and boundary sets are propagated globally.

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
    fine_cellsets  = _propagate_cellsets(coarse_grid, fine2coarse)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets,
                     cellsets = fine_cellsets)
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
    fine_cellsets  = _propagate_cellsets(coarse_grid, fine2coarse)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets,
                     cellsets = fine_cellsets)
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
    fine_cellsets  = _propagate_cellsets(coarse_grid, fine2coarse)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets,
                     cellsets = fine_cellsets)
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
    fine_cellsets  = _propagate_cellsets(coarse_grid, fine2coarse)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets,
                     cellsets = fine_cellsets)
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
    fine_cellsets  = _propagate_cellsets(coarse_grid, fine2coarse)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets,
                     cellsets = fine_cellsets)
    return fine_grid, fine2coarse, child_ref_coords
end


# ──────────────────────────── Wedge (3D) ─────────────────────────────────────
# Each wedge → 8 children (tensor product: 4 triangle halves × 2 z-halves).
# RefPrism nodes: v1=(0,0,0), v2=(1,0,0), v3=(0,1,0), v4=(0,0,1), v5=(1,0,1), v6=(0,1,1)
# Edges: e12,e23,e13 (bottom), e45,e56,e46 (top), e14,e25,e36 (vertical)
# Quad-face centres: fc12 (v1,v2,v5,v4), fc13 (v1,v3,v4,v6), fc23 (v2,v3,v6,v5)

function uniform_refinement(coarse_grid::Grid{3, Wedge, T}) where T
    coarse_nodes = getnodes(coarse_grid)
    coarse_cells = getcells(coarse_grid)
    n_coarse     = length(coarse_cells)

    fine_nodes   = collect(coarse_nodes)
    edge_mid     = Dict{NTuple{2,Int}, Int}()
    face_ctr     = Dict{NTuple{4,Int}, Int}()
    node_parents = Dict{Int, Vector{Int}}()

    fine_cells       = Vector{Wedge}(undef, 8 * n_coarse)
    fine2coarse      = Vector{Int}(undef,  8 * n_coarse)
    child_ref_coords = Vector{Vector{Vec{3,T}}}(undef, 8 * n_coarse)

    # Reference coordinates of the 8 child wedges in the parent's RefPrism space.
    # Node ordering per child: (bottom-tri-n1, bottom-tri-n2, bottom-tri-n3,
    #                            top-tri-n1,   top-tri-n2,   top-tri-n3)
    # Lower half (z ∈ [0,0.5]) — 4 children mirroring the 4 triangle sub-cells
    # Upper half (z ∈ [0.5,1]) — 4 children on the same lateral sub-triangles
    crc = [
        # c1: lower, corner at v1
        [Vec((0.0,0.0,0.0)), Vec((0.5,0.0,0.0)), Vec((0.0,0.5,0.0)),
         Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5))],
        # c2: lower, corner at v2
        [Vec((0.5,0.0,0.0)), Vec((1.0,0.0,0.0)), Vec((0.5,0.5,0.0)),
         Vec((0.5,0.0,0.5)), Vec((1.0,0.0,0.5)), Vec((0.5,0.5,0.5))],
        # c3: lower, corner at v3
        [Vec((0.0,0.5,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,1.0,0.0)),
         Vec((0.0,0.5,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,1.0,0.5))],
        # c4: lower, central
        [Vec((0.5,0.0,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,0.5,0.0)),
         Vec((0.5,0.0,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,0.5,0.5))],
        # c5: upper, corner at v4
        [Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5)),
         Vec((0.0,0.0,1.0)), Vec((0.5,0.0,1.0)), Vec((0.0,0.5,1.0))],
        # c6: upper, corner at v5
        [Vec((0.5,0.0,0.5)), Vec((1.0,0.0,0.5)), Vec((0.5,0.5,0.5)),
         Vec((0.5,0.0,1.0)), Vec((1.0,0.0,1.0)), Vec((0.5,0.5,1.0))],
        # c7: upper, corner at v6
        [Vec((0.0,0.5,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,1.0,0.5)),
         Vec((0.0,0.5,1.0)), Vec((0.5,0.5,1.0)), Vec((0.0,1.0,1.0))],
        # c8: upper, central
        [Vec((0.5,0.0,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,0.5,0.5)),
         Vec((0.5,0.0,1.0)), Vec((0.5,0.5,1.0)), Vec((0.0,0.5,1.0))],
    ]

    for (c, cell) in enumerate(coarse_cells)
        n1,n2,n3,n4,n5,n6 = cell.nodes
        # Bottom-triangle edge midpoints
        e12  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
        e23  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
        e13  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n3, coarse_nodes)
        # Top-triangle edge midpoints
        e45  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n5, coarse_nodes)
        e56  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n5, n6, coarse_nodes)
        e46  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n6, coarse_nodes)
        # Vertical edge midpoints
        e14  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n4, coarse_nodes)
        e25  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n5, coarse_nodes)
        e36  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n6, coarse_nodes)
        # Quad-face centres (shared with adjacent cells)
        fc12 = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n2,n5,n4), coarse_nodes)
        fc13 = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n3,n4,n6), coarse_nodes)
        fc23 = _face_center!(fine_nodes, face_ctr, node_parents, (n2,n3,n6,n5), coarse_nodes)
        k = 8 * (c - 1)
        # Lower half
        fine_cells[k+1] = Wedge((n1,  e12,  e13,  e14,  fc12, fc13)); fine2coarse[k+1] = c; child_ref_coords[k+1] = crc[1]
        fine_cells[k+2] = Wedge((e12, n2,   e23,  fc12, e25,  fc23)); fine2coarse[k+2] = c; child_ref_coords[k+2] = crc[2]
        fine_cells[k+3] = Wedge((e13, e23,  n3,   fc13, fc23, e36 )); fine2coarse[k+3] = c; child_ref_coords[k+3] = crc[3]
        fine_cells[k+4] = Wedge((e12, e23,  e13,  fc12, fc23, fc13)); fine2coarse[k+4] = c; child_ref_coords[k+4] = crc[4]
        # Upper half
        fine_cells[k+5] = Wedge((e14,  fc12, fc13, n4,  e45,  e46 )); fine2coarse[k+5] = c; child_ref_coords[k+5] = crc[5]
        fine_cells[k+6] = Wedge((fc12, e25,  fc23, e45, n5,   e56 )); fine2coarse[k+6] = c; child_ref_coords[k+6] = crc[6]
        fine_cells[k+7] = Wedge((fc13, fc23, e36,  e46, e56,  n6  )); fine2coarse[k+7] = c; child_ref_coords[k+7] = crc[7]
        fine_cells[k+8] = Wedge((fc12, fc23, fc13, e45, e56,  e46 )); fine2coarse[k+8] = c; child_ref_coords[k+8] = crc[8]
    end

    fine_facetsets = _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    fine_nodesets  = _propagate_nodesets(coarse_grid, node_parents)
    fine_cellsets  = _propagate_cellsets(coarse_grid, fine2coarse)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets,
                     cellsets = fine_cellsets)
    return fine_grid, fine2coarse, child_ref_coords
end

#######################################################################
## Mixed-mesh support: per-cell-type helpers + general fallback      ##
#######################################################################

# Each `_add_fine_cells!` method refines one coarse cell of a specific type and
# appends the resulting children (cells, parent-map, child ref-coords) to the
# shared output vectors.  The same shared `edge_mid` / `face_ctr` / `node_parents`
# dicts are used so midpoint nodes are properly reused across different cell types.

function _add_fine_cells!(fine_nodes, edge_mid, _, node_parents, coarse_nodes,
                           fine_cells, fine2coarse, child_ref_coords, c, cell::Line)
    n1, n2 = cell.nodes
    m = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
    push!(fine_cells, Line((n1, m)), Line((m, n2)))
    push!(fine2coarse, c, c)
    push!(child_ref_coords,
          [Vec((-1.0,)), Vec(( 0.0,))],
          [Vec(( 0.0,)), Vec(( 1.0,))])
end

function _add_fine_cells!(fine_nodes, edge_mid, _, node_parents, coarse_nodes,
                           fine_cells, fine2coarse, child_ref_coords, c, cell::Triangle)
    n1, n2, n3 = cell.nodes
    e12 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
    e23 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
    e13 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n3, coarse_nodes)
    push!(fine_cells,
          Triangle((n1, e12, e13)), Triangle((e12, n2, e23)),
          Triangle((e13, e23, n3)), Triangle((e12, e23, e13)))
    for _ in 1:4; push!(fine2coarse, c); end
    push!(child_ref_coords,
          [Vec((0.0,0.0)), Vec((0.5,0.0)), Vec((0.0,0.5))],
          [Vec((0.5,0.0)), Vec((1.0,0.0)), Vec((0.5,0.5))],
          [Vec((0.0,0.5)), Vec((0.5,0.5)), Vec((0.0,1.0))],
          [Vec((0.5,0.0)), Vec((0.5,0.5)), Vec((0.0,0.5))])
end

function _add_fine_cells!(fine_nodes, edge_mid, face_ctr, node_parents, coarse_nodes,
                           fine_cells, fine2coarse, child_ref_coords, c, cell::Quadrilateral)
    n1, n2, n3, n4 = cell.nodes
    e12 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
    e23 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
    e34 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n4, coarse_nodes)
    e41 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n1, coarse_nodes)
    fc  = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n2,n3,n4), coarse_nodes)
    push!(fine_cells,
          Quadrilateral((n1, e12, fc, e41)),  Quadrilateral((e12, n2, e23, fc)),
          Quadrilateral((fc, e23, n3, e34)),  Quadrilateral((e41, fc, e34, n4)))
    for _ in 1:4; push!(fine2coarse, c); end
    push!(child_ref_coords,
          [Vec((-1.0,-1.0)), Vec(( 0.0,-1.0)), Vec(( 0.0,0.0)), Vec((-1.0,0.0))],
          [Vec(( 0.0,-1.0)), Vec(( 1.0,-1.0)), Vec(( 1.0,0.0)), Vec(( 0.0,0.0))],
          [Vec(( 0.0, 0.0)), Vec(( 1.0, 0.0)), Vec(( 1.0,1.0)), Vec(( 0.0,1.0))],
          [Vec((-1.0, 0.0)), Vec(( 0.0, 0.0)), Vec(( 0.0,1.0)), Vec((-1.0,1.0))])
end

function _add_fine_cells!(fine_nodes, edge_mid, _, node_parents, coarse_nodes,
                           fine_cells, fine2coarse, child_ref_coords, c, cell::Tetrahedron)
    n1, n2, n3, n4 = cell.nodes
    e12 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
    e23 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
    e13 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n3, coarse_nodes)
    e14 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n4, coarse_nodes)
    e24 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n4, coarse_nodes)
    e34 = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n4, coarse_nodes)
    push!(fine_cells,
          Tetrahedron((n1,  e12, e13, e14)), Tetrahedron((e12, n2,  e23, e24)),
          Tetrahedron((e13, e23, n3,  e34)), Tetrahedron((e14, e24, e34, n4 )),
          Tetrahedron((e13, e12, e23, e24)), Tetrahedron((e13, e23, e34, e24)),
          Tetrahedron((e13, e14, e24, e34)), Tetrahedron((e12, e13, e14, e24)))
    for _ in 1:8; push!(fine2coarse, c); end
    push!(child_ref_coords,
          [Vec((0.0,0.0,0.0)), Vec((0.5,0.0,0.0)), Vec((0.0,0.5,0.0)), Vec((0.0,0.0,0.5))],
          [Vec((0.5,0.0,0.0)), Vec((1.0,0.0,0.0)), Vec((0.5,0.5,0.0)), Vec((0.5,0.0,0.5))],
          [Vec((0.0,0.5,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,1.0,0.0)), Vec((0.0,0.5,0.5))],
          [Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5)), Vec((0.0,0.0,1.0))],
          [Vec((0.0,0.5,0.0)), Vec((0.5,0.0,0.0)), Vec((0.5,0.5,0.0)), Vec((0.5,0.0,0.5))],
          [Vec((0.0,0.5,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,0.5,0.5)), Vec((0.5,0.0,0.5))],
          [Vec((0.0,0.5,0.0)), Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5))],
          [Vec((0.5,0.0,0.0)), Vec((0.0,0.5,0.0)), Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5))])
end

function _add_fine_cells!(fine_nodes, edge_mid, face_ctr, node_parents, coarse_nodes,
                           fine_cells, fine2coarse, child_ref_coords, c, cell::Hexahedron)
    n1,n2,n3,n4,n5,n6,n7,n8 = cell.nodes
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
    fc_bot = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n2,n3,n4), coarse_nodes)
    fc_frt = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n2,n6,n5), coarse_nodes)
    fc_rht = _face_center!(fine_nodes, face_ctr, node_parents, (n2,n3,n7,n6), coarse_nodes)
    fc_bck = _face_center!(fine_nodes, face_ctr, node_parents, (n3,n4,n8,n7), coarse_nodes)
    fc_lft = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n5,n8,n4), coarse_nodes)
    fc_top = _face_center!(fine_nodes, face_ctr, node_parents, (n5,n6,n7,n8), coarse_nodes)
    body   = _body_center!(fine_nodes, node_parents, (n1,n2,n3,n4,n5,n6,n7,n8), coarse_nodes)
    push!(fine_cells,
          Hexahedron((n1,   me12,  fc_bot, me41,  me15,  fc_frt, body,   fc_lft)),
          Hexahedron((me12, n2,    me23,   fc_bot, fc_frt,me26,  fc_rht, body  )),
          Hexahedron((fc_bot,me23, n3,     me34,  body,  fc_rht, me37,  fc_bck)),
          Hexahedron((me41, fc_bot,me34,   n4,    fc_lft,body,  fc_bck, me48  )),
          Hexahedron((me15, fc_frt,body,   fc_lft,n5,    me56,  fc_top, me85  )),
          Hexahedron((fc_frt,me26, fc_rht, body,  me56,  n6,    me67,  fc_top)),
          Hexahedron((body, fc_rht,me37,   fc_bck,fc_top,me67,  n7,    me78  )),
          Hexahedron((fc_lft,body, fc_bck, me48,  me85,  fc_top,me78,  n8    )))
    for _ in 1:8; push!(fine2coarse, c); end
    push!(child_ref_coords,
          [Vec((-1.0,-1.0,-1.0)), Vec((0.0,-1.0,-1.0)), Vec((0.0,0.0,-1.0)), Vec((-1.0,0.0,-1.0)),
           Vec((-1.0,-1.0, 0.0)), Vec((0.0,-1.0, 0.0)), Vec((0.0,0.0, 0.0)), Vec((-1.0,0.0, 0.0))],
          [Vec((0.0,-1.0,-1.0)), Vec((1.0,-1.0,-1.0)), Vec((1.0,0.0,-1.0)), Vec((0.0,0.0,-1.0)),
           Vec((0.0,-1.0, 0.0)), Vec((1.0,-1.0, 0.0)), Vec((1.0,0.0, 0.0)), Vec((0.0,0.0, 0.0))],
          [Vec((0.0,0.0,-1.0)), Vec((1.0,0.0,-1.0)), Vec((1.0,1.0,-1.0)), Vec((0.0,1.0,-1.0)),
           Vec((0.0,0.0, 0.0)), Vec((1.0,0.0, 0.0)), Vec((1.0,1.0, 0.0)), Vec((0.0,1.0, 0.0))],
          [Vec((-1.0,0.0,-1.0)), Vec((0.0,0.0,-1.0)), Vec((0.0,1.0,-1.0)), Vec((-1.0,1.0,-1.0)),
           Vec((-1.0,0.0, 0.0)), Vec((0.0,0.0, 0.0)), Vec((0.0,1.0, 0.0)), Vec((-1.0,1.0, 0.0))],
          [Vec((-1.0,-1.0,0.0)), Vec((0.0,-1.0,0.0)), Vec((0.0,0.0,0.0)), Vec((-1.0,0.0,0.0)),
           Vec((-1.0,-1.0,1.0)), Vec((0.0,-1.0,1.0)), Vec((0.0,0.0,1.0)), Vec((-1.0,0.0,1.0))],
          [Vec((0.0,-1.0,0.0)), Vec((1.0,-1.0,0.0)), Vec((1.0,0.0,0.0)), Vec((0.0,0.0,0.0)),
           Vec((0.0,-1.0,1.0)), Vec((1.0,-1.0,1.0)), Vec((1.0,0.0,1.0)), Vec((0.0,0.0,1.0))],
          [Vec((0.0,0.0,0.0)), Vec((1.0,0.0,0.0)), Vec((1.0,1.0,0.0)), Vec((0.0,1.0,0.0)),
           Vec((0.0,0.0,1.0)), Vec((1.0,0.0,1.0)), Vec((1.0,1.0,1.0)), Vec((0.0,1.0,1.0))],
          [Vec((-1.0,0.0,0.0)), Vec((0.0,0.0,0.0)), Vec((0.0,1.0,0.0)), Vec((-1.0,1.0,0.0)),
           Vec((-1.0,0.0,1.0)), Vec((0.0,0.0,1.0)), Vec((0.0,1.0,1.0)), Vec((-1.0,1.0,1.0))])
end

function _add_fine_cells!(fine_nodes, edge_mid, face_ctr, node_parents, coarse_nodes,
                           fine_cells, fine2coarse, child_ref_coords, c, cell::Wedge)
    n1,n2,n3,n4,n5,n6 = cell.nodes
    e12  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n2, coarse_nodes)
    e23  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n3, coarse_nodes)
    e13  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n3, coarse_nodes)
    e45  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n5, coarse_nodes)
    e56  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n5, n6, coarse_nodes)
    e46  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n4, n6, coarse_nodes)
    e14  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n1, n4, coarse_nodes)
    e25  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n2, n5, coarse_nodes)
    e36  = _edge_midpoint!(fine_nodes, edge_mid, node_parents, n3, n6, coarse_nodes)
    fc12 = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n2,n5,n4), coarse_nodes)
    fc13 = _face_center!(fine_nodes, face_ctr, node_parents, (n1,n3,n4,n6), coarse_nodes)
    fc23 = _face_center!(fine_nodes, face_ctr, node_parents, (n2,n3,n6,n5), coarse_nodes)
    push!(fine_cells,
          Wedge((n1,  e12,  e13,  e14,  fc12, fc13)),
          Wedge((e12, n2,   e23,  fc12, e25,  fc23)),
          Wedge((e13, e23,  n3,   fc13, fc23, e36 )),
          Wedge((e12, e23,  e13,  fc12, fc23, fc13)),
          Wedge((e14,  fc12, fc13, n4,  e45,  e46 )),
          Wedge((fc12, e25,  fc23, e45, n5,   e56 )),
          Wedge((fc13, fc23, e36,  e46, e56,  n6  )),
          Wedge((fc12, fc23, fc13, e45, e56,  e46 )))
    for _ in 1:8; push!(fine2coarse, c); end
    push!(child_ref_coords,
          [Vec((0.0,0.0,0.0)), Vec((0.5,0.0,0.0)), Vec((0.0,0.5,0.0)),
           Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5))],
          [Vec((0.5,0.0,0.0)), Vec((1.0,0.0,0.0)), Vec((0.5,0.5,0.0)),
           Vec((0.5,0.0,0.5)), Vec((1.0,0.0,0.5)), Vec((0.5,0.5,0.5))],
          [Vec((0.0,0.5,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,1.0,0.0)),
           Vec((0.0,0.5,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,1.0,0.5))],
          [Vec((0.5,0.0,0.0)), Vec((0.5,0.5,0.0)), Vec((0.0,0.5,0.0)),
           Vec((0.5,0.0,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,0.5,0.5))],
          [Vec((0.0,0.0,0.5)), Vec((0.5,0.0,0.5)), Vec((0.0,0.5,0.5)),
           Vec((0.0,0.0,1.0)), Vec((0.5,0.0,1.0)), Vec((0.0,0.5,1.0))],
          [Vec((0.5,0.0,0.5)), Vec((1.0,0.0,0.5)), Vec((0.5,0.5,0.5)),
           Vec((0.5,0.0,1.0)), Vec((1.0,0.0,1.0)), Vec((0.5,0.5,1.0))],
          [Vec((0.0,0.5,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,1.0,0.5)),
           Vec((0.0,0.5,1.0)), Vec((0.5,0.5,1.0)), Vec((0.0,1.0,1.0))],
          [Vec((0.5,0.0,0.5)), Vec((0.5,0.5,0.5)), Vec((0.0,0.5,0.5)),
           Vec((0.5,0.0,1.0)), Vec((0.5,0.5,1.0)), Vec((0.0,0.5,1.0))])
end

# ────────────────── General fallback for mixed / unknown grids ────────────────
# Dispatches on each coarse cell's runtime type via `_add_fine_cells!`.
# Handles any Grid{dim,C,T} not already matched by a specific method above,
# including Union cell types such as Grid{3, Union{Tetrahedron,Wedge}, T}.

function uniform_refinement(coarse_grid::Grid{dim, C, T}) where {dim, C, T}
    coarse_nodes = getnodes(coarse_grid)
    coarse_cells = getcells(coarse_grid)

    fine_nodes   = collect(coarse_nodes)
    edge_mid     = Dict{NTuple{2,Int}, Int}()
    face_ctr     = Dict{NTuple{4,Int}, Int}()
    node_parents = Dict{Int, Vector{Int}}()

    fine_cells       = C[]
    fine2coarse      = Int[]
    child_ref_coords = Vector{Vector{Vec{dim,T}}}()

    for (c, cell) in enumerate(coarse_cells)
        _add_fine_cells!(fine_nodes, edge_mid, face_ctr, node_parents, coarse_nodes,
                         fine_cells, fine2coarse, child_ref_coords, c, cell)
    end

    fine_facetsets = _propagate_facetsets(coarse_grid, fine_cells, node_parents)
    fine_nodesets  = _propagate_nodesets(coarse_grid, node_parents)
    fine_cellsets  = _propagate_cellsets(coarse_grid, fine2coarse)
    fine_grid = Grid(fine_cells, fine_nodes;
                     facetsets = fine_facetsets, nodesets = fine_nodesets,
                     cellsets = fine_cellsets)
    return fine_grid, fine2coarse, child_ref_coords
end

#######################################################################
## Boundary-set propagation                                          ##
#######################################################################

## Propagate cellsets: each fine cell belongs to the same named set as its parent coarse cell.
function _propagate_cellsets(coarse_grid, fine2coarse)
    coarse_cellsets = Ferrite.getcellsets(coarse_grid)
    fine_cellsets = Dict{String, Set{Int}}()
    for (name, coarse_cs) in coarse_cellsets
        fine_cs = Set{Int}()
        for (fine_id, coarse_id) in enumerate(fine2coarse)
            coarse_id ∈ coarse_cs && push!(fine_cs, fine_id)
        end
        fine_cellsets[name] = fine_cs
    end
    return fine_cellsets
end


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
