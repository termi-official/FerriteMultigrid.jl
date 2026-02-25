## 1D poisson equation with Dirichlet boundary conditions ##
function poisson(N::Int64, p::Int, nqr::Int)
    sz = (N,)
    ∂Ω_f = grid -> union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right")
    )
    return _poisson(sz, p, nqr, Line, RefLine, ∂Ω_f)
end

## 2D poisson equation with Dirichlet boundary conditions ##
function poisson(sz::NTuple{2,Int64}, p::Int, nqr::Int)
    ∂Ω_f = grid -> union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right"),
        getfacetset(grid, "bottom"),
        getfacetset(grid, "top")
    )
    return _poisson(sz, p, nqr, Quadrilateral, RefQuadrilateral, ∂Ω_f)
end


function _poisson(sz::NTuple{N,Int}, p, nqr, celltype::Type{<:AbstractCell}, refshapetype::Type{<:AbstractRefShape}, ∂Ω_f) where {N}
    grid = generate_grid(celltype, sz)
    ip = Lagrange{refshapetype, p}() 
    qr = QuadratureRule{refshapetype}(nqr)  
    cellvalues = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    K = allocate_matrix(dh)

    ch = ConstraintHandler(dh)

    ∂Ω = ∂Ω_f(grid)

    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0.0)
    add!(ch, dbc)
    close!(ch)

    K, f = _assemble_global(cellvalues, K, dh)
    apply!(K, f, ch)

    return K, f, dh, ch
end

function _assemble_element!(Ke, fe, cellvalues)
    fill!(Ke, 0.0)
    fill!(fe, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q)
        for i in 1:n_basefuncs
            ∇δu = shape_gradient(cellvalues, q, i)
            δu = shape_value(cellvalues, q, i)
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q, j)
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
            fe[i] += δu * dΩ  # RHS = 1
        end
    end
    return Ke, fe
end

function _assemble_global(cellvalues, K, dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        _assemble_element!(Ke, fe, cellvalues)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end
