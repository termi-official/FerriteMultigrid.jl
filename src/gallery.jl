## 1D poisson equation with Dirichlet boundary conditions ##
function poisson(N::Int64, p::Vector{Int}, nqr::Int)
    sz = (N,)
    ∂Ω_f = grid -> union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right")
    )
    return poisson(sz, p, nqr, Line, RefLine, ∂Ω_f)
end

## 2D poisson equation with Dirichlet boundary conditions ##
function poisson(sz::NTuple{2,Int64}, p::Vector{Int}, nqr::Int)
    ∂Ω_f = grid -> union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right"),
        getfacetset(grid, "bottom"),
        getfacetset(grid, "top")
    )
    return poisson(sz, p, nqr, Quadrilateral, RefQuadrilateral, ∂Ω_f)
end

function assemble_poisson(dh::DofHandler, ch::ConstraintHandler, nqr::Int)
    K   = allocate_matrix(dh)
    f   = zeros(ndofs(dh))
    asm = start_assemble(K, f)
    grid = Ferrite.get_grid(dh)
    for sdh in dh.subdofhandlers
        ip  = Ferrite.getfieldinterpolation(sdh, :u)
        qr  = QuadratureRule{Ferrite.getrefshape(getcells(grid, first(sdh.cellset)))}(nqr)
        cv  = CellValues(qr, ip)
        n   = getnbasefunctions(cv)
        Ke  = zeros(n, n)
        fe  = zeros(n)
        for cell in CellIterator(sdh)
            reinit!(cv, cell)
            assemble_poisson_element!(Ke, fe, cv)
            assemble!(asm, celldofs(cell), Ke, fe)
        end
    end
    apply!(K, f, ch)
    return K, f
end

function poisson(sz::NTuple{N,Int}, ps::Vector, nqr::Int, celltype::Type{<:AbstractCell}, refshapetype::Type{<:AbstractRefShape}, ∂Ω_f) where {N}
    grid = generate_grid(celltype, sz)
    dhh = DofHandlerHierarchy(grid, length(ps))
    for (i, p) in enumerate(ps)
        add!(dhh[i], :u, Lagrange{refshapetype, p}())
    end
    close!(dhh)

    chh = ConstraintHandlerHierarchy(dhh)
    add!(chh, dh -> Dirichlet(:u, ∂Ω_f(dh.grid), (x, t) -> 0.0))
    close!(chh)

    K, f = assemble_poisson(dhh[end], chh[end], nqr)

    return K, f, dhh, chh
end

function assemble_poisson_element!(Ke, fe, cellvalues)
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
