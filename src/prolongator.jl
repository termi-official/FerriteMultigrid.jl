## P_ij = M^{-1}_{ij} ∫ ϕ_i(x) ⋅ ϕc_j(x) dx (ϕc is the coarse basis function)
function element_prolongator!(
    Pe::AbstractMatrix,
    Me::AbstractMatrix,
    fine_cv::AbstractCellValues,
    coarse_cv::AbstractCellValues,
    Pe_buffer::AbstractMatrix,
)
    fill!(Pe_buffer, zero(eltype(Pe)))
    n_fine_basefuncs = getnbasefunctions(fine_cv)
    n_coarse_basefuncs = getnbasefunctions(coarse_cv)

    for q = 1:getnquadpoints(fine_cv)
        dΩ = getdetJdV(fine_cv, q)
        for i = 1:n_fine_basefuncs
            ϕᵢ = shape_value(fine_cv, q, i)
            for j = 1:n_coarse_basefuncs
                ϕⱼ = shape_value(coarse_cv, q, j)
                Pe_buffer[i, j] += (ϕᵢ ⋅ ϕⱼ) * dΩ
            end
        end
    end

    # Invert the mass matrix to get the prolongator
    _element_mass_matrix!(Me, fine_cv)
    lu_fact = lu!(Me)
    ldiv!(Pe, lu_fact, Pe_buffer)

    # _element_mass_matrix_lumped!(Me, fine_cv)
    # for i in 1:size(Me, 1)
    #     Me[i,i] = inv(Me[i,i])
    # end
    # mul!(Pe, Me, Pe_buffer)
    return drop_small_entries!(Pe)
end


function drop_small_entries!(A::AbstractMatrix, tol::Float64 = 1e-10)
    A[abs.(A) .< tol] .= 0.0
    return A
end

## M_{ij} = ∫ ϕ_i(x) ⋅ ϕ_j(x) dx
function _element_mass_matrix!(Me::AbstractMatrix, cv::AbstractCellValues)
    fill!(Me, zero(eltype(Me)))
    n_basefuncs = getnbasefunctions(cv)
    for q = 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q)
        for i = 1:n_basefuncs
            δu = shape_value(cv, q, i)
            for j = 1:n_basefuncs
                u = shape_value(cv, q, j)
                Me[i, j] += (δu ⋅ u) * dΩ
            end
        end
    end
    return Me
end

function _element_mass_matrix_lumped!(Me::AbstractMatrix, cv::AbstractCellValues)
    fill!(Me, zero(eltype(Me)))
    n_basefuncs = getnbasefunctions(cv)
    for q = 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q)
        for i = 1:n_basefuncs
            δu = shape_value(cv, q, i)
            for j = 1:n_basefuncs
                u = shape_value(cv, q, j)
                Me[i, i] += (δu ⋅ u) * dΩ
            end
        end
    end
    return Me
end

# TODO use FerriteOperators
function build_prolongator(fine_fespace::FESpace, coarse_fespace::FESpace)
    fine_ndofs = ndofs(fine_fespace)
    coarse_ndofs = ndofs(coarse_fespace)
    P = spzeros(fine_ndofs, coarse_ndofs)
    row_contrib = zeros(Int, fine_ndofs)  # NEW: track contributions

    fine_nbasefuncs = getnbasefunctions(fine_fespace)
    coarse_nbasefuncs = getnbasefunctions(coarse_fespace)
    Pe = zeros(fine_nbasefuncs, coarse_nbasefuncs)
    Pe_buffer = zeros(fine_nbasefuncs, coarse_nbasefuncs)
    Me = zeros(fine_nbasefuncs, fine_nbasefuncs)
    for cell in CellIterator(fine_fespace.dh)
        reinit!(fine_fespace.cv, cell)
        reinit!(coarse_fespace.cv, cell)
        element_prolongator!(Pe, Me, fine_fespace.cv, coarse_fespace.cv, Pe_buffer)

        fine_dofs = celldofs(cell)
        coarse_dofs = celldofs(coarse_fespace.dh, cell.cellid)

        for i = 1:fine_nbasefuncs
            global_i = fine_dofs[i]
            row_contrib[global_i] += 1  # NEW: track how often each fine DOF is used
            for j = 1:coarse_nbasefuncs
                global_j = coarse_dofs[j]
                P[global_i, global_j] += Pe[i, j]
            end
        end
    end
    # Normalize the rows of P by how many times they were visited
    # Iterate over CSC nonzeros directly — O(nnz) instead of O(nrows × ncols)
    @timeit_debug "row normalization" begin
        rows = rowvals(P)
        vals = nonzeros(P)
        for j in 1:size(P, 2)
            for idx in nzrange(P, j)
                row = rows[idx]
                if row_contrib[row] > 1
                    vals[idx] /= row_contrib[row]
                end
            end
        end
    end

    return P
end

function build_restriction(coarse_fespace, fine_fespace, P, is_sym)
    if !is_sym
        return build_prolongator(coarse_fespace, fine_fespace)
    else
        return P'
    end
end
