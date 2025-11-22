## P_ij = M^{-1}_{ij} ∫ ϕ_i(x) ⋅ ϕc_j(x) dx (ϕc is the coarse basis function)
function _element_prolongator!(
    Pe::AbstractMatrix,
    Me::AbstractMatrix,
    fine_cv::AbstractCellValues,
    coarse_cv::AbstractCellValues,
)
    fill!(Pe, zero(eltype(Pe)))
    n_fine_basefuncs = getnbasefunctions(fine_cv)
    n_coarse_basefuncs = getnbasefunctions(coarse_cv)

    for q = 1:getnquadpoints(fine_cv)
        dΩ = getdetJdV(fine_cv, q)
        for i = 1:n_fine_basefuncs
            δu = shape_value(fine_cv, q, i)
            for j = 1:n_coarse_basefuncs
                u_c = shape_value(coarse_cv, q, j)
                Pe[i, j] += (δu ⋅ u_c) * dΩ
            end
        end
    end

    # Invert the mass matrix to get the prolongator
    _element_mass_matrix!(Me, fine_cv)

    # Check if matrix is singular and use pseudo-inverse if needed
    try
        lu_fact = lu!(Me)
        ldiv!(Pe, lu_fact, Pe)
    catch e
        if e isa SingularException
            # Use pseudo-inverse for singular mass matrices
            # This can happen at constrained DOFs
            Me_pinv = pinv(Me)
            Pe_temp = copy(Pe)
            mul!(Pe, Me_pinv, Pe_temp)
        else
            rethrow(e)
        end
    end

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

function build_prolongator(fine_fespace::FESpace, coarse_fespace::FESpace)
    fine_ndofs = ndofs(fine_fespace)
    coarse_ndofs = ndofs(coarse_fespace)
    P = spzeros(fine_ndofs, coarse_ndofs)
    row_contrib = zeros(Int, fine_ndofs)  # NEW: track contributions

    fine_nbasefuncs = getnbasefunctions(fine_fespace)
    coarse_nbasefuncs = getnbasefunctions(coarse_fespace)
    Pe = zeros(fine_nbasefuncs, coarse_nbasefuncs)
    Me = zeros(fine_nbasefuncs, fine_nbasefuncs)
    for cell in CellIterator(fine_fespace.dh)
        reinit!(fine_fespace.cv, cell)
        reinit!(coarse_fespace.cv, cell)
        _element_prolongator!(Pe, Me, fine_fespace.cv, coarse_fespace.cv)
        
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
    for i in 1:fine_ndofs
        if row_contrib[i] > 1
            P[i, :] ./= row_contrib[i]
        end
    end

    #dropzeros!(P) # we don't this, do we?
    return P
end
