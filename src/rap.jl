"""
    rap_threaded(R, A, P) -> SparseMatrixCSC

Fused, multi-threaded triple product `C = R * A * P` for sparse matrices.

Avoids forming the large intermediate `A * P`.  For each coarse column `j`
of `P` the algorithm computes `w = A * P[:,j]` (dense scatter into a boolean-
flagged workspace) and then `c = R * w` (sparse scatter).  Both phases track
touched indices explicitly so cleanup is O(nnz) rather than O(n_fine) + O(n_coarse).
Columns of `C` are independent so the outer loop is parallelised with `Threads.@threads`.

Compared to Julia's default `R * (A * P)`, this saves:
- one large sparse-matrix allocation (the intermediate `A * P`)
- the repeated symbolic analysis that goes with it
- O(n_fine + n_coarse) scan work per column (replaced by O(nnz) tracked cleanup)

Use this for Galerkin coarsening `R * A * P` where fine DOFs number in the
tens to hundreds of thousands.

# Arguments
- `R` – restriction  (m × n), `SparseMatrixCSC`
- `A` – fine operator (n × n), `SparseMatrixCSC`
- `P` – prolongation  (n × k), `SparseMatrixCSC`

# Returns
`SparseMatrixCSC{Tv, Ti}` of size `m × k`.
"""
function rap_threaded(
    R::SparseMatrixCSC{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    P::SparseMatrixCSC{Tv,Ti},
) where {Tv,Ti}
    mR, nR = size(R)
    mA, nA = size(A)
    mP, nP = size(P)
    nR == mA || throw(DimensionMismatch("R has $nR columns but A has $mA rows"))
    nA == mP || throw(DimensionMismatch("A has $nA columns but P has $mP rows"))

    # Per-column result buffers (one pair per coarse column j).
    col_rows = Vector{Vector{Ti}}(undef, nP)
    col_vals = Vector{Vector{Tv}}(undef, nP)

    nt = Threads.maxthreadid()

    # Per-thread dense workspaces to avoid false sharing and inner-loop allocation.
    # w      accumulates (A * P[:,j]) as a dense vector of length mA
    # c      accumulates (R * w)       as a dense vector of length mR
    # *_nz   explicit lists of touched indices — avoids O(n) full-array scans
    ws     = [zeros(Tv, mA) for _ in 1:nt]
    wflags = [falses(mA)    for _ in 1:nt]
    w_nzs  = [Ti[]          for _ in 1:nt]
    cs_ws  = [zeros(Tv, mR) for _ in 1:nt]
    cflags = [falses(mR)    for _ in 1:nt]
    c_nzs  = [Ti[]          for _ in 1:nt]

    # Pre-size lists to avoid repeated growths for typical FEM sparsity patterns.
    for t in 1:nt
        sizehint!(w_nzs[t], 512)
        sizehint!(c_nzs[t], 512)
    end

    Threads.@threads :static for j in 1:nP
        tid    = Threads.threadid()
        w      = ws[tid]
        w_flag = wflags[tid]
        w_nz   = w_nzs[tid]
        c      = cs_ws[tid]
        c_flag = cflags[tid]
        c_nz   = c_nzs[tid]

        # Phase 1: w = A * P[:,j]  (scatter A's columns weighted by P[:,j])
        @inbounds for pp in nzrange(P, j)
            p_val = nonzeros(P)[pp]
            k     = rowvals(P)[pp]
            for ap in nzrange(A, k)
                i = rowvals(A)[ap]
                v = nonzeros(A)[ap] * p_val
                if !w_flag[i]
                    w_flag[i] = true
                    w[i] = v
                    push!(w_nz, i)
                else
                    w[i] += v
                end
            end
        end

        # Phase 2: c = R * w  (only visit tracked nonzero entries of w)
        @inbounds for r in w_nz
            wr        = w[r]
            w[r]      = zero(Tv)
            w_flag[r] = false
            for rp in nzrange(R, r)
                i = rowvals(R)[rp]
                v = nonzeros(R)[rp] * wr
                if !c_flag[i]
                    c_flag[i] = true
                    c[i] = v
                    push!(c_nz, i)
                else
                    c[i] += v
                end
            end
        end
        empty!(w_nz)

        # Phase 3: gather into a sorted per-column buffer (sort c_nz for ascending row indices).
        nc = length(c_nz)
        sort!(c_nz)
        rows_j = Vector{Ti}(undef, nc)
        vals_j = Vector{Tv}(undef, nc)
        @inbounds for (pos, i) in enumerate(c_nz)
            rows_j[pos] = i
            vals_j[pos] = c[i]
            c[i]      = zero(Tv)
            c_flag[i] = false
        end
        empty!(c_nz)

        col_rows[j] = rows_j
        col_vals[j] = vals_j
    end

    # Serial CSC assembly from per-column buffers.
    nnzC = sum(length, col_rows)
    colptrC = Vector{Ti}(undef, nP + 1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC  = Vector{Tv}(undef, nnzC)

    ip = Ti(1)
    @inbounds for j in 1:nP
        colptrC[j] = ip
        rj = col_rows[j]
        vj = col_vals[j]
        for p in eachindex(rj)
            rowvalC[ip] = rj[p]
            nzvalC[ip]  = vj[p]
            ip += 1
        end
    end
    colptrC[nP + 1] = ip

    return SparseMatrixCSC(mR, nP, colptrC, rowvalC, nzvalC)
end

# Dispatch for the symmetric Galerkin case where R = P' is an Adjoint.
# Materialises once here so the inner loop always sees a concrete SparseMatrixCSC.
function rap_threaded(
    R::Adjoint{Tv, <:SparseMatrixCSC{Tv,Ti}},
    A::SparseMatrixCSC{Tv,Ti},
    P::SparseMatrixCSC{Tv,Ti},
) where {Tv,Ti}
    return rap_threaded(sparse(R), A, P)
end
