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

For repeated RAP computations with fixed R, A-sparsity, and P, prefer
[`rap_symbolic`](@ref) + [`rap_numeric!`](@ref) to avoid all allocation on
subsequent calls.

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
        # QuickSort is allocation-free (O(log n) stack only) unlike the default MergeSort.
        nc = length(c_nz)
        sort!(c_nz; alg = Base.Sort.QuickSort)
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

# ---------------------------------------------------------------------------
# Pre-allocated (symbolic + numeric) variant
# ---------------------------------------------------------------------------

"""
    RAPWorkspace{Tv,Ti}

Pre-allocated workspace for repeated `C = R * A * P` computations where R and P
are fixed (mesh geometry) but A changes across Newton iterations.

Built by [`rap_symbolic`](@ref); used by [`rap_numeric!`](@ref).

# Fields
- `R`, `P` – materialized restriction / prolongation matrices
- `C`       – output matrix; `colptr` and `rowval` are fixed; `nzval` is filled in-place
- remaining fields – per-thread accumulator buffers, pre-sized and reused
"""
struct RAPWorkspace{Tv, Ti}
    R::SparseMatrixCSC{Tv, Ti}
    P::SparseMatrixCSC{Tv, Ti}
    C::SparseMatrixCSC{Tv, Ti}
    ws::Vector{Vector{Tv}}
    wflags::Vector{BitVector}
    w_nzs::Vector{Vector{Ti}}
    cs_ws::Vector{Vector{Tv}}
    cflags::Vector{BitVector}
    c_nzs::Vector{Vector{Ti}}
end

"""
    rap_symbolic(R, A, P) -> RAPWorkspace

Symbolic phase of the fused triple product `C = R * A * P`.

Runs [`rap_threaded`](@ref) once to determine the sparsity pattern of `C`, then
pre-allocates the CSC output matrix and all per-thread scratch buffers.  The
returned [`RAPWorkspace`](@ref) can be passed to [`rap_numeric!`](@ref) for
subsequent allocation-free numeric computations.

The sparsity of `C` depends only on the *structure* of R, A, P — not their values.
In FEM, A's sparsity is determined by the mesh and never changes across Newton
iterations, so this function needs to be called only once.
"""
function rap_symbolic(
    R::SparseMatrixCSC{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    P::SparseMatrixCSC{Tv,Ti},
) where {Tv,Ti}
    C  = rap_threaded(R, A, P)
    mA = size(A, 1)
    mR = size(R, 1)
    nt = Threads.maxthreadid()

    # Size hints from the actual C sparsity: c_nz holds at most max_col_C entries per column,
    # and w_nz holds at most max_col_P * max_col_A entries (capped at mA for safety).
    max_col_C = maximum(diff(C.colptr); init = 512)
    max_col_P = maximum(diff(P.colptr); init = 64)
    max_col_A = maximum(diff(A.colptr); init = 64)
    w_hint    = min(mA, max_col_P * max_col_A)

    ws     = [zeros(Tv, mA) for _ in 1:nt]
    wflags = [falses(mA)    for _ in 1:nt]
    w_nzs  = [Ti[]          for _ in 1:nt]
    cs_ws  = [zeros(Tv, mR) for _ in 1:nt]
    cflags = [falses(mR)    for _ in 1:nt]
    c_nzs  = [Ti[]          for _ in 1:nt]
    for t in 1:nt
        sizehint!(w_nzs[t], w_hint)
        sizehint!(c_nzs[t], max_col_C)
    end

    return RAPWorkspace(R, P, C, ws, wflags, w_nzs, cs_ws, cflags, c_nzs)
end

function rap_symbolic(
    R::Adjoint{Tv, <:SparseMatrixCSC{Tv,Ti}},
    A::SparseMatrixCSC{Tv,Ti},
    P::SparseMatrixCSC{Tv,Ti},
) where {Tv,Ti}
    return rap_symbolic(sparse(R), A, P)
end

"""
    rap_numeric!(ws::RAPWorkspace, A) -> SparseMatrixCSC

Numeric phase of the fused triple product `C = R * A * P` using a pre-built
[`RAPWorkspace`](@ref).

Fills `ws.C.nzval` in-place with zero heap allocations.  `A` must have the same
nonzero structure as when [`rap_symbolic`](@ref) built the workspace (guaranteed
for FEM stiffness matrices across Newton iterations).  Columns of C are written in
parallel; each column's row range in `nzval` is disjoint so there is no data race.

Returns `ws.C`.
"""
function rap_numeric!(
    rws::RAPWorkspace{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
) where {Tv,Ti}
    R, P, C = rws.R, rws.P, rws.C
    nP = size(P, 2)

    Threads.@threads :static for j in 1:nP
        tid    = Threads.threadid()
        w      = rws.ws[tid]
        w_flag = rws.wflags[tid]
        w_nz   = rws.w_nzs[tid]
        c      = rws.cs_ws[tid]
        c_flag = rws.cflags[tid]
        c_nz   = rws.c_nzs[tid]

        # Phase 1: w = A * P[:,j]
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

        # Phase 2: c = R * w
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

        # Phase 3: write directly into the pre-allocated nzval range for column j.
        # After sort!, c_nz is in ascending order and matches C.rowval[colptr[j]:colptr[j+1]-1].
        # QuickSort is allocation-free unlike the default MergeSort.
        sort!(c_nz; alg = Base.Sort.QuickSort)
        cptr = C.colptr[j]
        @inbounds for (pos, i) in enumerate(c_nz)
            C.nzval[cptr + pos - 1] = c[i]
            c[i]      = zero(Tv)
            c_flag[i] = false
        end
        empty!(c_nz)
    end

    return C
end
