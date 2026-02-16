using Ferrite, Tensors, TimerOutputs, IterativeSolvers

using FerriteMultigrid

struct NeoHooke
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

function constitutive_driver(C, mp::NeoHooke)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

function assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    b = Vec{3}((0.0, -0.5, 0.0)) # Body force
    tn = 0.1 # Traction (to be scaled with surface normal)
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += (∇δui ⊡ P - δui ⋅ b) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end

    # Surface integral for the traction
    for facet in 1:nfacets(cell)
        if (cellid(cell), facet) in ΓN
            reinit!(fv, cell, facet)
            for q_point in 1:getnquadpoints(fv)
                t = tn * getnormal(fv, q_point)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ t) * dΓ
                end
            end
        end
    end
    return
end;

function assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and g
    assembler = start_assemble(K, g)

    # Loop over all cells in the grid
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN)
        assemble!(assembler, global_dofs, ke, ge)
    end
    return
end;

function create_nns(dh, fieldname = first(dh.field_names))
    @assert length(dh.field_names) == 1 "Only a single field is supported for now."

    coords_flat = zeros(ndofs(dh))
    apply_analytical!(coords_flat, dh, fieldname, x -> x)
    coords = reshape(coords_flat, (length(coords_flat) ÷ 3, 3))

    grid = dh.grid
    B = zeros(Float64, ndofs(dh), 6)
    B[1:3:end, 1] .= 1 # x - translation
    B[2:3:end, 2] .= 1 # y - translation
    B[3:3:end, 3] .= 1 # z - translation

    # rotations
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    # Around x
    B[2:3:end, 4] .= -z
    B[3:3:end, 4] .= y
    # Around y
    B[1:3:end, 5] .= z
    B[3:3:end, 5] .= -x
    # Around z
    B[1:3:end, 6] .= -y
    B[2:3:end, 6] .= x

    return B
end

function _solve()
    reset_timer!()

    # Generate a grid
    N = 10
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Tetrahedron, (N, N, N), left, right)

    # Material parameters
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    # Finite element base
    ip = Lagrange{RefTetrahedron, 2}()^3
    qr = QuadratureRule{RefTetrahedron}(4)
    qr_facet = FacetQuadratureRule{RefTetrahedron}(3)
    cv = CellValues(qr, ip)
    fv = FacetValues(qr_facet, ip)

    # DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, ip) # Add a displacement field
    close!(dh)

    function rotation(X, t)
        θ = pi / 3 # 60°
        x, y, z = X
        return t * Vec{3}(
            (
                0.0,
                L / 2 - y + (y - L / 2) * cos(θ) - (z - L / 2) * sin(θ),
                L / 2 - z + (y - L / 2) * sin(θ) + (z - L / 2) * cos(θ),
            )
        )
    end

    ch = ConstraintHandler(dh)
    # Add a homogeneous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(ch, dbc)
    dbc = Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> rotation(x, t), [1, 2, 3])
    add!(ch, dbc)
    close!(ch)
    t = 0.5
    Ferrite.update!(ch, t)

    # Neumann part of the boundary
    ΓN = union(
        getfacetset(grid, "top"),
        getfacetset(grid, "bottom"),
        getfacetset(grid, "front"),
        getfacetset(grid, "back"),
    )

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)
    apply!(un, ch)

    # Create sparse matrix and residual vector
    K = allocate_matrix(dh)
    g = zeros(_ndofs)

    # FIXME this needs better integration
    dh_coarse = DofHandler(grid)
    add!(dh_coarse, :u, Lagrange{RefTetrahedron, 1}()^3) # Add a displacement field
    close!(dh_coarse)
    B = create_nns(dh_coarse)
    config_gal = pmultigrid_config(coarse_strategy = Galerkin())
    fe_space = FESpace(dh, cv, ch)

    builder = PMultigridPreconBuilder(fe_space, config_gal)

    # Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1.0e-8
    NEWTON_MAXITER = 30

    @info ndofs(dh)

    while true
        newton_itr += 1
        # Construct the current guess
        u .= un .+ Δu
        # Compute residual and tangent for current guess
        assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
        # Apply boundary conditions
        apply_zero!(K, g, ch)
        # Compute the residual norm and compare with tolerance
        normg = norm(g)
        if normg < NEWTON_TOL
            break
        elseif newton_itr > NEWTON_MAXITER
            error("Reached maximum Newton iterations, aborting")
        end

        # Compute increment using conjugate gradients
        fill!(ΔΔu, 0.0)
        @timeit "Setup preconditioner" Pl = builder(K)[1]
        @timeit "Galerkin CG" _, ch_gal = IterativeSolvers.cg!(ΔΔu, K, g; Pl, maxiter = 1000, log=true, verbose=false)
        @info "Galerkin CG iterations: $(ch_gal.iters)"
        @timeit "Galerkin only" solve(K, g, fe_space, config_gal;B = B, log=true, rtol = 1e-10)
        fill!(ΔΔu, 0.0)
        @timeit "CG" _, ch_cg = IterativeSolvers.cg!(ΔΔu, K, g; maxiter = 1000, log=true, verbose=false)
        @info "CG iterations: $(ch_cg.iters)"

        apply_zero!(ΔΔu, ch)
        Δu .-= ΔΔu
    end

    # Save the solution
    @timeit "export" begin
        VTKGridFile("hyperelasticity", dh) do vtk
            write_solution(vtk, dh, u)
        end
    end

    print_timer(title = "Analysis with $(getncells(grid)) elements", linechars = :ascii)
    return u
end

u = _solve();

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
