function create_1d_1element_mass_matrix(p = 1, nqp = 2)
    grid = generate_grid(Line, (1,), Vec((0.0,)), Vec((1.0,)))

    ip = Lagrange{RefLine,p}()
    qr = QuadratureRule{RefLine}(nqp)
    cellvalues = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        _element_mass_matrix!(Me, cellvalues)
    end
    return Me
end

@testset "Mass Matrix" begin
    # p =1 ,nqp = 2
    Me = create_1d_1element_mass_matrix(1, 2)
    Me_expected = 1/6 * [2 1; 1 2]
    @test Me ≈ Me_expected
    # p = 2, nqp = 3
    Me2 = create_1d_1element_mass_matrix(2, 3)
    Me2_expected = (1/30) * [
        4 -1 2;
        -1 4 2;
        2 2 16
    ]

    @test Me2 ≈ Me2_expected
end

@testset "Prolongator" begin
    grid = generate_grid(Line, (3,))

    # fine space
    ip_fine = Lagrange{RefLine,2}()
    qr_fine = QuadratureRule{RefLine}(3)
    cv_fine = CellValues(qr_fine, ip_fine)
    dh_fine = DofHandler(grid)
    add!(dh_fine, :u, ip_fine)
    close!(dh_fine)
    ch_fine = ConstraintHandler(dh_fine)
    close!(ch_fine)
    fine_fespace = FESpace(dh_fine, cv_fine,ch_fine)

    # coarse space
    ip_coarse = Lagrange{RefLine,1}()
    qr_coarse = QuadratureRule{RefLine}(3) # in current implementation, we use same quadrature rule
    cv_coarse = CellValues(qr_coarse, ip_coarse)
    dh_coarse = DofHandler(grid)
    add!(dh_coarse, :u, ip_coarse)
    close!(dh_coarse)
    ch_coarse = ConstraintHandler(dh_coarse)
    close!(ch_coarse)
    coarse_fespace = FESpace(dh_coarse, cv_coarse, ch_coarse)

    # test element prolongator
    Pe = zeros(getnbasefunctions(cv_fine), getnbasefunctions(cv_coarse))
    Pebuf = zeros(getnbasefunctions(cv_fine), getnbasefunctions(cv_coarse))
    Me = zeros(getnbasefunctions(cv_fine), getnbasefunctions(cv_fine))
    cell_iter = CellIterator(dh_fine)
    cell = first(cell_iter)
    reinit!(cv_fine, cell)
    element_prolongator!(Pe, Me, cv_fine, cv_coarse, Pebuf)
    Pe_expected = [
        1.0 0;
        0.0 1.0;
        0.5 0.5
    ]
    @test Pe ≈ Pe_expected


    # test assembled prolongator
    P = build_prolongator(fine_fespace, coarse_fespace)
    # Prolongator
    I = [1, 2, 3, 3, 4, 5, 5, 6, 7, 7];
    J = [1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
    V = [1, 1, 0.5, 0.5, 1, 0.5, 0.5, 1, 0.5, 0.5];
    P_expected = sparse(I, J, V)
    @test P ≈ P_expected
end
