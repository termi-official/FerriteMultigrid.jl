using FerriteMultigrid, Test
import FerriteMultigrid: build_prolongator

@testset "Prolongator" begin
    grid = generate_grid(Line, (3,))

    # fine space
    ip_fine = Lagrange{RefLine,2}()
    qr_fine = QuadratureRule{RefLine}(3)
    cv_fine = CellValues(qr_fine, ip_fine)
    dh_fine = DofHandler(grid)
    add!(dh_fine, :u, ip_fine)
    close!(dh_fine)

    # coarse space
    ip_coarse = Lagrange{RefLine,1}()
    qr_coarse = QuadratureRule{RefLine}(3) # same quadrature rule
    cv_coarse = CellValues(qr_coarse, ip_coarse)
    dh_coarse = DofHandler(grid)
    add!(dh_coarse, :u, ip_coarse)
    close!(dh_coarse)

    # test assembled prolongator
    P = build_prolongator(dh_fine, dh_coarse)
    # Prolongator
    I = [1, 2, 3, 3, 4, 5, 5, 6, 7, 7];
    J = [1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
    V = [1, 1, 0.5, 0.5, 1, 0.5, 0.5, 1, 0.5, 0.5];
    P_expected = sparse(I, J, V)
    @test P ≈ P_expected
end
