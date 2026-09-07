from dolfinx.mesh import Mesh
import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from Domain import DS_reg
from Config import FEM_TYPE, FEM_DEGREE


class SQUARE(DS_reg):
    def __init__(self, a, b, h, *args) -> None:
        self.__name__ = f"SQUARE::a={a}::b={b}::h={h}"
        self.hx = self.hy = h
        self.xL = self.yB = a
        self.xR = self.yT = b

        nx = round((self.xR - self.xL) / self.hx)
        ny = round((self.yT - self.yB) / self.hy)

        pointLeftBottom = (self.xL, self.yB)
        pointRightTop = (self.xR, self.yT)

        mesh = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom, pointRightTop),
            n=(nx, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )

        if mesh.comm.rank == 0:
            print("Warning: DS_reg ignored additional arguments", args)

        fem_type = FEM_TYPE
        degree = FEM_DEGREE

        super().__init__(mesh, fem_type, degree, [], [])

    def get_avg_weights(self) -> np.ndarray:
        return np.array(None)

    def assert_avg_ones(self):
        pass


class DS_2D_2SDSQUARE(DS_reg):
    def __init__(self, a, b, h, ell) -> None:
        self.__name__ = f"DS_2D_2SDSQUARE::a={a}::b={b}::h={h}::ell={ell}"
        self.prediction_marked = False
        self.hx = self.hy = h
        self.xL = self.yB = a
        self.xR = self.yT = b
        self.ell = ell

        nx = round((self.xR - self.xL) / self.hx)
        ny = round((self.yT - self.yB) / self.hy)

        pointLeftBottom = (self.xL, self.yB)
        pointRightTop = (self.xR, self.yT)

        mesh = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom, pointRightTop),
            n=(nx, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )
        self.Nx = nx + 1
        self.Ny = ny + 1

        mid_x = self.xL + int(self.Nx / 2) * self.hx
        alphax_1 = self.xL
        betax_1 = mid_x + self.ell * self.hx
        alphax_2 = mid_x - self.ell * self.hx
        betax_2 = self.xR

        nx_1 = round((betax_1 - alphax_1) / self.hx)
        nx_2 = round((betax_2 - alphax_2) / self.hx)

        non_ov_nx1 = round((mid_x - self.xL) / self.hx)
        non_ov_nx2 = round((self.xR - mid_x) / self.hx)

        fem_type = FEM_TYPE
        degree = FEM_DEGREE

        # ov1
        pointLeftBottom_ov1 = (alphax_1, self.yB)
        pointRightTop_ov1 = (betax_1, self.yT)
        mesh1 = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom_ov1, pointRightTop_ov1),
            n=(nx_1, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )
        # nov1
        pointLeftBottom_nov1 = (alphax_1, self.yB)
        pointRightTop_nov1 = (mid_x, self.yT)
        non_ov_mesh1 = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom_nov1, pointRightTop_nov1),
            n=(non_ov_nx1, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )

        # ov2
        pointLeftBottom_ov2 = (alphax_2, self.yB)
        pointRightTop_ov2 = (betax_2, self.yT)
        mesh2 = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom_ov2, pointRightTop_ov2),
            n=(nx_2, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )

        # nov2
        pointLeftBottom_nov2 = (mid_x, self.yB)
        pointRightTop_nov2 = (betax_2, self.yT)
        non_ov_mesh2 = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom_nov2, pointRightTop_nov2),
            n=(non_ov_nx2, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )

        SD_meshlist = [mesh1, mesh2]
        non_ov_SD_meshlist = [non_ov_mesh1, non_ov_mesh2]

        super().__init__(
            mesh,
            fem_type,
            degree,
            non_ov_SD_meshlist,
            SD_meshlist,
        )


class DS_2D_NSD_strip_SQUARE(DS_reg):
    def __init__(self, a, b, h, ell, N=8) -> None:
        self.hx = self.hy = h
        self.xL = self.yB = a
        self.xR = self.yT = b
        self.ell = ell
        self.N_subdomains = N
        self.__name__ = f"DS_2D_{self.N_subdomains}SD_strip_SQUARE::a={a}::b={b}::h={h}::ell={ell}::N={N}"

        nx = round((self.xR - self.xL) / self.hx)
        ny = round((self.yT - self.yB) / self.hy)

        pointLeftBottom = (self.xL, self.yB)
        pointRightTop = (self.xR, self.yT)

        mesh = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom, pointRightTop),
            n=(nx, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )
        self.Nx = nx + 1
        self.Ny = ny + 1

        fracs = np.linspace(0, 1, self.N_subdomains + 1)
        splitter_x = [a + int(round(nx * frac)) * self.hx for frac in fracs]
        alphas_x = [
            splitter_x[0],
            *[splitter_x[i] - ell * self.hx for i in range(1, self.N_subdomains)],
        ]
        betas_x = [
            *[splitter_x[i] + ell * self.hx for i in range(1, self.N_subdomains)],
            splitter_x[-1],
        ]
        nxs = [
            int(round((betas_x[i] - alphas_x[i]) / self.hx))
            for i in range(self.N_subdomains)
        ]
        non_ov_nxs = [
            int(round((splitter_x[i + 1] - splitter_x[i]) / h))
            for i in range(self.N_subdomains)
        ]

        fem_type = FEM_TYPE
        degree = FEM_DEGREE

        SD_meshlist = [
            dfx.mesh.create_rectangle(
                comm=MPI.COMM_SELF,
                points=((alphas_x[i], self.yB), (betas_x[i], self.yT)),
                n=(nxs[i], ny),
                cell_type=dfx.mesh.CellType.triangle,
            )
            for i in range(self.N_subdomains)
        ]
        non_ov_SD_meshlist = [
            dfx.mesh.create_rectangle(
                comm=MPI.COMM_SELF,
                points=((splitter_x[i], self.yB), (splitter_x[i + 1], self.yT)),
                n=(non_ov_nxs[i], ny),
                cell_type=dfx.mesh.CellType.triangle,
            )
            for i in range(self.N_subdomains)
        ]

        super().__init__(
            mesh,
            fem_type,
            degree,
            non_ov_SD_meshlist,
            SD_meshlist,
        )


class DS_2D_2SD_strip_SQUARE(DS_2D_NSD_strip_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 2
        super().__init__(a, b, h, ell, self.N)


class DS_2D_4SD_strip_SQUARE(DS_2D_NSD_strip_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 4
        super().__init__(a, b, h, ell, self.N)


class DS_2D_8SD_strip_SQUARE(DS_2D_NSD_strip_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 8
        super().__init__(a, b, h, ell, self.N)


class DS_2D_10SD_strip_SQUARE(DS_2D_NSD_strip_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 10
        super().__init__(a, b, h, ell, self.N)


class DS_2D_16SD_strip_SQUARE(DS_2D_NSD_strip_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 16
        super().__init__(a, b, h, ell, self.N)


class DS_2D_20SD_strip_SQUARE(DS_2D_NSD_strip_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 20
        super().__init__(a, b, h, ell, self.N)


class DS_2D_NMSD_cross_SQUARE(DS_reg):
    def __init__(self, a, b, h, ell, N=4, M=2) -> None:
        self.prediction_marked = False
        self.hx = self.hy = h
        self.xL = self.yB = a
        self.xR = self.yT = b
        self.ell = ell
        self.Nx_subdomains = N
        self.Ny_subdomains = M
        self.N_subdomains = N * M
        self.__name__ = f"DS_2D_{N}x{M}SD_cross_SQUARE_linFEM::a={a}::b={b}::h={h}::ell={ell}::N={N}::M={M}"

        nx = round((self.xR - self.xL) / self.hx)
        ny = round((self.yT - self.yB) / self.hy)

        pointLeftBottom = (self.xL, self.yB)
        pointRightTop = (self.xR, self.yT)

        mesh = dfx.mesh.create_rectangle(
            comm=MPI.COMM_SELF,
            points=(pointLeftBottom, pointRightTop),
            n=(nx, ny),
            cell_type=dfx.mesh.CellType.triangle,
        )
        self.Nx = nx + 1
        self.Ny = ny + 1

        fracs_x = np.linspace(0, 1, self.Nx_subdomains + 1)
        splitter_x = [self.xL + int(round(nx * frac)) * self.hx for frac in fracs_x]

        fracs_y = np.linspace(0, 1, self.Ny_subdomains + 1)
        splitter_y = [self.yB + int(round(ny * frac)) * self.hy for frac in fracs_y]

        alphas_x = [
            splitter_x[0],
            *[splitter_x[i] - ell * self.hx for i in range(1, self.Nx_subdomains)],
        ]
        betas_x = [
            *[splitter_x[i] + ell * self.hx for i in range(1, self.Nx_subdomains)],
            splitter_x[-1],
        ]
        alphas_y = [
            splitter_y[0],
            *[splitter_y[i] - ell * self.hy for i in range(1, self.Ny_subdomains)],
        ]
        betas_y = [
            *[splitter_y[i] + ell * self.hy for i in range(1, self.Ny_subdomains)],
            splitter_y[-1],
        ]

        bottom_left_corners = [
            (alphas_x[i], alphas_y[j])
            for i in range(self.Nx_subdomains)
            for j in range(self.Ny_subdomains)
        ]

        top_right_corners = [
            (betas_x[i], betas_y[j])
            for i in range(self.Nx_subdomains)
            for j in range(self.Ny_subdomains)
        ]

        nxs = [
            int(round((top_right_corners[i][0] - bottom_left_corners[i][0]) / self.hx))
            for i in range(self.N_subdomains)
        ]
        nys = [
            int(round((top_right_corners[i][1] - bottom_left_corners[i][1]) / self.hy))
            for i in range(self.N_subdomains)
        ]

        non_ov_nxs = [
            int(round((splitter_x[i + 1] - splitter_x[i]) / self.hx))
            for i in range(self.Nx_subdomains)
        ]
        non_ov_nxs *= self.Ny_subdomains
        non_ov_nys_h = [
            int(round((splitter_y[i + 1] - splitter_y[i]) / self.hy))
            for i in range(self.Ny_subdomains)
        ]
        non_ov_nys = [
            non_ov_nys_h[i % self.Ny_subdomains] for i in range(self.N_subdomains)
        ]

        fem_type = FEM_TYPE
        degree = FEM_DEGREE

        SD_meshlist = [
            dfx.mesh.create_rectangle(
                comm=MPI.COMM_SELF,
                points=(bottom_left_corners[i], top_right_corners[i]),
                n=(nxs[i], nys[i]),
                cell_type=dfx.mesh.CellType.triangle,
            )
            for i in range(self.N_subdomains)
        ]

        non_ov_SD_meshlist = [
            dfx.mesh.create_rectangle(
                comm=MPI.COMM_SELF,
                points=(
                    (splitter_x[i], splitter_y[j]),
                    (splitter_x[i + 1], splitter_y[j + 1]),
                ),
                n=(non_ov_nxs[i], non_ov_nys[j]),
                cell_type=dfx.mesh.CellType.triangle,
            )
            for i, j in [
                (i, j)
                for i in range(self.Nx_subdomains)
                for j in range(self.Ny_subdomains)
            ]
        ]

        super().__init__(
            mesh,
            fem_type,
            degree,
            non_ov_SD_meshlist,
            SD_meshlist,
        )


class DS_2D_4SD_cross_SQUARE(DS_2D_NMSD_cross_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 2
        self.M = 2
        super().__init__(a, b, h, ell, self.N, self.M)


class DS_2D_8SD_cross_SQUARE(DS_2D_NMSD_cross_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 4
        self.M = 2
        super().__init__(a, b, h, ell, self.N, self.M)


class DS_2D_9SD_cross_SQUARE(DS_2D_NMSD_cross_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 3
        self.M = 3
        super().__init__(a, b, h, ell, self.N, self.M)


class DS_2D_16SD_cross_SQUARE(DS_2D_NMSD_cross_SQUARE):
    def __init__(self, a, b, h, ell) -> None:
        self.N = 4
        self.M = 4
        super().__init__(a, b, h, ell, self.N, self.M)


def plot_dofs_2D(Omega):
    """
    motivates the mean averaging by plotting dofs with 1 function
    """
    u = dfx.fem.Function(Omega.V)
    test = np.ones(Omega.dim)
    u.x.array[:] = test
    local_sols = [Omega_i.restrict_to_local(u) for Omega_i in Omega.SDs]
    local_sols_plotting = [
        Omega_i.project_to_global(local_sols[i]) for i, Omega_i in enumerate(Omega.SDs)
    ]

    un_avg = Omega.avg(*local_sols)
    import matplotlib.pyplot as plt

    arrs = [u, *local_sols_plotting, un_avg]

    for i, sol in enumerate(arrs):
        try:
            import pyvista

            cells, types, x = dfx.plot.vtk_mesh(Omega.V)
            grid = pyvista.UnstructuredGrid(cells, types, x)
            grid.point_data["u"] = sol.x.array.real
            grid.set_active_scalars("u")
            plotter = pyvista.Plotter()
            plotter.background_color = "gray"
            plotter.add_mesh(grid, show_edges=True)
            warped = grid.warp_by_scalar()
            plotter.add_mesh(warped)
            if pyvista.OFF_SCREEN:
                pyvista.start_xvfb(wait=0.1)
                plotter.screenshot("uh_poisson.png")
            else:
                plotter.show()
        except ModuleNotFoundError:
            print("'pyvista' is required to visualise the solution")
            print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")


if __name__ == "__main__":
    a = 0
    b = 1
    h = 0.01
    ell = 2
    Omega = SQUARE(a, b, h, ell)
    # print(Omega.__name__)
    # plot_dofs_2D(Omega)
