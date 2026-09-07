import numpy as np
from mpi4py import MPI
import dolfinx as dfx
import ufl.finiteelement
from Domain import DS_reg
from Config import FEM_TYPE, FEM_DEGREE
import basix

import ufl


class DS_1D_2SD(DS_reg):
    def __init__(self, a: float, b: float, h: float, ell: int) -> None:
        self.__name__ = f"DS_1D_2SD::a={a}::b={b}::h={h}::ell={ell}"
        self.a = a
        self.b = b
        self.h = h
        self.ell = ell
        nx = int(round((b - a) / h))
        self.N = nx + 1  # number of nodes
        mid = a + int(self.N / 2) * h
        alpha_1 = a
        beta_1 = mid + ell * h
        alpha_2 = mid - ell * h
        beta_2 = b
        nx_1 = int(round((beta_1 - alpha_1) / h))
        nx_2 = int(round((beta_2 - alpha_2) / h))

        non_ov_nx1 = int(round((mid - a) / h))
        non_ov_nx2 = int(round((b - mid) / h))

        fem_type = FEM_TYPE
        degree = FEM_DEGREE
        mesh = dfx.mesh.create_interval(MPI.COMM_SELF, nx, [a, b])
        mesh1 = dfx.mesh.create_interval(MPI.COMM_SELF, nx_1, [alpha_1, beta_1])
        non_ov_mesh1 = dfx.mesh.create_interval(MPI.COMM_SELF, non_ov_nx1, [a, mid])
        mesh2 = dfx.mesh.create_interval(MPI.COMM_SELF, nx_2, [alpha_2, beta_2])
        non_ov_mesh2 = dfx.mesh.create_interval(MPI.COMM_SELF, non_ov_nx2, [mid, b])
        SD_meshlist = [mesh1, mesh2]
        non_ov_SD_meshlist = [non_ov_mesh1, non_ov_mesh2]

        super().__init__(
            mesh,
            fem_type,
            degree,
            non_ov_SD_meshlist,
            SD_meshlist,
        )


class DS_1D_4SD(DS_reg):
    def __init__(self, a: float, b: float, h: float, ell: int) -> None:
        self.__name__ = f"DS_1D_4SD::a={a}::b={b}::h={h}::ell={ell}"
        self.a = a
        self.b = b
        self.h = h
        self.ell = ell
        nx = int(round((b - a) / h))
        self.N = nx + 1  # number of nodes
        quad1 = a + int(self.N / 4) * h
        quad2 = a + int(self.N / 2) * h
        quad3 = a + int(3 * self.N / 4) * h
        quads = [a, quad1, quad2, quad3, b]

        alphas = [a, quad1 - ell * h, quad2 - ell * h, quad3 - ell * h]
        betas = [quad1 + ell * h, quad2 + ell * h, quad3 + ell * h, b]
        nxs = [int(round((betas[i] - alphas[i]) / h)) for i in range(4)]

        non_ov_nxs = [int(round((quads[i + 1] - quads[i]) / h)) for i in range(4)]

        fem_type = FEM_TYPE
        degree = FEM_DEGREE
        mesh = dfx.mesh.create_interval(MPI.COMM_SELF, nx, [a, b])
        SD_meshlist = [
            dfx.mesh.create_interval(MPI.COMM_SELF, nxs[i], [alphas[i], betas[i]])
            for i in range(4)
        ]
        non_ov_SD_meshlist = [
            dfx.mesh.create_interval(
                MPI.COMM_SELF, non_ov_nxs[i], [quads[i], quads[i + 1]]
            )
            for i in range(4)
        ]

        super().__init__(
            mesh,
            fem_type,
            degree,
            non_ov_SD_meshlist,
            SD_meshlist,
        )


class DS_1D_8SD(DS_reg):
    def __init__(self, a: float, b: float, h: float, ell: int) -> None:
        self.__name__ = f"DS_1D_4SD::a={a}::b={b}::h={h}::ell={ell}"
        self.a = a
        self.b = b
        self.h = h
        self.ell = ell
        nx = int(round((b - a) / h))
        self.N = nx + 1  # number of nodes
        eights = [a + int(i * self.N / 8) * h for i in range(1, 8)]
        splits = [a, *eights, b]

        eights_minus = [eight - ell * h for eight in eights]
        eights_plus = [eight + ell * h for eight in eights]
        alphas = [a, *eights_minus]
        betas = [*eights_plus, b]
        nxs = [int(round((betas[i] - alphas[i]) / h)) for i in range(8)]

        non_ov_nxs = [int(round((splits[i + 1] - splits[i]) / h)) for i in range(8)]

        fem_type = FEM_TYPE
        degree = FEM_DEGREE
        mesh = dfx.mesh.create_interval(MPI.COMM_SELF, nx, [a, b])
        SD_meshlist = [
            dfx.mesh.create_interval(MPI.COMM_SELF, nxs[i], [alphas[i], betas[i]])
            for i in range(8)
        ]
        non_ov_SD_meshlist = [
            dfx.mesh.create_interval(
                MPI.COMM_SELF, non_ov_nxs[i], [splits[i], splits[i + 1]]
            )
            for i in range(8)
        ]

        super().__init__(
            mesh,
            fem_type,
            degree,
            non_ov_SD_meshlist,
            SD_meshlist,
        )


class DS_1D_2SD_asym(DS_reg):
    def __init__(self, a, b, h, ell) -> None:
        self.__name__ = f"DS_1D_2SD_asym::a={a}::b={b}::h={h}::ell={ell}"
        self.a = a
        self.b = b
        self.h = h
        self.ell = ell
        nx = int(round((b - a) / h))
        self.N = nx + 1  # number of nodes
        mid = a + int(2 * self.N / 3) * h
        alpha_1 = a
        beta_1 = mid + ell * h
        alpha_2 = mid - ell * h
        beta_2 = b
        nx_1 = int(round((beta_1 - alpha_1) / h))
        nx_2 = int(round((beta_2 - alpha_2) / h))

        non_ov_nx1 = int(round((mid - a) / h))
        non_ov_nx2 = int(round((b - mid) / h))

        fem_type = FEM_TYPE
        degree = FEM_DEGREE
        mesh = dfx.mesh.create_interval(MPI.COMM_SELF, nx, [a, b])
        mesh1 = dfx.mesh.create_interval(MPI.COMM_SELF, nx_1, [alpha_1, beta_1])
        non_ov_mesh1 = dfx.mesh.create_interval(MPI.COMM_SELF, non_ov_nx1, [a, mid])
        mesh2 = dfx.mesh.create_interval(MPI.COMM_SELF, nx_2, [alpha_2, beta_2])
        non_ov_mesh2 = dfx.mesh.create_interval(MPI.COMM_SELF, non_ov_nx2, [mid, b])
        SD_meshlist = [mesh1, mesh2]
        non_ov_SD_meshlist = [non_ov_mesh1, non_ov_mesh2]

        super().__init__(
            mesh,
            fem_type,
            degree,
            non_ov_SD_meshlist,
            SD_meshlist,
        )


class DS_1D_2SD_np_mesh(DS_reg):
    def __init__(
        self,
        np_mesh_Omega,
        np_mesh_Omega1delta,
        np_mesh_Omega1,
        np_mesh_Omega2delta,
        np_mesh_Omega2,
    ) -> None:
        fem_type = FEM_TYPE
        degree = FEM_DEGREE
        communicator = MPI.COMM_SELF
        mesh = self.generate_interval_mesh_from_vertices(np_mesh_Omega, communicator)

        # V = dfx.fem.functionspace(mesh, ("P", 1))
        # print(V.tabulate_dof_coordinates(), mesh.geometry.dim)
        # print(mesh.topology.index_map(mesh.topology.dim).size_local)

        mesh1 = self.generate_interval_mesh_from_vertices(
            np_mesh_Omega1delta, communicator
        )
        non_ov_mesh1 = self.generate_interval_mesh_from_vertices(
            np_mesh_Omega1, communicator
        )
        mesh2 = self.generate_interval_mesh_from_vertices(
            np_mesh_Omega2delta, communicator
        )
        non_ov_mesh2 = self.generate_interval_mesh_from_vertices(
            np_mesh_Omega2, communicator
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

    @classmethod
    def generate_interval_mesh_from_vertices(cls, vertices, communicator):
        N = vertices.size
        gdim, name, degree = 1, "interval", 1
        cell_type = dfx.mesh.CellType.interval
        domain = ufl.Mesh(
            basix.ufl.element("Lagrange", cell_type.name, 1, shape=(gdim,))
        )
        x = vertices.reshape((N, 1))
        cells = np.array([[i - 1, i] for i in range(1, N)], dtype=np.int64)
        mesh = dfx.mesh.create_mesh(comm=communicator, cells=cells, e=domain, x=x)
        return mesh


class DS_1D_2SD_distorted(DS_1D_2SD_np_mesh):
    def __init__(self, a, b, h, ell) -> None:
        self.__name__ = (
            f"DS_1D_2SD_distorted::a={a}::b={b}::h={h}::ell={ell}::distortion={h/5}"
        )
        self.a = a
        self.b = b
        self.h = h
        self.distortion = h / 5
        self.ell = ell
        nx = int(round((b - a) / h))
        self.N = nx + 1  # number of nodes
        mid = a + int(self.N / 2) * h
        alpha_1 = a
        beta_1 = mid + ell * h
        alpha_2 = mid - ell * h
        beta_2 = b
        nx_1 = int(round((beta_1 - alpha_1) / h))
        nx_2 = int(round((beta_2 - alpha_2) / h))

        non_ov_nx1 = int(round((mid - a) / h))
        non_ov_nx2 = int(round((b - mid) / h))

        np.random.seed(0)
        distortion_vector = np.random.uniform(-self.distortion, self.distortion, nx - 1)

        np_mesh_Omega = np.linspace(a, b, nx + 1)
        np_mesh_Omega[1:-1] += distortion_vector
        np_mesh_Omega1delta = np_mesh_Omega[: nx_1 + 1]
        np_mesh_Omega1 = np_mesh_Omega[: non_ov_nx1 + 1]
        np_mesh_Omega2delta = np_mesh_Omega[-nx_2 - 1 :]
        np_mesh_Omega2 = np_mesh_Omega[-non_ov_nx2 - 1 :]

        super().__init__(
            np_mesh_Omega,
            np_mesh_Omega1delta,
            np_mesh_Omega1,
            np_mesh_Omega2delta,
            np_mesh_Omega2,
        )


class DS_1D_2SD_alternating(DS_1D_2SD_np_mesh):
    def __init__(self, a, b, h, ell) -> None:
        self.__name__ = f"DS_1D_2SD_alternating::a={a}::b={b}::h={h}::ell={ell}"
        part1 = np.arange(a, b + 0.1 * h, h)
        part2 = np.arange(a + 1.5 * h, b + 0.1 * h, 2 * h)
        full_grid = np.concatenate((part1, part2))
        np_mesh_Omega = np.sort(full_grid)
        np_mesh_Omega1 = np_mesh_Omega[: int(round(len(np_mesh_Omega) / 2))]
        np_mesh_Omega2 = np_mesh_Omega[int(round(len(np_mesh_Omega) / 2)) - 1 :]
        np_mesh_Omega1delta = np_mesh_Omega[: int(len(np_mesh_Omega) / 2) + ell + 1]
        np_mesh_Omega2delta = np_mesh_Omega[int(len(np_mesh_Omega) / 2) - ell :]

        super().__init__(
            np_mesh_Omega,
            np_mesh_Omega1delta,
            np_mesh_Omega1,
            np_mesh_Omega2delta,
            np_mesh_Omega2,
        )
