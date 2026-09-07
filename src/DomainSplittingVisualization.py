import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import dolfinx as dfx
from ProblemDataUFL import ProblemData
from SpaceDiscretization import SpaceDiscretization, FEM_ml
from TimeIntegration import TimeIntegratorWave, CrankNicolson, leapfrog
from Domain import Domain, SubDomain, OvSubDomain, DS_reg
from typing import Type, Tuple
from Config import DS_STR
from Plotting import *
from DolfinVersion import DIRICHLET_BC_TYPE

default_color = "silver"
color_cycle = ["crimson", "blue", "cyan", "yellow"]
color_cycle2 = [
    np.array([113 / 256, 0 / 256, 0 / 256, 1.0]),
    np.array([226 / 256, 199 / 256, 0 / 256, 1.0]),
    np.array([0 / 256, 0 / 256, 103 / 256, 1.0]),
    np.array([0, 184 / 256, 208 / 256, 1.0]),
]
values = [1, 2, 3, 0]


class DomainSplittingVisualization(TimeIntegratorWave):
    name = DS_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: DS_reg,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
        plotting_verbosity: int = 1,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)
        self.SDs: list[OvSubDomain] = self.Omega.SDs
        self.numSDs: int = len(self.SDs)
        self.problems: list[ProblemData] = [
            ProblemData(
                self.problem.example_number, Omega_i_delta.V, self.problem.c_init
            )
            for Omega_i_delta in self.SDs
        ]
        for subdomain in self.SDs:
            subdomain.generate_prediction_subdomain(layers=1)
        self.prediction_problems: list[ProblemData] = [
            ProblemData(
                self.problem.example_number,
                Omega_i_delta.prediction_subdomain.V,
                self.problem.c_init,
            )
            for Omega_i_delta in self.SDs
        ]
        self.plotting_verbosity = plotting_verbosity

    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        subdomain_bcs: list[DIRICHLET_BC_TYPE],
        subdomain_lfs: list[leapfrog],
        subdomain_CNs: list[CrankNicolson],
    ):
        qn_i_plus_list = []
        pn_i_plus_list = []
        n = int(round(tn / tau)) + 1
        """
        Step 1: prediction
        """
        predicted_bcs = []
        for i, Omega_i_delta in enumerate(self.SDs):
            q_D = dfx.fem.Function(self.Omega.V)
            q_bc = dfx.fem.dirichletbc(
                q_D, Omega_i_delta.prediction_subdomain.boundary_dofs
            )
            pred_i, pred_p_i = subdomain_lfs[i].step(
                tn,
                tau,
                Omega_i_delta.prediction_subdomain.restrict_to_local(qn),
                Omega_i_delta.prediction_subdomain.restrict_to_local(pn),
                q_bc,
            )

            pred = dfx.fem.Function(Omega_i_delta.V)
            pred.x.array[Omega_i_delta.interface_dofs] = pred_i.x.array[
                Omega_i_delta.prediction_subdomain.prediction_dofs
            ]

            predicted_bcs.append(
                dfx.fem.dirichletbc(
                    pred,
                    Omega_i_delta.interface_dofs,
                )
            )

            if self.plotting_verbosity > 0:
                ones = dfx.fem.Function(Omega_i_delta.prediction_subdomain.V)
                ones.x.array[Omega_i_delta.prediction_subdomain.prediction_dofs] = 1.0
                if n < 2:
                    plot_sol_pyvista_flat(
                        self.V,
                        Omega_i_delta.prediction_subdomain.project_to_global(ones),
                        name=f"Prediction interface {i+1}",
                        cmap=[default_color] + [color_cycle[values[i]]],
                        clim=[0.0, 1.0],
                    )
                plot_sol_pyvista(
                    self.V,
                    Omega_i_delta.project_to_global(pred),
                    name=f"Prediction_{i+1}^{n}",
                    clim=(0.0, 1.0),
                )

            """
            Step 2: local calculation
            """
        for i, Omega_i_delta in enumerate(self.SDs):
            qn_i_projection = Omega_i_delta.restrict_to_local(qn)
            pn_i_projection = Omega_i_delta.restrict_to_local(pn)

            subdomain_sol = subdomain_CNs[i].step(
                tn,
                tau,
                qn_i_projection,
                pn_i_projection,
                [subdomain_bcs[i], predicted_bcs[i]],
            )

            if self.plotting_verbosity > 0:
                ones = dfx.fem.Function(Omega_i_delta.V)
                ones.x.array[:] = 1.0
                if n < 2:
                    plot_sol_pyvista_flat(
                        self.V,
                        Omega_i_delta.project_to_global(ones),
                        cmap=[default_color] + [color_cycle[values[i]]],
                        clim=[0.0, 1.0],
                        name=f"subdomain {i+1}",
                    )
                plot_sol_pyvista(
                    self.V,
                    Omega_i_delta.project_to_global(subdomain_sol[0]),
                    name=f"q_{i+1}^{int(round(tn/tau))+1}",
                    clim=(0.0, 1.0),
                )

            qn_i_plus_list.append(subdomain_sol[0])
            pn_i_plus_list.append(subdomain_sol[1])

        if self.plotting_verbosity > 0:
            qn_p_sum = dfx.fem.Function(self.V)
            for i, qn_i_p in enumerate(qn_i_plus_list):
                glob_func = self.SDs[i].non_ov.project_to_global(
                    self.SDs[i].restrict_from_ov_to_non_ov_SD(qn_i_p)
                )
                qn_p_sum.x.array[:] += glob_func.x.array[:]
                plot_sol_pyvista(
                    self.V,
                    glob_func,
                    name=f"q_{i+1}^{int(round(tn/tau))+1} cutted",
                    clim=(0.0, 1.0),
                )
            plot_sol_pyvista(
                self.V, qn_p_sum, name=f"simple sum ({n})", clim=(0.0, 1.0)
            )

        """
        Step 3: averaging
        """
        qn_p = self.Omega.avg(*qn_i_plus_list)
        pn_p = self.Omega.avg(*pn_i_plus_list)

        if self.plotting_verbosity > 0:
            constants = []
            for i, Omega_i_delta in enumerate(self.SDs):
                constant = dfx.fem.Function(Omega_i_delta.V)
                constant.x.array[:] = 0.5 + 0.25 * (i + 1)
                constants.append(constant)
            if n < 2:

                def Omega_0(x):
                    return (x[0] <= 0.5) & (x[1] <= 0.5)

                def Omega_1(x):
                    return (x[0] <= 0.5) & (x[1] >= 0.5)

                def Omega_2(x):
                    return (x[0] >= 0.5) & (x[1] <= 0.5)

                def Omega_3(x):
                    return (x[0] >= 0.5) & (x[1] >= 0.5)

                plot_subdomains(
                    Omega.V,
                    [Omega_0, Omega_1, Omega_2, Omega_3],
                    [0.5 + 0.25 * (i + 1) for i in range(4)],
                    cmap="jet",
                )

                plot_sol_pyvista(
                    self.V,
                    self.Omega.avg(*constants),
                    name="averaging between the domains",
                    cmap="jet",
                )

            plot_sol_pyvista(
                self.V, qn_p, name=f"qDS{int(round(tn/tau))+1}", clim=(0.0, 1.0)
            )

        return qn_p, pn_p

    def assemble_space_discretization(self):
        return super().assemble_space_discretization()

    def integrate(self, startup_ritz_projection=False):
        if self.problem.u_bc is None:
            u_Ds = [
                dfx.fem.Constant(Omega_i_delta.V.mesh, 0.0)
                for Omega_i_delta in self.SDs
            ]
            subdomain_bcs = [
                dfx.fem.dirichletbc(
                    u_Ds[i], Omega_i_delta.parent_boundary_dofs, Omega_i_delta.V
                )
                for i, Omega_i_delta in enumerate(self.SDs)
            ]

        subdomain_lfs = [
            leapfrog(
                self.T,
                self.tau,
                Omega_i_delta.prediction_subdomain,
                self.SpaceDiscretizationClass,
                self.prediction_problems[i],
            )
            for i, Omega_i_delta in enumerate(self.SDs)
        ]
        for slf in subdomain_lfs:
            slf.assemble_space_discretization()

        subdomains_CNs = [
            CrankNicolson(
                self.T,
                self.tau,
                Omega_i_delta,
                self.SpaceDiscretizationClass,
                self.problems[i],
            )
            for i, Omega_i_delta in enumerate(self.SDs)
        ]
        for sCN in subdomains_CNs:
            sCN.assemble_space_discretization()

        qn, pn = self.startup(ritz_projection=startup_ritz_projection)
        if self.plotting_verbosity > 0:
            plot_sol_pyvista(self.V, qn, name=f"qDS0", clim=(0.0, 1.0))
        for n in range(self.start_index, self.N):
            tn = n * self.tau
            if self.problem.u_bc is not None:
                subdomain_bcs = []
                for i, Omega_i_delta in enumerate(self.SDs):
                    u_D = dfx.fem.Function(Omega_i_delta.V)
                    u_D.interpolate(
                        self.problem.dfExpression(
                            self.problems[i].u_bc(tn + self.tau), self.SDs[i].V
                        )
                    )
                    sd_bc = dfx.fem.dirichletbc(u_D, Omega_i_delta.parent_boundary_dofs)
                    subdomain_bcs.append(sd_bc)

            qn_plus_1, pn_plus_1 = self.step(
                tn,
                self.tau,
                qn,
                pn,
                subdomain_bcs,
                subdomain_lfs,
                subdomains_CNs,
            )

            qn.x.array[:] = qn_plus_1.x.array
            pn.x.array[:] = pn_plus_1.x.array

        return qn, pn


if __name__ == "__main__":
    # data
    ex = 228
    a = 0
    b = 1
    h = 0.05
    ell = 4
    tau = 0.01
    T = 0.03
    import pyvista

    # pyvista.rcParams["transparent_background"] = True
    # pyvista.OFF_SCREEN = True

    Omega = SplittedGrids2D.DS_2D_4SD_cross_SQUARE(a, b, h, ell)
    problem = ProblemData(ex, Omega.V)
    TI = DomainSplittingVisualization(T, tau, Omega, FEM_ml, problem)
    qn, pn = TI.integrate()
