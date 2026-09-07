import numpy as np
import dolfinx as dfx
from ProblemDataUFL import ProblemData
from SpaceDiscretization import SpaceDiscretization
from TimeIntegration import TimeIntegratorWave, CrankNicolson, leapfrog
from Domain import Domain, SubDomain, OvSubDomain, DS_reg
from typing import Type, Tuple
from Config import DS_STR, DSE_STR, DSLP_STR
from Projections import Ritz_projection_initial_values
from DomainSplitting import DomainSplitting
from DolfinVersion import DIRICHLET_BC_TYPE

class DomainSplittingExtrapolation(DomainSplitting):
    name = DSE_STR
    _default_extrapolation_order = 2
    _default_startup = "gloabl"

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: DS_reg,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
        extrapolation_order: int = _default_extrapolation_order,
        global_startup: bool = (_default_startup == "global"),
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)
        self.SDs: list[OvSubDomain] = self.Omega.SDs
        self.numSDs: int = len(self.SDs)
        self.problems: list[ProblemData] = [
            ProblemData(self.problem.example_number, Omega_i_delta.V, self.problem.c_init)
            for Omega_i_delta in self.SDs
        ]
        self.extrapolation_order: int = extrapolation_order
        self.global_startup: bool = global_startup

    def startup(self, ritz_projection=False):
        self.start_index = 0
        if ritz_projection:
            qn, pn = Ritz_projection_initial_values(self.problem, self.Omega)
        else:
            qn = dfx.fem.Function(self.V)
            qn.interpolate(self.problem.dfExpression(self.problem.u0, self.V))
            pn = dfx.fem.Function(self.V)
            pn.interpolate(self.problem.dfExpression(self.problem.v0, self.V))
        if self.global_startup:
            global_CN = CrankNicolson(
                self.tau * self.extrapolation_order,
                self.tau,
                self.Omega,
                self.SpaceDiscretizationClass,
                self.problem,
            )
            global_CN.assemble_space_discretization()
            approximations = []
            for n in range(
                self.start_index, min(max(0, self.extrapolation_order), self.N)
            ):
                tn = n * self.tau
                qD = dfx.fem.Function(self.V)
                qD.interpolate(
                    self.problem.dfExpression(self.problem.u_bc(tn + self.tau), self.V)
                )
                bc = dfx.fem.dirichletbc(qD, self.Omega.boundary_dofs)
                qn_plus_1, pn_plus_1 = global_CN.step(
                    tn, self.tau, qn, pn, self.problem.f, self.problem.c, [bc]
                )
                old_approx_n = dfx.fem.Function(self.Omega.V)
                old_approx_n.x.array[:] = qn.x.array
                approximations.append(old_approx_n)
                qn.x.array[:] = qn_plus_1.x.array
                pn.x.array[:] = pn_plus_1.x.array
            self.start_index = self.extrapolation_order
            return qn, pn, approximations
        else:
            return qn, pn

    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        old_approximations: list[dfx.fem.function.Function],
        subdomain_bcs: list[DIRICHLET_BC_TYPE],
        subdomain_CNs: list[CrankNicolson],
        step_extrapolation_order: int,
    ):
        qn_i_plus_list = []
        pn_i_plus_list = []
        """
        Step 1: prediction
        """

        for i, Omega_i_delta in enumerate(self.SDs):
            pred = dfx.fem.Function(Omega_i_delta.V)
            if step_extrapolation_order == 0:
                pred.x.array[Omega_i_delta.interface_dofs] = qn.x.array[
                    Omega_i_delta.interface_dofs_global
                ]
            elif step_extrapolation_order == 1:
                qn_minus = old_approximations[-1]
                pred.x.array[Omega_i_delta.interface_dofs] = (
                    2 * qn.x.array[Omega_i_delta.interface_dofs_global]
                    - qn_minus.x.array[Omega_i_delta.interface_dofs_global]
                )
            elif step_extrapolation_order == 2:
                qn_minus = old_approximations[-1]
                qn_minus_2 = old_approximations[-2]
                pred.x.array[Omega_i_delta.interface_dofs] = (
                    3 * qn.x.array[Omega_i_delta.interface_dofs_global]
                    - 3 * qn_minus.x.array[Omega_i_delta.interface_dofs_global]
                    + qn_minus_2.x.array[Omega_i_delta.interface_dofs_global]
                )
            else:
                raise ValueError("step_approximation_order must be 0,1 or 2")

            predicted_bc_i = dfx.fem.dirichletbc(
                pred,
                Omega_i_delta.interface_dofs,
            )

            """
            Step 2: local calculation
            """
            qn_i_projection = Omega_i_delta.restrict_to_local(qn)
            pn_i_projection = Omega_i_delta.restrict_to_local(pn)

            subdomain_sol = subdomain_CNs[i].step(
                tn,
                tau,
                qn_i_projection,
                pn_i_projection,
                [subdomain_bcs[i], predicted_bc_i],
            )

            qn_i_plus_list.append(subdomain_sol[0])
            pn_i_plus_list.append(subdomain_sol[1])

        """
        Step 3: averaging
        """
        qn_p = self.Omega.avg(*qn_i_plus_list)
        pn_p = self.Omega.avg(*pn_i_plus_list)

        return qn_p, pn_p

    def assemble_space_discretization(self):
        return super().assemble_space_discretization()

    def integrate(
        self,
        startup_ritz_projection=False,
        extrapolation_order=_default_extrapolation_order,
        global_startup=(_default_startup == "global"),
    ):
        self.global_startup = global_startup
        self.extrapolation_order = extrapolation_order
        if self.problem.u_bc is None:
            u_Ds = [
                dfx.fem.Constant(Omega_i_delta.V.mesh, 0.0) for Omega_i_delta in self.SDs
            ]
            subdomain_bcs = [
                dfx.fem.dirichletbc(
                    u_Ds[i], Omega_i_delta.parent_boundary_dofs, Omega_i_delta.V
                )
                for i, Omega_i_delta in enumerate(self.SDs)
            ]

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
        if self.global_startup:
            qn, pn, old_approximations = self.startup(
                ritz_projection=startup_ritz_projection
            )
        else:
            qn, pn = self.startup(ritz_projection=startup_ritz_projection)
            old_approximations = [
                dfx.fem.Function(self.Omega.V) for i in range(self.extrapolation_order)
            ]
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
                old_approximations,
                subdomain_bcs,
                subdomains_CNs,
                step_extrapolation_order=min(self.extrapolation_order, n),
            )

            for n in range(self.extrapolation_order - 1):
                old_approximations[n].x.array[:] = old_approximations[n + 1].x.array
            if self.extrapolation_order > 0:
                old_approximations[self.extrapolation_order - 1].x.array[:] = qn.x.array

            qn.x.array[:] = qn_plus_1.x.array
            pn.x.array[:] = pn_plus_1.x.array

        return qn, pn
