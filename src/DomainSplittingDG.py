import dolfinx as dfx
from ProblemDataUFL import ProblemData
from SpaceDiscretization import DG
from TimeIntegrationDG import (
    TimeIntegratorWaveDG,
    CrankNicolsonDG,
    leapfrogDG2,
    leapfrogDG,
)
from Domain import OvSubDomain, DS_reg
from typing import Type
from Config import DS_DG_STR


class DomainSplittingDG(TimeIntegratorWaveDG):
    name = DS_DG_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: DS_reg,
        SpaceDiscretizationClass: Type[DG],
        problem: ProblemData,
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

    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        subdomain_lfs: list[leapfrogDG | leapfrogDG2],
        subdomain_CNs: list[CrankNicolsonDG],
    ):
        qn_i_plus_list = []
        pn_i_plus_list = []
        """
        Step 1: prediction
        """
        for i, Omega_i_delta in enumerate(self.SDs):
            q_D = dfx.fem.Function(
                Omega_i_delta.prediction_subdomain.V
            )  # zero boundary condition
            pred_i, pred_p_i = subdomain_lfs[i].step(
                tn,
                tau,
                Omega_i_delta.prediction_subdomain.restrict_to_local(qn),
                Omega_i_delta.prediction_subdomain.restrict_to_local(pn),
                g_bc=q_D,
            )

            pred = dfx.fem.Function(Omega_i_delta.V)
            pred.x.array[Omega_i_delta.interface_dofs] = pred_i.x.array[
                Omega_i_delta.prediction_subdomain.prediction_dofs
            ]

            if self.problem.u_bc is not None:  # add outer boundary conditions
                u_D = dfx.fem.Function(Omega_i_delta.V)
                u_D.interpolate(
                    self.problem.dfExpression(
                        self.problems[i].u_bc(tn + self.tau), self.SDs[i].V
                    )
                )
                pred.x.array[Omega_i_delta.parent_boundary_dofs] = u_D.x.array[
                    Omega_i_delta.parent_boundary_dofs
                ]

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
                g_bc=pred,
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

    def integrate(self, startup_ritz_projection=False):
        subdomain_lfs = [
            leapfrogDG2(
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
            CrankNicolsonDG(
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
        for n in range(self.start_index, self.N):
            tn = n * self.tau
            qn_plus_1, pn_plus_1 = self.step(
                tn,
                self.tau,
                qn,
                pn,
                subdomain_lfs,
                subdomains_CNs,
            )

            qn.x.array[:] = qn_plus_1.x.array
            pn.x.array[:] = pn_plus_1.x.array

        return qn, pn
