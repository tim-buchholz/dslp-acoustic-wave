import numpy as np
import dolfinx as dfx
from ProblemDataUFL import ProblemData
from SpaceDiscretization import SpaceDiscretization
from TimeIntegration import TimeIntegratorWave, CrankNicolson, leapfrog, CrankNicolson2
from Domain import Domain, SubDomain, OvSubDomain, DS_reg
from typing import Type, Tuple
from Config import DSLP_STR
from Projections import Ritz_projection_initial_values
from DomainSplitting import DomainSplitting
import math
from DolfinVersion import DIRICHLET_BC_TYPE


class DSLP(DomainSplitting):
    name = DSLP_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: DS_reg,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
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
            hmin = subdomain.get_h_min()
            numlayers = math.ceil(2 * self.tau * self.problem.c / hmin)
            # sufficient condition for numlayers*h >= 2*tau*c
            subdomain.generate_prediction_subdomain(layers=numlayers)
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
        subdomain_bcs: list[dfx.fem.bcs.DirichletBC],
        prediction_CNs: list[CrankNicolson],
        subdomain_CNs: list[CrankNicolson2],
    ):
        qn_i_plus_list = []
        pn_i_plus_list = []
        """
        Step 1: prediction
        """
        for i, Omega_i_delta in enumerate(self.SDs):
            loc_qn = Omega_i_delta.prediction_subdomain.restrict_to_local(qn)
            loc_pn = Omega_i_delta.prediction_subdomain.restrict_to_local(pn)
            q_bc1 = dfx.fem.dirichletbc(
                loc_qn,
                Omega_i_delta.prediction_subdomain.interface_dofs,  # bc condition matching qn
            )
            if len(Omega_i_delta.prediction_subdomain.parent_boundary_dofs) > 0:
                q_D = dfx.fem.Function(Omega_i_delta.prediction_subdomain.V)
                if self.prediction_problems[i].u_bc is not None:
                    q_D.interpolate(
                        self.prediction_problems[i].dfExpression(
                            self.prediction_problems[i].u_bc(tn + tau),
                            Omega_i_delta.prediction_subdomain.V,
                        )
                    )
                q_bc2 = dfx.fem.dirichletbc(
                    q_D, Omega_i_delta.prediction_subdomain.parent_boundary_dofs
                )
                prediction_bcs = [q_bc1, q_bc2]
            else:
                prediction_bcs = [q_bc1]

            pred_i, pred_p_i = prediction_CNs[i].step(
                tn,
                tau,
                loc_qn,
                loc_pn,
                prediction_bcs,
            )

            pred = dfx.fem.Function(Omega_i_delta.V)
            pred.x.array[Omega_i_delta.interface_dofs] = pred_i.x.array[
                Omega_i_delta.prediction_subdomain.prediction_dofs
            ]

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

        prediction_CNs = [
            CrankNicolson(
                self.T,
                self.tau,
                Omega_i_delta.prediction_subdomain,
                self.SpaceDiscretizationClass,
                self.prediction_problems[i],
            )
            for i, Omega_i_delta in enumerate(self.SDs)
        ]
        for pCN in prediction_CNs:
            pCN.assemble_space_discretization()

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
                prediction_CNs,
                subdomains_CNs,
            )

            qn.x.array[:] = qn_plus_1.x.array
            pn.x.array[:] = pn_plus_1.x.array

        return qn, pn
