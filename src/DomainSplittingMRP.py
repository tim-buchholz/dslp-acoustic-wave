import numpy as np
import dolfinx as dfx
from ProblemDataUFL import ProblemData
from SpaceDiscretization import SpaceDiscretization
from TimeIntegration import TimeIntegratorWave, CrankNicolson, leapfrog
from DomainSplitting import DomainSplitting
from Domain import Domain, SubDomain, OvSubDomain, DS_reg
from typing import Type, Tuple
from Config import DSMPR_STR
from Projections import Ritz_projection_initial_values
from DolfinVersion import DIRICHLET_BC_TYPE

class DomainSplittingMRP(DomainSplitting):
    name = DSMPR_STR
    _default_relative_rate = 8

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: DS_reg,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
        relative_rate: int = _default_relative_rate,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)
        self.SDs: list[OvSubDomain] = self.Omega.SDs
        self.numSDs: int = len(self.SDs)
        self.problems: list[ProblemData] = [
            ProblemData(self.problem.example_number, Omega_i_delta.V, self.problem.c_init)
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
        self.relative_rate: int = relative_rate

    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        subdomain_bcs: list[DIRICHLET_BC_TYPE],
        subdomain_lfs: list[leapfrog],
        subdomain_CNs: list[CrankNicolson],
        rates: list[int],
    ):
        qn_i_plus_list = []
        pn_i_plus_list = []
        """
        Step 1: prediction
        """
        for i, Omega_i_delta in enumerate(self.SDs):
            q_D = dfx.fem.Function(self.Omega.V)
            q_bc = dfx.fem.dirichletbc(
                q_D, Omega_i_delta.prediction_subdomain.boundary_dofs
            )
            qn_loc = Omega_i_delta.prediction_subdomain.restrict_to_local(qn)
            pn_loc = Omega_i_delta.prediction_subdomain.restrict_to_local(pn)
            for m in range(rates[i]):
                mr_tau = tau / rates[i]
                qm, pm = subdomain_lfs[i].step(
                    tn + m * mr_tau,
                    mr_tau,
                    qn_loc,
                    pn_loc,
                    q_bc,
                )
                qn_loc.x.array[:] = qm.x.array
                pn_loc.x.array[:] = pm.x.array

            pred = dfx.fem.Function(Omega_i_delta.V)
            pred.x.array[Omega_i_delta.interface_dofs] = qn_loc.x.array[
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

    def integrate(
        self, startup_ritz_projection=False, relative_rate=_default_relative_rate
    ):
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

        if relative_rate >= 1:
            rates = [relative_rate] * self.numSDs
        else:
            rates = [
                max(int(round(self.tau / subdomain.get_h_min())), 1)
                for subdomain in self.SDs
            ]

        subdomain_lfs = [
            leapfrog(
                self.T,
                self.tau / rates[i],
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
                rates,
            )

            qn.x.array[:] = qn_plus_1.x.array
            pn.x.array[:] = pn_plus_1.x.array

        return qn, pn
