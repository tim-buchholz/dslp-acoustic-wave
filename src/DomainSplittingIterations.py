import numpy as np
import dolfinx as dfx
from ProblemDataUFL import ProblemData
from SpaceDiscretization import SpaceDiscretization
from TimeIntegration import TimeIntegratorWave, CrankNicolson, leapfrog
from Domain import Domain, SubDomain, OvSubDomain, DS_reg
from typing import Type, Tuple
from Config import DS_STR
from Projections import Ritz_projection_initial_values
from DolfinVersion import DIRICHLET_BC_TYPE


class DomainSplittingIterations(TimeIntegratorWave):
    name = "RAS timestepping"

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: DS_reg,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
        N_iterations: int = 4,
        K_local_steps: int = 1,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)
        self.N_iterations: int = N_iterations
        self.K_local_steps: int = K_local_steps
        self.SDs: list[OvSubDomain] = self.Omega.SDs
        self.numSDs: int = len(self.SDs)
        self.problems: list[ProblemData] = [
            ProblemData(
                self.problem.example_number, Omega_i_delta.V, self.problem.c_init
            )
            for Omega_i_delta in self.SDs
        ]
        for subdomain in self.SDs:
            subdomain.generate_prediction_subdomain(layers=K_local_steps)
        self.prediction_problems: list[ProblemData] = [
            ProblemData(
                self.problem.example_number,
                Omega_i_delta.prediction_subdomain.V,
                self.problem.c_init,
            )
            for Omega_i_delta in self.SDs
        ]

    def step(self, tn, tau, qn, pn, bc):
        raise NotImplementedError("Call multi_step instead")

    def multi_step(self, tn, tau, num_steps, qn, pn, subdomain_lfs, subdomain_CNs):

        predictions = [[] for i in range(num_steps)]
        qlf_npk = [
            Omega_i_delta.prediction_subdomain.restrict_to_local(qn)
            for Omega_i_delta in self.SDs
        ]
        plf_npk = [
            Omega_i_delta.prediction_subdomain.restrict_to_local(pn)
            for Omega_i_delta in self.SDs
        ]
        for k in range(num_steps):
            tnk = tn + k * tau
            """
            Step 1: prediction
            """
            for i, Omega_i_delta in enumerate(self.SDs):

                q_D = dfx.fem.Function(self.Omega.V)
                q_bc = dfx.fem.dirichletbc(
                    q_D, Omega_i_delta.prediction_subdomain.boundary_dofs
                )
                pred_i, pred_p_i = subdomain_lfs[i].step(
                    tnk,
                    tau,
                    qlf_npk[i],
                    plf_npk[i],
                    q_bc,
                )

                qlf_npk[i].x.array[:] = pred_i.x.array
                plf_npk[i].x.array[:] = pred_p_i.x.array

                pred = dfx.fem.Function(Omega_i_delta.V)
                pred.x.array[Omega_i_delta.interface_dofs] = pred_i.x.array[
                    Omega_i_delta.prediction_subdomain.prediction_dofs
                ]

                predicted_bc_i = dfx.fem.dirichletbc(
                    pred,
                    Omega_i_delta.interface_dofs,
                )
                predictions[k].append(predicted_bc_i)

        for iteration in range(self.N_iterations):
            qCNi_npk = [
                Omega_i_delta.restrict_to_local(qn) for Omega_i_delta in self.SDs
            ]
            pCNi_npk = [
                Omega_i_delta.restrict_to_local(pn) for Omega_i_delta in self.SDs
            ]
            for k in range(num_steps):
                tnk = tn + k * tau
                subdomain_bcs = []
                for i, Omega_i_delta in enumerate(self.SDs):
                    u_D = dfx.fem.Function(Omega_i_delta.V)
                    if self.problem.u_bc is not None:
                        u_D.interpolate(
                            self.problem.dfExpression(
                                self.problems[i].u_bc(tnk + self.tau), self.SDs[i].V
                            )
                        )
                    sd_bc = dfx.fem.dirichletbc(u_D, Omega_i_delta.parent_boundary_dofs)
                    subdomain_bcs.append(sd_bc)

                    """
                    Step 2: local calculation
                    """
                qnpk_i_plus_list = []
                pnpk_i_plus_list = []
                for i, Omega_i_delta in enumerate(self.SDs):
                    subdomain_sol = subdomain_CNs[i].step(
                        tnk,
                        tau,
                        qCNi_npk[i],
                        pCNi_npk[i],
                        [subdomain_bcs[i], predictions[k][i]],
                    )

                    qCNi_npk[i].x.array[:] = subdomain_sol[0].x.array
                    pCNi_npk[i].x.array[:] = subdomain_sol[1].x.array

                    qnpk_i_plus_list.append(qCNi_npk[i])
                    pnpk_i_plus_list.append(pCNi_npk[i])

                """
                step 3: averaging (later we do not want to to that in every sub-time step, but we need to think about how to get the correct prediction values then)
                this means it has to move out of k loop 
                and  qnpk_i_plus_lists,pnpk_i_plus_list have to be gathered over k
                """
                q_npk = self.Omega.avg(*qnpk_i_plus_list)
                p_npk = self.Omega.avg(*pnpk_i_plus_list)
                if iteration < self.N_iterations - 1:
                    for i, Omega_i_delta in enumerate(self.SDs):
                        pred_i_npk = Omega_i_delta.restrict_to_local(q_npk)
                        predictions[k][i] = dfx.fem.dirichletbc(
                            pred_i_npk,
                            Omega_i_delta.interface_dofs,
                        )
                else:
                    return q_npk, p_npk

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

        n = self.start_index

        while n < self.N:
            tn = n * self.tau
            # try to do K_local_steps, but do maximal N-n steps
            num_steps = min(self.K_local_steps, self.N - n)

            qn_plus_k, pn_plus_k = self.multi_step(
                tn,
                self.tau,
                num_steps,
                qn,
                pn,
                subdomain_lfs,
                subdomains_CNs,
            )

            qn.x.array[:] = qn_plus_k.x.array
            pn.x.array[:] = pn_plus_k.x.array

            if n + self.K_local_steps < self.N:
                n += self.K_local_steps
            else:
                n = self.N

        return qn, pn
