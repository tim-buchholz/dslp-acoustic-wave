import math
from typing import Type

import dolfinx as dfx
import numpy as np
import ufl

from Config import DSTLP_STR
from DolfinVersion import DIRICHLET_BC_TYPE, PETSC_PATH
from Domain import DS_reg, OvSubDomain
from DomainSplitting import DomainSplitting
from ProblemDataUFL import ProblemData
from SpaceDiscretization import SpaceDiscretization
from TimeIntegration import CrankNicolson


def binary_inner_strip_cutoff(
    subdomain: OvSubDomain,
    ell_p1: int,
) -> dfx.fem.function.Function:
    if subdomain.fem_type == "DG":
        inner_dofs_global = subdomain.get_dofs_around_interface_DG(layers=ell_p1)
    else:
        inner_dofs_global = subdomain.get_dofs_around_interface(layers=ell_p1)

    prediction_subdomain = subdomain.prediction_subdomain
    inner_dofs_local = prediction_subdomain.global_dof_to_local_dof(
        inner_dofs_global
    )
    cutoff = dfx.fem.Function(prediction_subdomain.V)
    cutoff.x.array[inner_dofs_local] = 1.0
    return cutoff


class TestLocalizedCrankNicolsonIncrement(CrankNicolson):
    def __init__(
        self,
        T: float,
        tau: float,
        Omega,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
        cutoff: dfx.fem.function.Function,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)
        self.cutoff = cutoff

        phi = ufl.TestFunction(self.V)
        self.increment_rhs_form = dfx.fem.form(
            (
                self.tau_ufl * self.pn * phi
                + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
            )
            * self.space_discretization.mass_measure
            - (
                self.tau_ufl**2
                / 2
                * ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
            )
            * self.space_discretization.stiffness_measure
        )
        self.zero_increment = dfx.fem.Function(self.V)
        self.increment_bc = dfx.fem.dirichletbc(
            self.zero_increment, self.Omega.boundary_dofs
        )

    def assemble_space_discretization(self) -> None:
        self.space_discretization.assemble(self.V, self.problem.c, [self.increment_bc])
        self.space_discretization.set_System(
            self.tau**2 / 4, self.problem.c, self.V, [self.increment_bc]
        )

    def _update_increment_rhs_form_if_needed(self, tn: float, tau: float) -> None:
        self.F_tn.t.value = tn
        self.F_tn_p.t.value = tn + tau
        if not self.F.reevaluation:
            return

        self.f_tn = self.F_tn(tn)
        self.f_tn_p = self.F_tn_p(tn + tau)
        phi = ufl.TestFunction(self.V)
        self.increment_rhs_form = dfx.fem.form(
            (
                self.tau_ufl * self.pn * phi
                + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
            )
            * self.space_discretization.mass_measure
            - (
                self.tau_ufl**2
                / 2
                * ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
            )
            * self.space_discretization.stiffness_measure
        )

    def assemble_localized_rhs(self, tn: float, tau: float, qn, pn):
        self._update_increment_rhs_form_if_needed(tn, tau)
        self.qn.x.array[:] = qn.x.array
        self.pn.x.array[:] = pn.x.array

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.increment_rhs_form)
        self.b.array[:] *= self.cutoff.x.array[:]
        self.space_discretization.apply_bc_rhs(
            self.b, [self.increment_bc], self.space_discretization.system_form
        )
        return self.b

    def step(self, tn, tau, qn, pn, bc=None):
        qn_p = dfx.fem.Function(self.V)
        increment = dfx.fem.Function(self.V)
        rhs = self.assemble_localized_rhs(tn, tau, qn, pn)
        self.space_discretization.solverSystem.solve(rhs, increment.x.petsc_vec)
        qn_p.x.array[:] = qn.x.array + increment.x.array
        pn_p = dfx.fem.Function(self.V)
        pn_p.x.array[:] = 2.0 / tau * increment.x.array - pn.x.array
        return qn_p, pn_p


class DSTLP(DomainSplitting):
    name = DSTLP_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: DS_reg,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
        ell_p1: int | None = None,
        ell_p2: int | None = None,
        minimal_pred_ells: tuple[int, int] = (2, 2),
        gamma: float = 1.0,
    ) -> None:
        if len(minimal_pred_ells) != 2:
            raise ValueError("minimal_pred_ells must contain two entries")
        if minimal_pred_ells[0] < 1 or minimal_pred_ells[1] < 0:
            raise ValueError("minimal_pred_ells must be at least (1, 0)")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if ell_p1 is not None and ell_p1 < 1:
            raise ValueError("ell_p1 must be at least 1")
        if ell_p2 is not None and ell_p2 < 0:
            raise ValueError("ell_p2 must be non-negative")
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)
        self.ell_p1 = ell_p1
        self.ell_p2 = ell_p2
        self.minimal_pred_ells = minimal_pred_ells
        self.gamma = gamma
        self.SDs: list[OvSubDomain] = self.Omega.SDs
        self.numSDs: int = len(self.SDs)
        self.problems: list[ProblemData] = [
            ProblemData(
                self.problem.example_number, Omega_i_delta.V, self.problem.c_init
            )
            for Omega_i_delta in self.SDs
        ]

        self.prediction_patch_layers: list[int] = []
        self.prediction_inner_layers: list[int] = []
        self.prediction_outer_layers: list[int] = []
        for subdomain in self.SDs:
            hmin = subdomain.get_h_min()
            heuristic_layers = math.ceil(gamma * self.tau * self.problem.c / hmin)
            inner_layers = (
                max(minimal_pred_ells[0], heuristic_layers)
                if ell_p1 is None
                else ell_p1
            )
            outer_layers = (
                max(minimal_pred_ells[1], heuristic_layers)
                if ell_p2 is None
                else ell_p2
            )
            total_layers = inner_layers + outer_layers
            subdomain.generate_prediction_subdomain(layers=total_layers)
            self.prediction_inner_layers.append(inner_layers)
            self.prediction_outer_layers.append(outer_layers)
            self.prediction_patch_layers.append(total_layers)

        self.prediction_cutoffs = [
            binary_inner_strip_cutoff(subdomain, self.prediction_inner_layers[i])
            for i, subdomain in enumerate(self.SDs)
        ]
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
        prediction_CNs: list[TestLocalizedCrankNicolsonIncrement],
        subdomain_CNs: list[CrankNicolson],
    ):
        qn_i_plus_list = []
        pn_i_plus_list = []

        for i, Omega_i_delta in enumerate(self.SDs):
            loc_qn = Omega_i_delta.prediction_subdomain.restrict_to_local(qn)
            loc_pn = Omega_i_delta.prediction_subdomain.restrict_to_local(pn)

            pred_i, _ = prediction_CNs[i].step(tn, tau, loc_qn, loc_pn)

            pred = dfx.fem.Function(Omega_i_delta.V)
            pred.x.array[Omega_i_delta.interface_dofs] = pred_i.x.array[
                Omega_i_delta.prediction_subdomain.prediction_dofs
            ]

            predicted_bc_i = dfx.fem.dirichletbc(
                pred,
                Omega_i_delta.interface_dofs,
            )

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

        qn_p = self.Omega.avg(*qn_i_plus_list)
        pn_p = self.Omega.avg(*pn_i_plus_list)

        return qn_p, pn_p

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
            TestLocalizedCrankNicolsonIncrement(
                self.T,
                self.tau,
                Omega_i_delta.prediction_subdomain,
                self.SpaceDiscretizationClass,
                self.prediction_problems[i],
                self.prediction_cutoffs[i],
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
