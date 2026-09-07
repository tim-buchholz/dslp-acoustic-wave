import numpy as np
import dolfinx as dfx
import dolfinx.fem.petsc
import ufl
from ProblemDataUFL import ProblemData
from Domain import Domain
from SpaceDiscretization import DG
from abc import ABC, abstractmethod
from typing import Type, Optional
from petsc4py import PETSc
from Config import (
    DIRECT,
    ITERATIVE,
    ATI_DG_STR,
    CN_DG_STR,
    LF_DG_STR,
    FEM_TYPE,
    FEM_DEGREE,
)
from Projections import Ritz_projection_initial_values
from MeshIO import save_vtk, save_xdmf, write_hdf5, write_adios_function
from DolfinVersion import PETSC_PATH


class TimeIntegratorWaveDG(ABC):
    name = ATI_DG_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[DG],
        problem: ProblemData,
    ) -> None:
        self.T = T
        self.tau = tau
        self.N = int(round(T / tau))
        assert np.isclose(
            self.N * self.tau, self.T, rtol=0.0, atol=1e-12, equal_nan=False
        )
        f"step size {tau} doesn't fit to {T}"
        self.Omega = Omega
        self.V = Omega.V
        self.SpaceDiscretizationClass = SpaceDiscretizationClass
        self.space_discretization = self.SpaceDiscretizationClass()
        self.problem = problem
        self.tau_ufl = dfx.fem.Constant(self.V.mesh, self.tau)
        self.assemble_space_discretization()

    def __repr__(self) -> str:
        problem_number = self.problem.example_number
        wave_propagation_speed = self.problem.c
        domain_type = self.Omega.__name__
        space_discretization = type(self.space_discretization).__name__
        fem_type = FEM_TYPE + "-deg" + str(FEM_DEGREE)
        return f"TimeIntegrator={type(self).__name__}::problem={problem_number}::c={wave_propagation_speed}::domain={domain_type}::sd={space_discretization}::fem={fem_type}::tau={self.tau}::T={self.T}"

    def startup(self, ritz_projection=False):
        self.start_index = 0
        if ritz_projection:
            qn, pn = Ritz_projection_initial_values(self.problem, self.Omega)
        else:
            qn = dfx.fem.Function(self.V)
            qn.interpolate(self.problem.dfExpression(self.problem.u0, self.V))
            pn = dfx.fem.Function(self.V)
            pn.interpolate(self.problem.dfExpression(self.problem.v0, self.V))
        return qn, pn

    @abstractmethod
    def assemble_space_discretization(self):
        pass

    @abstractmethod
    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        g_bc: Optional[dfx.fem.function.Function] = None,
    ):
        pass

    def integrate(self, startup_ritz_projection: bool = False):
        qn, pn = self.startup(ritz_projection=startup_ritz_projection)
        tn = 0.0
        for n in range(self.start_index, self.N):
            tn = n * self.tau
            qn_plus_1, pn_plus_1 = self.step(tn, self.tau, qn, pn, g_bc=None)

            qn.x.array[:] = qn_plus_1.x.array
            pn.x.array[:] = pn_plus_1.x.array
        return qn, pn

    def save_vtk(self, uh: dfx.fem.function.Function, name="output", t: float = 0.0):
        save_vtk(self.V, uh, name, t)

    def save_xdmf(self, uh: dfx.fem.function.Function, name: str = "output"):
        save_xdmf(self.V, uh, name)

    def save_hdf5(
        self,
        uh: dfx.fem.function.Function,
        V: dfx.fem.function.FunctionSpace,
        name="output",
        t: float = 0.0,
    ):
        write_hdf5(name, V, uh, t)

    def save_adios(
        self,
        uh: dfx.fem.function.Function,
        V: dfx.fem.function.FunctionSpace,
        name="output",
        t: float = 0.0,
        function_name: str = "f",
    ):
        write_adios_function(name, V, uh, t, function_name)


class CrankNicolsonDG(TimeIntegratorWaveDG):
    name = CN_DG_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[DG],
        problem: ProblemData,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)

        phi = ufl.TestFunction(self.V)
        nF = ufl.FacetNormal(self.V.mesh)

        self.F = self.problem.f  # a time dependent functor
        self.F_tn = self.F.copy()  # ufl expression 1 in rhs form
        self.f_tn = self.F_tn(0.0)
        self.F_tn_p = self.F.copy()  # ufl expression 2 in rhs form
        self.f_tn_p = self.F_tn_p(self.tau)

        if self.problem.u_bc is not None:
            self.UD_p = self.problem.u_bc.copy()
            self.u_D_p = self.UD_p(self.tau)
        else:
            self.u_D_p = dfx.fem.Constant(self.V.mesh, 0.0)

        self.g_bc = dfx.fem.Function(
            self.V
        )  # g_bc gives the opportunity to overwrite u_D_p
        self.g_bc.x.array[:] = 0.0

        self.qn = dfx.fem.Function(self.V)
        self.pn = dfx.fem.Function(self.V)

        _m_f = (
            self.pn * phi + self.tau_ufl / 2 * (self.f_tn + self.f_tn_p) * phi
        ) * self.space_discretization.mass_measure

        avg_qn = ufl.avg(self.problem.c**2 * ufl.grad(self.qn))
        avg_pn = ufl.avg(self.problem.c**2 * ufl.grad(self.pn))
        avg_phi = ufl.avg(self.problem.c**2 * ufl.grad(phi))

        _a_cell_qn = (
            self.problem.c**2
            * ufl.dot(ufl.grad(self.qn), ufl.grad(phi))
            * self.space_discretization.stiffness_measure
        )  # (A)

        _a_cell_pn = (
            self.problem.c**2
            * ufl.dot(ufl.grad(self.pn), ufl.grad(phi))
            * self.space_discretization.stiffness_measure
        )  # (A)

        _a_iface_qn = (
            -ufl.dot(avg_qn, ufl.jump(phi, nF)) * ufl.dS  # (B)
            - ufl.dot(avg_phi, ufl.jump(self.qn, nF)) * ufl.dS  # (C inner faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFi
            )
            * ufl.dot(ufl.jump(phi, nF), ufl.jump(self.qn, nF))
            * ufl.dS  # (D inner faces)
        )

        _a_iface_pn = (
            -ufl.dot(avg_pn, ufl.jump(phi, nF)) * ufl.dS  # (B)
            - ufl.dot(avg_phi, ufl.jump(self.pn, nF)) * ufl.dS  # (C inner faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFi
            )
            * ufl.dot(ufl.jump(phi, nF), ufl.jump(self.pn, nF))
            * ufl.dS  # (D inner faces)
        )

        _a_bface_qn = (
            -self.problem.c**2
            * ufl.dot(ufl.grad(self.qn), phi * nF)
            * ufl.ds  # (B boundary faces)
            - self.problem.c**2
            * ufl.dot(ufl.grad(phi), self.qn * nF)
            * ufl.ds  # (C boundary faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFb
            )
            * self.qn
            * phi
            * ufl.ds  #  (D boundary faces)
        )

        _a_bface_pn = (
            -self.problem.c**2
            * ufl.dot(ufl.grad(self.pn), phi * nF)
            * ufl.ds  # (B boundary faces)
            - self.problem.c**2
            * ufl.dot(ufl.grad(phi), self.pn * nF)
            * ufl.ds  # (C boundary faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFb
            )
            * self.pn
            * phi
            * ufl.ds  #  (D boundary faces)
        )

        _bc_term = (
            -self.problem.c**2
            * ufl.dot(
                ufl.grad(phi), (self.u_D_p + self.g_bc + self.qn) * nF
            )  # either g_bc or u_D_p is zero
            * ufl.ds  #  (like C boundary faces)
            + (self.space_discretization.eta_S * self.problem.c**2)
            / self.space_discretization.hFb  #  (like D boundary faces)
            * (self.u_D_p + self.g_bc + self.qn)  # either g_bc or u_D_p is zero
            * phi
            * ufl.ds
        )

        self.rhs_form = dfx.fem.form(
            _m_f
            - self.tau_ufl**2 / 4 * (_a_cell_pn + _a_iface_pn + _a_bface_pn)
            - self.tau_ufl * (_a_cell_qn + _a_iface_qn + _a_bface_qn)
            + self.tau_ufl / 2 * (_bc_term)  # treat boundary term as the inhomogeneity
        )

        self.b = PETSC_PATH.create_vector(self.V)

    def assemble_space_discretization(self) -> None:
        self.space_discretization.assemble(self.V, self.problem.c)
        factor_before_StiffnessMatrix = self.tau**2 / 4
        self.space_discretization.set_System(factor_before_StiffnessMatrix, self.V)

    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        g_bc: Optional[dfx.fem.function.Function] = None,
    ):
        self.F_tn.t.value = tn
        self.F_tn_p.t.value = tn + tau
        if g_bc is not None:
            self.u_D_p = dfx.fem.Constant(self.V.mesh, 0.0)
            self.g_bc.x.array[:] = g_bc.x.array[:]
        elif self.problem.u_bc is not None:
            self.UD_p.t.value = tn + tau
        qn_p = dfx.fem.Function(self.V)
        pn_p = dfx.fem.Function(self.V)
        self.qn.x.array[:] = qn.x.array  # do we need this copy?
        self.pn.x.array[:] = pn.x.array  # do we need this copy?

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.rhs_form)
        # solve system for p^{n+1}
        self.space_discretization.solverSystem.solve(self.b, pn_p.x.petsc_vec)
        # explicit update for q^{n+1}
        qn_p.x.array[:] = qn.x.array + self.tau / 2 * (pn_p.x.array + pn.x.array)

        return qn_p, pn_p


class leapfrogDG(TimeIntegratorWaveDG):
    name = LF_DG_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[DG],
        problem: ProblemData,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)

        phi = ufl.TestFunction(self.V)
        nF = ufl.FacetNormal(self.V.mesh)

        self.F = self.problem.f  # a time dependent functor
        self.F_tn = self.F.copy()  # ufl expression 1 in rhs form
        self.f_tn = self.F_tn(0.0)
        if self.problem.u_bc is not None:
            self.UD = self.problem.u_bc.copy()
            self.u_D = self.UD(0.0)
        else:
            self.u_D = dfx.fem.Constant(self.V.mesh, 0.0)

        self.g_bc = dfx.fem.Function(self.V)
        self.g_bc.x.array[:] = 0.0
        self.qn = dfx.fem.Function(self.V)
        self.pn = dfx.fem.Function(self.V)

        _m_f = (
            self.pn * phi + self.tau_ufl / 2 * (self.f_tn) * phi
        ) * self.space_discretization.mass_measure

        avg_qn = ufl.avg(self.problem.c**2 * ufl.grad(self.qn))
        avg_phi = ufl.avg(self.problem.c**2 * ufl.grad(phi))

        _a_cell_qn = (
            self.problem.c**2
            * ufl.dot(ufl.grad(self.qn), ufl.grad(phi))
            * self.space_discretization.stiffness_measure
        )  # (A)

        _a_iface_qn = (
            -ufl.dot(avg_qn, ufl.jump(phi, nF)) * ufl.dS  # (B)
            - ufl.dot(avg_phi, ufl.jump(self.qn, nF)) * ufl.dS  # (C inner faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFi
            )
            * ufl.dot(ufl.jump(phi, nF), ufl.jump(self.qn, nF))
            * ufl.dS  # (D inner faces)
        )

        _a_bface_qn = (
            -self.problem.c**2
            * ufl.dot(ufl.grad(self.qn), phi * nF)
            * ufl.ds  # (B boundary faces)
            - self.problem.c**2
            * ufl.dot(ufl.grad(phi), self.qn * nF)
            * ufl.ds  # (C boundary faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFb
            )
            * self.qn
            * phi
            * ufl.ds  #  (D boundary faces)
        )

        _bc_term = (
            -self.problem.c**2
            * ufl.dot(ufl.grad(phi), self.u_D * nF)
            * ufl.ds  #  (like C boundary faces)
            + (self.space_discretization.eta_S * self.problem.c**2)
            / self.space_discretization.hFb  #  (like D boundary faces)
            * (self.u_D + self.g_bc)  # either u_D or g_bc should be zero
            * phi
            * ufl.ds
        )

        self.rhs_form = dfx.fem.form(
            _m_f
            - self.tau_ufl / 2 * (_a_cell_qn + _a_iface_qn + _a_bface_qn)
            + self.tau_ufl / 2 * _bc_term  # treat boundary term as the inhomogeneity
        )

        self.b = PETSC_PATH.create_vector(self.V)

    def assemble_space_discretization(self) -> None:
        self.space_discretization.assemble(self.V, self.problem.c)

    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        g_bc: Optional[dfx.fem.function.Function] = None,
    ):
        qn_p = dfx.fem.Function(self.V)
        pn_p = dfx.fem.Function(self.V)
        self.qn.x.array[:] = qn.x.array  # do we need this copy?
        self.pn.x.array[:] = pn.x.array  # do we need this copy?

        # halfstep for p^{n+1/2}
        self.F_tn.t.value = tn
        if g_bc is not None:
            self.u_D = dfx.fem.Constant(self.V.mesh, 0.0)
            self.g_bc.x.array[:] = qn.x.array
        elif self.problem.u_bc is not None:
            self.UD.t.value = tn
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.rhs_form)
        self.space_discretization.solverM.solve(self.b, pn_p.x.petsc_vec)
        self.pn.x.array[:] = pn_p.x.array
        # explicit update for q^{n+1}
        qn_p.x.array[:] = self.qn.x.array + self.tau * self.pn.x.array
        self.qn.x.array[:] = qn_p.x.array
        # second halfstep for p^{n+1}
        self.F_tn.t.value = tn + tau
        if g_bc is not None:
            self.g_bc.x.array[:] = g_bc.x.array
        elif self.problem.u_bc is not None:
            self.UD.t.value = tn + tau
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.rhs_form)
        self.space_discretization.solverM.solve(self.b, pn_p.x.petsc_vec)

        return qn_p, pn_p


class leapfrogDG2(TimeIntegratorWaveDG):
    name = LF_DG_STR + "2"

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[DG],
        problem: ProblemData,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)

        phi = ufl.TestFunction(self.V)
        nF = ufl.FacetNormal(self.V.mesh)

        self.F = self.problem.f  # a time dependent functor
        self.F_tn = self.F.copy()  # ufl expression 1 in rhs form
        self.f_tn = self.F_tn(0.0)
        self.F_tn_p = self.F.copy()  # ufl expression 2 in rhs form
        self.f_tn_p = self.F_tn_p(self.tau)
        if self.problem.u_bc is not None:
            self.UD_p = self.problem.u_bc.copy()
            self.u_D_p = self.UD_p(self.tau)
        else:
            self.u_D_p = dfx.fem.Constant(self.V.mesh, 0.0)
        self.g_bc = dfx.fem.Function(self.V)
        self.g_bc.x.array[:] = 0.0

        self.qn = dfx.fem.Function(self.V)
        self.pn = dfx.fem.Function(self.V)

        _m_f = (
            self.qn * phi
            + self.tau_ufl * self.pn * phi
            + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
        ) * self.space_discretization.mass_measure

        avg_qn = ufl.avg(self.problem.c**2 * ufl.grad(self.qn))
        avg_pn = ufl.avg(self.problem.c**2 * ufl.grad(self.pn))
        avg_phi = ufl.avg(self.problem.c**2 * ufl.grad(phi))

        _a_cell_qn = (
            self.problem.c**2
            * ufl.dot(ufl.grad(self.qn), ufl.grad(phi))
            * self.space_discretization.stiffness_measure
        )  # (A)

        _a_cell_pn = (
            self.problem.c**2
            * ufl.dot(ufl.grad(self.pn), ufl.grad(phi))
            * self.space_discretization.stiffness_measure
        )  # (A)

        _a_iface_qn = (
            -ufl.dot(avg_qn, ufl.jump(phi, nF)) * ufl.dS  # (B)
            - ufl.dot(avg_phi, ufl.jump(self.qn, nF)) * ufl.dS  # (C inner faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFi
            )
            * ufl.dot(ufl.jump(phi, nF), ufl.jump(self.qn, nF))
            * ufl.dS  # (D inner faces)
        )

        _a_iface_pn = (
            -ufl.dot(avg_pn, ufl.jump(phi, nF)) * ufl.dS  # (B)
            - ufl.dot(avg_phi, ufl.jump(self.pn, nF)) * ufl.dS  # (C inner faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFi
            )
            * ufl.dot(ufl.jump(phi, nF), ufl.jump(self.pn, nF))
            * ufl.dS  # (D inner faces)
        )

        _a_bface_qn = (
            -self.problem.c**2
            * ufl.dot(ufl.grad(self.qn), phi * nF)
            * ufl.ds  # (B boundary faces)
            - self.problem.c**2
            * ufl.dot(ufl.grad(phi), self.qn * nF)
            * ufl.ds  # (C boundary faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFb
            )
            * self.qn
            * phi
            * ufl.ds  #  (D boundary faces)
        )

        _a_bface_pn = (
            -self.problem.c**2
            * ufl.dot(ufl.grad(self.pn), phi * nF)
            * ufl.ds  # (B boundary faces)
            - self.problem.c**2
            * ufl.dot(ufl.grad(phi), self.pn * nF)
            * ufl.ds  # (C boundary faces)
            + (
                self.space_discretization.eta_S
                * self.problem.c**2
                / self.space_discretization.hFb
            )
            * self.pn
            * phi
            * ufl.ds  #  (D boundary faces)
        )

        _bc_term = (
            -self.problem.c**2
            * ufl.dot(ufl.grad(phi), (self.u_D_p + self.g_bc + self.qn) * nF)
            * ufl.ds  #  (like C boundary faces)
            + (self.space_discretization.eta_S * self.problem.c**2)
            / self.space_discretization.hFb  #  (like D boundary faces)
            * (self.u_D_p + self.g_bc + self.qn)
            * phi
            * ufl.ds
        )

        self.rhs_form = dfx.fem.form(
            _m_f
            - self.tau_ufl**3 / 4 * (_a_cell_pn + _a_iface_pn + _a_bface_pn)
            - self.tau_ufl**2 / 2 * (_a_cell_qn + _a_iface_qn + _a_bface_qn)
            + self.tau_ufl**2 / 4 * _bc_term  # treat boundary term as the inhomogeneity
        )

        self.b = PETSC_PATH.create_vector(self.V)

    def assemble_space_discretization(self) -> None:
        self.space_discretization.assemble(self.V, self.problem.c)

    def step(
        self,
        tn: float,
        tau: float,
        qn: dfx.fem.function.Function,
        pn: dfx.fem.function.Function,
        g_bc: Optional[dfx.fem.function.Function] = None,
    ):
        self.F_tn.t.value = tn
        self.F_tn_p.t.value = tn + tau
        if g_bc is not None:
            self.u_D_p = dfx.fem.Constant(self.V.mesh, 0.0)
            self.g_bc.x.array[:] = g_bc.x.array[:]
        elif self.problem.u_bc is not None:
            self.UD_p.t.value = tn + tau
        qn_p = dfx.fem.Function(self.V)
        pn_p = dfx.fem.Function(self.V)
        self.qn.x.array[:] = qn.x.array  # do we need this copy?
        self.pn.x.array[:] = pn.x.array  # do we need this copy?

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.rhs_form)
        # solve system for q^{n+1}
        self.space_discretization.solverM.solve(self.b, qn_p.x.petsc_vec)
        # explicit update for p^{n+1}
        pn_p.x.array[:] = 2.0 / tau * (qn_p.x.array[:] - qn.x.array[:]) - pn.x.array[:]

        return qn_p, pn_p
