import numpy as np
import dolfinx as dfx
import dolfinx.fem.petsc
import ufl
from ProblemDataUFL import ProblemData
from Domain import Domain
from SpaceDiscretization import SpaceDiscretization, FEM, FEM_ml
from abc import ABC, abstractmethod
from typing import Type
from petsc4py import PETSc
from Config import DIRECT, ITERATIVE, ATI_STR, CN_STR, LF_STR, FEM_DEGREE, FEM_TYPE
from Projections import Ritz_projection_initial_values
from MeshIO import save_vtk, save_xdmf, write_hdf5, read_hdf5, write_adios_function
from DolfinVersion import PETSC_PATH


class TimeIntegratorWave(ABC):
    name = ATI_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
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
        u_D = dfx.fem.Function(self.V)
        if self.problem.u_bc is not None:
            u_D.interpolate(self.problem.dfExpression(self.problem.u_bc(0.0), self.V))
        self.bc = self.Omega.DirichletBC(u_D)
        self.tau_ufl = dfx.fem.Constant(self.V.mesh, self.tau)

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
    def step(self, tn, tau, qn, pn, bc):
        pass

    def integrate(self, startup_ritz_projection=False):
        u_D = dfx.fem.Function(self.V)
        qn, pn = self.startup(ritz_projection=startup_ritz_projection)
        self.assemble_space_discretization()
        for n in range(self.start_index, self.N):
            tn = n * self.tau
            if self.problem.u_bc is not None:
                u_D.interpolate(
                    self.problem.dfExpression(self.problem.u_bc(tn + self.tau), self.V)
                )
                self.bc = self.Omega.DirichletBC(u_D)
            qn_plus_1, pn_plus_1 = self.step(tn, self.tau, qn, pn, self.bc)

            qn.x.array[:] = qn_plus_1.x.array
            pn.x.array[:] = pn_plus_1.x.array

        return qn, pn

    def save_vtk(self, uh: dfx.fem.function.Function, name="output", t: float = 0.0):
        save_vtk(self.V, uh, name, t)

    def save_xdmf(self, uh: dfx.fem.function.Function, name="output"):
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


class leapfrog(TimeIntegratorWave):
    name = LF_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)

        phi = ufl.TestFunction(self.V)
        self.F = self.problem.f  # a time dependent functor
        self.F_tn = self.F.copy()  # ufl expression 1 in rhs form
        self.f_tn = self.F_tn(0.0)
        self.F_tn_p = self.F.copy()  # ufl expression 2 in rhs form
        self.f_tn_p = self.F_tn_p(self.tau)
        self.qn = dfx.fem.Function(self.V)
        self.pn = dfx.fem.Function(self.V)
        self.rhs_form = dfx.fem.form(
            (
                self.qn * phi
                + self.tau_ufl * self.pn * phi
                + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
            )
            * self.space_discretization.mass_measure
            - (
                self.tau_ufl**2
                / 2
                * ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
                + self.tau_ufl**3
                / 4
                * ufl.inner(self.problem.c**2 * ufl.grad(self.pn), ufl.grad(phi))
            )
            * self.space_discretization.stiffness_measure
        )
        self.b = PETSC_PATH.create_vector(self.V)

    def assemble_space_discretization(self) -> None:
        self.space_discretization.assemble(self.V, self.problem.c, [self.bc])

    def step(self, tn, tau, qn, pn, bc):
        self.F_tn.t.value = tn
        self.F_tn_p.t.value = tn + tau
        if self.F.reevaluation:
            self.f_tn = self.F_tn(tn)
            self.f_tn_p = self.F_tn_p(tn + tau)
            phi = ufl.TestFunction(self.V)
            self.rhs_form = dfx.fem.form(
                (
                    self.qn * phi
                    + self.tau_ufl * self.pn * phi
                    + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
                )
                * self.space_discretization.mass_measure
                - (
                    self.tau_ufl**2
                    / 2
                    * ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
                    + self.tau_ufl**3
                    / 4
                    * ufl.inner(self.problem.c**2 * ufl.grad(self.pn), ufl.grad(phi))
                )
                * self.space_discretization.stiffness_measure
            )
        qn_p = dfx.fem.Function(self.V)
        pn_p = dfx.fem.Function(self.V)
        self.qn.x.array[:] = qn.x.array  # assign by reference?
        self.pn.x.array[:] = pn.x.array  # assign by reference?

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.rhs_form)
        bcs = bc if type(bc) == list else [bc]
        self.space_discretization.apply_bc_rhs(
            self.b, bcs, form=self.space_discretization.mass_form
        )
        self.space_discretization.solverM.solve(self.b, qn_p.x.petsc_vec)

        pn_p.x.petsc_vec[:] = (
            2.0 / tau * (qn_p.x.petsc_vec[:] - qn.x.petsc_vec[:]) - pn.x.petsc_vec[:]
        )

        return qn_p, pn_p


class CrankNicolson(TimeIntegratorWave):
    name = CN_STR

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)

        phi = ufl.TestFunction(self.V)
        self.F = self.problem.f  # a time dependent functor
        self.F_tn = self.F.copy()  # ufl expression 1 in rhs form
        self.f_tn = self.F_tn(0.0)
        self.F_tn_p = self.F.copy()  # ufl expression 2 in rhs form
        self.f_tn_p = self.F_tn_p(self.tau)
        self.qn = dfx.fem.Function(self.V)
        self.pn = dfx.fem.Function(self.V)
        self.rhs_form = dfx.fem.form(
            (
                self.qn * phi
                + self.tau_ufl * self.pn * phi
                + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
            )
            * self.space_discretization.mass_measure
            - (
                self.tau_ufl**2
                / 4
                * ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
            )
            * self.space_discretization.stiffness_measure
        )
        self.b = PETSC_PATH.create_vector(self.V)

    def assemble_space_discretization(self) -> None:
        self.space_discretization.assemble(self.V, self.problem.c, [self.bc])
        factor_before_StiffnessMatrix = self.tau**2 / 4
        self.space_discretization.set_System(
            factor_before_StiffnessMatrix, self.problem.c, self.V, [self.bc]
        )

    def step(self, tn, tau, qn, pn, bc):
        self.F_tn.t.value = tn
        self.F_tn_p.t.value = tn + tau
        if self.F.reevaluation:
            self.f_tn = self.F_tn(tn)
            self.f_tn_p = self.F_tn_p(tn + tau)
            phi = ufl.TestFunction(self.V)
            self.rhs_form = dfx.fem.form(
                (
                    self.qn * phi
                    + self.tau_ufl * self.pn * phi
                    + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
                )
                * self.space_discretization.mass_measure
                - (
                    self.tau_ufl**2
                    / 4
                    * ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
                )
                * self.space_discretization.stiffness_measure
            )
        qn_p = dfx.fem.Function(self.V)
        pn_p = dfx.fem.Function(self.V)
        self.qn.x.array[:] = qn.x.array  # assign by reference
        self.pn.x.array[:] = pn.x.array  # assign by reference

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.rhs_form)

        bcs = bc if type(bc) == list else [bc]
        self.space_discretization.apply_bc_rhs(
            self.b, bcs, self.space_discretization.system_form
        )
        self.space_discretization.solverSystem.solve(self.b, qn_p.x.petsc_vec)

        pn_p.x.array[:] = 2.0 / tau * (qn_p.x.array - qn.x.array) - pn.x.array

        return qn_p, pn_p


class CrankNicolson2(CrankNicolson):
    name = CN_STR + "_doubleSolve"
    """
    Formulation obtained by Schur komplement without replacing the pn equation
    Solves with System for qn and with Mass for pn
    Needs a boundary condition for the p component
    """

    def __init__(
        self,
        T: float,
        tau: float,
        Omega: Domain,
        SpaceDiscretizationClass: Type[SpaceDiscretization],
        problem: ProblemData,
    ) -> None:
        super().__init__(T, tau, Omega, SpaceDiscretizationClass, problem)

        phi = ufl.TestFunction(self.V)
        self.F = self.problem.f  # a time dependent functor
        self.F_tn = self.F.copy()  # ufl expression 1 in rhs form
        self.f_tn = self.F_tn(0.0)
        self.F_tn_p = self.F.copy()  # ufl expression 2 in rhs form
        self.f_tn_p = self.F_tn_p(self.tau)
        self.qn = dfx.fem.Function(self.V)
        self.pn = dfx.fem.Function(self.V)
        self.rhs_form = dfx.fem.form(
            (
                self.qn * phi
                + self.tau_ufl * self.pn * phi
                + self.tau_ufl**2 / 4 * (self.f_tn + self.f_tn_p) * phi
            )
            * self.space_discretization.mass_measure
            - (
                self.tau_ufl**2
                / 4
                * ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
            )
            * self.space_discretization.stiffness_measure
        )
        self.b = PETSC_PATH.create_vector(self.V)

        self.qn_p = dfx.fem.Function(self.V)
        self.rhs_p = dfx.fem.form(
            (self.pn * phi + self.tau_ufl / 2 * (self.f_tn + self.f_tn_p) * phi)
            * self.space_discretization.mass_measure
            - (
                self.tau_ufl
                / 2
                * (
                    ufl.inner(self.problem.c**2 * ufl.grad(self.qn), ufl.grad(phi))
                    + ufl.inner(self.problem.c**2 * ufl.grad(self.qn_p), ufl.grad(phi))
                )
                * self.space_discretization.stiffness_measure
            )
        )
        self.b_p = PETSC_PATH.create_vector(self.V)

    def step(self, tn, tau, qn, pn, bc, pred_bc_p=None):
        self.F_tn.t.value = tn
        self.F_tn_p.t.value = tn + tau
        if self.F.reevaluation:
            self.f_tn = self.F_tn(tn)
            self.f_tn_p = self.F_tn_p(tn + tau)
        qn_p = dfx.fem.Function(self.V)
        pn_p = dfx.fem.Function(self.V)
        self.qn.x.array[:] = qn.x.array
        self.pn.x.array[:] = pn.x.array

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b, self.rhs_form)
        bcs = bc if type(bc) == list else [bc]

        self.space_discretization.apply_bc_rhs(
            self.b, bcs, self.space_discretization.system_form
        )
        self.space_discretization.solverSystem.solve(self.b, qn_p.x.petsc_vec)

        p_bc = dfx.fem.Function(self.V)
        if pred_bc_p is None:
            p_bc.x.array[:] = 2.0 / tau * (qn_p.x.array - qn.x.array) - pn.x.array
            bcs_p = [self.Omega.DirichletBC(p_bc)]
        else:
            p_bc.x.array[:] = 2.0 / tau * (qn_p.x.array - qn.x.array) - pn.x.array
            bcs_p = [
                dfx.fem.dirichletbc(p_bc, self.Omega.parent_boundary_dofs),
                pred_bc_p,
            ]

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        PETSC_PATH.assemble_vector(self.b_p, self.rhs_p)
        self.space_discretization.apply_bc_rhs(
            self.b, bcs_p, self.space_discretization.mass_form
        )
        self.space_discretization.solverM.solve(self.b, pn_p.x.petsc_vec)

        return qn_p, pn_p
