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
from Config import DIRECT, ITERATIVE, ATI_STR, CN_STR, LF_STR
from Projections import Ritz_projection_initial_values
from MeshIO import save_vtk, save_xdmf, write_adios_function
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
        return f"TimeIntegrator={type(self).__name__}::problem={problem_number}::c={wave_propagation_speed}::domain={domain_type}::sd={space_discretization}::tau={self.tau}::T={self.T}"

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

    def save_vtk(self, uh: dfx.fem.function.Function, name="output"):
        save_vtk(self.V, uh, name)

    def save_xdmf(self, uh: dfx.fem.function.Function, name="output"):
        save_xdmf(self.V, uh, name)

    def save_adios(
        self,
        uh: dfx.fem.function.Function,
        V: dfx.fem.function.FunctionSpace,
        name="output",
        t: float = 0.0,
        function_name: str = "f",
    ):
        write_adios_function(name, V, uh, t, function_name)
