from petsc4py import PETSc
from mpi4py import MPI
import dolfinx as dfx
import dolfinx.fem.petsc
import numpy as np
from abc import ABC, abstractmethod
import ufl
from typing import Optional
from Matrices import get_scipy_sparse_matrix, sum_rows_to_diagonal
from Config import (
    DIRECT,
    ITERATIVE,
    ITERATIVE_METHOD,
    PRECONDITIONER,
    MAX_ITERATIONS,
    ABS_TOLERANCE,
    REL_TOLERANCE,
    JIT_OPTIONS,
    DEFAULT_SOLVING_TYP,
    FEM_DEGREE,
    FEM_TYPE,
)
from DolfinVersion import (
    DIRICHLET_BC_TYPE,
    LA_VECTOR,
    DFX_FORM,
    PETSC_PATH,
    geometric_dimension,
)

# How should B.c. be represented, somehow they are intertangled between space and time discretization ?


class MassLumping_solver:
    def __init__(self, V: dfx.fem.function.FunctionSpace, diagM: PETSc.Vec) -> None:
        self.V = V
        self.diagM = diagM

    def lumpedProject(self, rhs):
        # v = ufl.TestFunction(self.V)
        # make f function
        ##ff = dfx.fem.Function(self.V)
        ##ff.x.array[:] = f[:]
        # lhs = PETSC_PATH.assemble_vector(dfx.fem.form(ufl.inner(dfx.fem.Constant(self.V.mesh,1.0),v)*ufl.dx))
        # rhs = PETSC_PATH.assemble_vector(dfx.fem.form(ufl.inner(ff,v)*ufl.dx))
        u = dfx.fem.Function(self.V)
        u.x.petsc_vec.pointwiseDivide(rhs, self.diagM)
        return u

    def solve(self, b: LA_VECTOR, x: LA_VECTOR) -> None:
        u = self.lumpedProject(b)
        x.array[:] = u.x.array[:]
        # make sure this is called after
        # PETSC_PATH.set_bc(b, bcs=bcs)


class SpaceDiscretization(ABC):
    """
    The SpaceDiscretization class basically provides mass and stiffness matrix \n
    implicit TimeIntegrators can invoke the method set_SystemMatrix to preassemble
    the matrix one has to solve with
    """

    def __init__(self) -> None:
        self.solvingType: str = DEFAULT_SOLVING_TYP
        self.solverM: Optional[PETSc.KSP | PETSc.Mat] = None
        self.mass_form: Optional[ufl.form.Form] = None
        self.mass_measure: Optional[ufl.measure.Measure] = None
        self.stiffness_form: Optional[ufl.form.Form] = None
        self.stiffness_measure: Optional[ufl.measure.Measure] = None
        self.system_form: Optional[ufl.form.Form] = None
        self.solverSystem: Optional[PETSc.KSP | PETSc.Mat] = None
        self.assert_correct_config()

    @abstractmethod
    def assert_correct_config(self):
        pass

    @abstractmethod
    def assemble(
        self,
        V: dfx.fem.function.FunctionSpace,
        c: float | dfx.fem.Constant | dfx.fem.function.Function,
        bcs: list[DIRICHLET_BC_TYPE],
    ) -> None:
        pass

    @abstractmethod
    def set_System(
        self,
        factor_before_StiffnessMatrix: float,
        c: float | dfx.fem.Constant | dfx.fem.function.Function,
        V: dfx.fem.function.FunctionSpace,
        bcs: list[DIRICHLET_BC_TYPE],
    ) -> None:
        pass

    def LinearSolver(
        self,
        form: DFX_FORM,
        V: dfx.fem.function.FunctionSpace,
        bcs: list[DIRICHLET_BC_TYPE],
    ) -> PETSc.Mat | PETSc.KSP:
        if self.solvingType == DIRECT:
            """
            # PETSc interface for LU factorization
            solver = PETSC_PATH.assemble_matrix(form, bcs=bcs)
            solver.assemble()
            num_rows, num_cols = solver.getSize()
            rows = PETSc.IS().createStride(num_rows, first=0, step=1)
            cols = PETSc.IS().createStride(num_cols, first=0, step=1)
            # Factorize using LU factorization
            solver.factorLU(rows, cols)
            # Solve the linear system: A * x = b
            # solver.solve(b, x)
            return solver
            """
            # FenicsX interface
            A = PETSC_PATH.assemble_matrix(form, bcs=bcs)
            A.assemble()
            solver = PETSc.KSP().create(V.mesh.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(
                PETSc.PC.Type.CHOLESKY
            )  # or e.g. Type.LU, see https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html
            return solver

        elif self.solvingType == ITERATIVE:
            A = PETSC_PATH.assemble_matrix(form, bcs=bcs)
            A.assemble()
            solver = PETSc.KSP().create(V.mesh.comm)
            solver.setOperators(A)
            solver.setType(ITERATIVE_METHOD)  # PETSc.KSP.Type.CG
            solver.getPC().setType(PRECONDITIONER)  # PETSc.PC.Type.JACOBI
            solver.setTolerances(
                rtol=REL_TOLERANCE, atol=ABS_TOLERANCE, max_it=MAX_ITERATIONS
            )
            return solver

        raise NotImplementedError

    def apply_bc_rhs(
        self,
        rhs: LA_VECTOR,
        bcs: list[DIRICHLET_BC_TYPE],
        form: Optional[ufl.form.Form] = None,
    ):
        if form is None:
            PETSC_PATH.apply_lifting(rhs, [self.system_form], bcs=[bcs])
        else:
            PETSC_PATH.apply_lifting(rhs, [form], bcs=[bcs])
        rhs.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )  # is this needed?
        dfx.fem.set_bc(rhs, bcs)

        # link for later (Robin bc)
        # https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html


class FEM(SpaceDiscretization):
    def __init__(self) -> None:
        super().__init__()
        self.mass_measure = ufl.dx
        self.stiffness_measure = ufl.dx

    def assert_correct_config(self):
        assert FEM_TYPE in ["Lagrange", "CG"]

    def assemble(
        self,
        V: dfx.fem.function.FunctionSpace,
        c: float | dfx.fem.Constant | dfx.fem.function.Function,
        bcs: list[DIRICHLET_BC_TYPE],
    ) -> None:
        """
        Important note:
        Matrices are assembled WITHOUT bcs, so that they can be used for matrix vector multiplications
        The Linear Solvers however include the boundary conditions
        DO NOT FORGET to call apply_bc_rhs before solving
        """
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        if c is float:
            wave_c = dfx.fem.Constant(V.mesh, c)
        else:
            wave_c = c
        self.stiffness_form = dfx.fem.form(
            ufl.dot(wave_c**2 * ufl.grad(u), ufl.grad(v)) * self.stiffness_measure
        )
        self.mass_form = dfx.fem.form(u * v * self.mass_measure)

        self.solverM = self.LinearSolver(self.mass_form, V, bcs)

    def set_System(
        self,
        factor_before_StiffnessMatrix: float,
        c: float | dfx.fem.Constant | dfx.fem.function.Function,
        V: dfx.fem.function.FunctionSpace,
        bcs: list[DIRICHLET_BC_TYPE],
    ) -> None:
        """
        sets the SystemMatrix based on the passed factor
        Args:
            factor_before_StiffnessMatrix (float): should be known by time integrator, e.g. tau**2/4 (CN)
        """
        if c is float:
            wave_c = dfx.fem.Constant(V.mesh, c)
        else:
            wave_c = c
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        self.system_form = dfx.fem.form(
            (
                u * v
                + factor_before_StiffnessMatrix
                * ufl.dot(wave_c**2 * ufl.grad(u), ufl.grad(v))
            )
            * ufl.dx
        )
        self.solverSystem = self.LinearSolver(self.system_form, V, bcs)


class FEM_ml(FEM):
    def __init__(self) -> None:
        super().__init__()
        self.stiffness_measure = ufl.dx
        self.mass_measure = ufl.Measure(
            "dx", metadata={"quadrature_degree": 1, "quadrature_rule": "vertex"}
        )

    def assert_correct_config(self):
        return super().assert_correct_config()

    def assemble(
        self,
        V: dfx.fem.function.FunctionSpace,
        c: float | dfx.fem.Constant | dfx.fem.function.Function,
        bcs: list[DIRICHLET_BC_TYPE],
    ) -> None:
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        if c is float:
            wave_c = dfx.fem.Constant(V.mesh, c)
        else:
            wave_c = c
        self.stiffness_form = dfx.fem.form(
            ufl.dot(wave_c**2 * ufl.grad(u), ufl.grad(v)) * self.stiffness_measure
        )

        self.mass_form = dfx.fem.form(ufl.inner(u, v) * self.mass_measure)
        MassMatrix = PETSC_PATH.assemble_matrix(self.mass_form, bcs=bcs)
        MassMatrix.assemble()
        diagonalMass = MassMatrix.getDiagonal()
        self.solverM = MassLumping_solver(V, diagonalMass)

    def set_System(
        self,
        factor_before_StiffnessMatrix: float,
        c: float | dfx.fem.Constant | dfx.fem.function.Function,
        V: dfx.fem.function.FunctionSpace,
        bcs: list[DIRICHLET_BC_TYPE],
    ) -> None:
        """
        sets the SystemMatrix based on the passed factor
        Args:
            factor_before_StiffnessMatrix (float): should be known by time integrator, e.g. tau**2/4 (CN)
        """
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        if c is float:
            wave_c = dfx.fem.Constant(V.mesh, c)
        else:
            wave_c = c
        self.system_form = dfx.fem.form(
            ufl.inner(u, v) * self.mass_measure
            + factor_before_StiffnessMatrix
            * ufl.dot(wave_c**2 * ufl.grad(u), ufl.grad(v))
            * self.stiffness_measure
        )
        self.solverSystem = self.LinearSolver(self.system_form, V, bcs)


class DG(SpaceDiscretization):
    def __init__(self, weighted_version=True) -> None:
        super().__init__()
        self.wsip: bool = weighted_version  # for later
        self.hF_min: bool = True
        self.mass_measure = ufl.dx
        self.stiffness_measure = ufl.dx

    def assert_correct_config(self):
        assert FEM_TYPE == "DG"

    def assemble(
        self,
        V: dfx.fem.function.FunctionSpace,
        c: float | dfx.fem.Constant | dfx.fem.function.Function,
    ) -> None:
        self.eta_S = (
            4.0 * FEM_DEGREE * (FEM_DEGREE + geometric_dimension(V.mesh))
        )

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # bilinear form for mass matrix
        self._m = u * v * self.mass_measure
        self.mass_form = dfx.fem.form(self._m)
        self.solverM = self.LinearSolver(self.mass_form, V, [])

        if c is float:
            wave_c_squared = dfx.fem.Constant(V.mesh, c**2)
        else:
            wave_c_squared = c**2
        # define normal vector and mesh size
        nF = ufl.FacetNormal(V.mesh)
        hK = ufl.CellDiameter(V.mesh)
        hF = ufl.FacetArea(V.mesh)
        dim = V.mesh.topology.dim

        # SIPdG bilinear forms

        # hF functions in penalty term
        self.hFi = None
        self.hFb = None
        if self.hF_min or dim == 1:
            self.hFb = hK
            self.hFi = ufl.conditional(ufl.lt(hK("+"), hK("-")), hK("+"), hK("-"))
        else:
            self.hFb = hF
            self.hFi = (hF("+") + hF("-")) / 2.0

        avg_u = ufl.avg(wave_c_squared * ufl.grad(u))
        avg_v = ufl.avg(wave_c_squared * ufl.grad(v))

        # bilinear form for stiffness matrix
        self._a_cell = (
            wave_c_squared * ufl.dot(ufl.grad(u), ufl.grad(v)) * self.stiffness_measure
        )  # (A)

        self._a_iface = (
            -ufl.dot(avg_u, ufl.jump(v, nF)) * ufl.dS  # (B)
            - ufl.dot(avg_v, ufl.jump(u, nF)) * ufl.dS  # (C inner faces)
            + (self.eta_S * wave_c_squared / self.hFi)
            * ufl.dot(ufl.jump(v, nF), ufl.jump(u, nF))
            * ufl.dS  # (D inner faces)
        )

        self._a_bface = (
            -wave_c_squared
            * ufl.dot(ufl.grad(u), v * nF)
            * ufl.ds  # (B boundary faces)
            - wave_c_squared
            * ufl.dot(ufl.grad(v), u * nF)
            * ufl.ds  # (C boundary faces)
            + (self.eta_S * wave_c_squared / self.hFb)
            * u
            * v
            * ufl.ds  #  (D boundary faces)
        )

        self.stiffness_form = dfx.fem.form(self._a_cell + self._a_iface + self._a_bface)

    def set_System(
        self,
        factor_before_StiffnessMatrix: float,
        V: dfx.fem.function.FunctionSpace,
    ) -> None:
        """
        sets the SystemMatrix based on the passed factor
        Args:
            factor_before_StiffnessMatrix (float): should be known by time integrator, e.g. tau**2/4 (CN)
        """
        self.system_form = dfx.fem.form(
            (
                self._m
                + factor_before_StiffnessMatrix
                * (self._a_cell + self._a_iface + self._a_bface)
            )
        )
        self.solverSystem = self.LinearSolver(self.system_form, V, [])
