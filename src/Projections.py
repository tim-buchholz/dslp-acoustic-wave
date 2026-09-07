import numpy as np
import dolfinx as dfx
import dolfinx.fem.petsc
from petsc4py import PETSc
import ufl
from Domain import Domain
from SpaceDiscretization import SpaceDiscretization
from ProblemDataUFL import ProblemData
from DolfinVersion import PETSC_PATH
from Config import DOF_TOLERANCE
import basix


def Ritz_projection_initial_values(problem: ProblemData, Omega: Domain):
    phi = ufl.TrialFunction(Omega.V)
    psi = ufl.TestFunction(Omega.V)

    a_K = ufl.dot(problem.c**2 * ufl.grad(phi), ufl.grad(psi)) * ufl.dx
    b_q = problem.c**2 * ufl.dot(ufl.grad(problem.u0), ufl.grad(psi)) * ufl.dx
    b_p = problem.c**2 * ufl.dot(ufl.grad(problem.v0), ufl.grad(psi)) * ufl.dx

    if problem.u_bc is None:
        u_D = dfx.fem.Constant(Omega.V.mesh, 0.0)
        bc = dfx.fem.dirichletbc(u_D, Omega.boundary_dofs, Omega.V)
    else:
        u_D = dfx.fem.Function(Omega.V)
        u_D.interpolate(problem.dfExpression(problem.u_bc(0.0), Omega.V))
        bc = dfx.fem.dirichletbc(u_D, Omega.boundary_dofs)

    LinearProblem_q = PETSC_PATH.LinearProblem(
        a_K,
        b_q,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="Ritz",
    )
    q0 = LinearProblem_q.solve()

    u_D = dfx.fem.Constant(Omega.V.mesh, 0.0)
    bc = dfx.fem.dirichletbc(u_D, Omega.boundary_dofs, Omega.V)
    LinearProblem_p = PETSC_PATH.LinearProblem(
        a_K,
        b_p,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="Ritz",
    )
    p0 = LinearProblem_p.solve()
    return q0, p0


def project_L2(self, function_expression):
    projected_sol = dfx.Vector()
    degree_FE = self.V.ufl_element().degree()

    v = dfx.TestFunction(self.V)
    lu = function_expression * v * dfx.dx
    Lu = dfx.assemble(
        lu, form_compiler_parameters={"quadrature_degree": 2 * degree_FE + 2}
    )
    self.bc.apply(Lu)
    self.solverM.solve(projected_sol, Lu)
    # self.bc.apply(projected_sol)
    return projected_sol


def project_L2(self, function_expression):
    projected_sol = dfx.Vector()
    degree_FE = self.V.ufl_element().degree()

    v = dfx.TestFunction(self.V)
    lu = ufl.real(function_expression) * v * dfx.dx
    Lu = dfx.assemble(
        lu, form_compiler_parameters={"quadrature_degree": 2 * degree_FE + 2}
    )
    self.bc.apply(Lu)
    self.solverM.solve(projected_sol, Lu)
    # self.bc.apply(projected_sol)
    return projected_sol


def project_to_fine(uh_coarse, V_coarse, V_fine, degree, degree_raise=2):
    # quadrature points of fine mesh
    deg = degree + degree_raise
    gdim = V_fine.ufl_domain()._geometric_dimension
    Qe = basix.ufl.quadrature_element(
        V_fine.element.basix_element.cell_type,
        (),  # V_fine.element.basix_element.value_shape[0],
        scheme="default",
        degree=deg,
    )
    V_fine.mesh.ufl_domain().ufl_cell()._gdim = V_fine.ufl_domain()._geometric_dimension
    V_quadrature = dfx.fem.functionspace(V_fine.mesh, Qe)

    #  interpolation_data = (
    #             dfx.fem.create_interpolation_data(
    #                 self.V,
    #                 self.parentV,
    #                 cells,
    #                 padding=DOF_TOLERANCE,
    #             )
    #         )
    #         local.interpolate_nonmatching(
    #             vec,
    #             cells,
    #             interpolation_data,
    #         )

    num_cells_target = V_quadrature.dofmap.list.shape[0]
    cells = np.arange(num_cells_target, dtype=np.int32)

    nmmid = dfx.fem.create_interpolation_data(
        V_quadrature,
        V_coarse,
        cells,
        padding=DOF_TOLERANCE,
    )

    q_func = dfx.fem.Function(V_quadrature)
    q_func.interpolate(uh_coarse, cells, nmmid)

    # Project coarse function at quadrature points to fine grid
    u = ufl.TrialFunction(V_fine)
    v = ufl.TestFunction(V_fine)
    a_fine = ufl.inner(u, v) * ufl.dx
    L_fine = ufl.inner(q_func, v) * ufl.dx
    problem = dfx.fem.petsc.LinearProblem(
        a_fine, L_fine, petsc_options_prefix="projection"
    )
    uh_fine = problem.solve()

    return uh_fine


def project_to_coarse(u_fine, V_fine, V_coarse, degree, degree_raise):
    # quadrature points of coarse mesh
    deg = degree + degree_raise
    Qe = basix.ufl.quadrature_element(
        V_coarse.element.basix_element.cell_type, (), scheme="default", degree=deg
    )
    V_quadrature = dfx.fem.functionspace(V_coarse.mesh, Qe)
    nmmid = dfx.fem.create_nonmatching_meshes_interpolation_data(
        V_quadrature.mesh._cpp_object,
        V_quadrature.element,
        V_fine.mesh._cpp_object,
        padding=1e-6,
    )

    q_func = dfx.fem.Function(V_quadrature)
    q_func.interpolate(u_fine, nmm_interpolation_data=nmmid)

    # Project coarse function at quadrature points to fine grid
    u = ufl.TrialFunction(V_coarse)
    v = ufl.TestFunction(V_coarse)
    a_coarse = ufl.inner(u, v) * ufl.dx
    L_coarse = ufl.inner(q_func, v) * ufl.dx
    problem = dfx.fem.petsc.LinearProblem(
        a_coarse, L_coarse, petsc_options_prefix="projection"
    )
    uh_coarse = problem.solve()

    return uh_coarse
