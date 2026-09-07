from mpi4py import MPI
import dolfinx as dfx
import dolfinx.fem.petsc
import ufl
import basix
import numpy as np
from typing import Optional, Callable


def _linear_problem_prefix(label: str, V: dfx.fem.FunctionSpace) -> str:
    return f"{label}_{id(V):x}_"


def norm_L2(V: dfx.fem.FunctionSpace, u: dfx.fem.Function):
    # Integrate the error
    error = dfx.fem.form(ufl.inner(u, u) * ufl.dx)
    error_local = dfx.fem.assemble_scalar(error)
    error_global = V.mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def norm_H10(V: dfx.fem.FunctionSpace, u: dfx.fem.Function):
    # Integrate the error
    error = dfx.fem.form(ufl.dot(ufl.grad(u), ufl.grad(u)) * ufl.dx)
    error_local = dfx.fem.assemble_scalar(error)
    error_global = V.mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def norm_H1(V: dfx.fem.FunctionSpace, u: dfx.fem.Function):
    # Integrate the error
    return np.sqrt(norm_H10(V, u) ** 2 + norm_L2(V, u) ** 2)


def norm_WSIP(
    V: dfx.fem.FunctionSpace,
    u: dfx.fem.Function,
    kappa_function: dfx.fem.Function,
):
    hK = ufl.CellDiameter(V.mesh)

    hF_b = hK
    hF_i = ufl.conditional(ufl.lt(hK("+"), hK("-")), hK("+"), hK("-"))

    kappa_b = kappa_function
    kappa_i = (2 * kappa_function("+") * kappa_function("-")) / (
        kappa_function("+") + kappa_function("-")
    )

    aF_b = kappa_b / hF_b
    aF_i = kappa_i / hF_i

    volume_term = (
        kappa_function * ufl.dot(ufl.grad(u), ufl.grad(u)) * ufl.dx(domain=V.mesh)
    )
    interface_term = aF_i * ufl.jump(u) * ufl.jump(u) * ufl.dS(domain=V.mesh)
    boundary_term = aF_b * u * u * ufl.ds(domain=V.mesh)
    error = dfx.fem.form(volume_term + interface_term + boundary_term)
    error_local = dfx.fem.assemble_scalar(error)
    error_global = V.mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def L2_project(
    V_to: dfx.fem.FunctionSpace, expr: ufl.core.expr.Expr
) -> dfx.fem.Function:
    u = ufl.TrialFunction(V_to)
    v = ufl.TestFunction(V_to)
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(expr, v) * ufl.dx(domain=V_to.mesh)
    problem = dfx.fem.petsc.LinearProblem(
        a, L, petsc_options_prefix=_linear_problem_prefix("l2_project", V_to)
    )
    func_proj = problem.solve()
    return func_proj


def L2_project_func(
    V_to: dfx.fem.FunctionSpace,
    V_from: dfx.fem.FunctionSpace,
    func: dfx.fem.Function,
    degree_raise: int = 0,
    via_quadSpace: bool = True,
) -> dfx.fem.Function:
    tdim = V_to.mesh.topology.dim
    if via_quadSpace:
        degree_V_to = V_to.element.basix_element.degree
        degree = degree_V_to + degree_raise
        Qe = basix.ufl.quadrature_element(V_to.mesh.topology.cell_name(), degree=degree)
        V_quadrature = dfx.fem.functionspace(V_to.mesh, Qe)
        func_quad = dfx.fem.Function(V_quadrature)
        cells_quad = np.arange(
            V_quadrature.mesh.topology.index_map(tdim).size_local, dtype=np.int32 # size local or size global?
        )
        int_data_quadSpace = dfx.fem.create_interpolation_data(
            V_quadrature, V_from, cells_quad
        )
        func_quad.interpolate_nonmatching(func, cells_quad, int_data_quadSpace)
    else:
        func_quad = func

    u = ufl.TrialFunction(V_to)
    v = ufl.TestFunction(V_to)
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(func_quad, v) * ufl.dx(domain=V_to.mesh)
    problem = dfx.fem.petsc.LinearProblem(
        a, L, petsc_options_prefix=_linear_problem_prefix("l2_project_func", V_to)
    )
    func_proj = problem.solve()
    return func_proj


def error_norm_ref(
    uh: dfx.fem.function.Function,
    u_ref: dfx.fem.function.Function,
    norm_str: str,
    degree_raise=1,
    L2project: bool = True,
    **kwargs
):
    # Create higher order function space
    dg = u_ref.function_space.ufl_element().discontinuous
    family_name_ref = u_ref.function_space.ufl_element().family_name
    if dg and family_name_ref == "P":
        family_ref = "DG"
    else: 
        family_ref = u_ref.function_space.ufl_element().element_family

    degree_ref = u_ref.function_space.ufl_element().degree
    mesh_ref = u_ref.function_space.mesh
    V_ref = dfx.fem.functionspace(mesh_ref, (family_ref, degree_ref))
    W_ref = dfx.fem.functionspace(mesh_ref, (family_ref, degree_ref + degree_raise))
    e_W = dfx.fem.Function(W_ref)
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().element_family
    mesh = uh.function_space.mesh
    V_h = dfx.fem.functionspace(mesh, (family, degree))
    if L2project:
        uh_proj = L2_project_func(W_ref, V_h, uh)
        uref_proj = L2_project_func(W_ref, V_ref, u_ref)
        e_W.x.array[:] = uh_proj.x.array - uref_proj.x.array
    else:
        uh_int = interpolate_nonmatching(W_ref, V_h, uh)
        uref_int = dfx.fem.Function(W_ref)
        uref_int.interpolate(u_ref)
        e_W.x.array[:] = uh_int.x.array - uref_int.x.array

    if norm_str == "L2":
        return norm_L2(W_ref, e_W)
    elif norm_str == "H10":
        return norm_H10(W_ref, e_W)
    elif norm_str == "H1":
        return norm_H1(W_ref, e_W)
    elif norm_str == "WSIP":
        if "kappa" in kwargs.keys():
            kappa_proj = L2_project_func(W_ref, V_h, kwargs["kappa"])
        elif "kappa_ref" in kwargs.keys():
            kappa_proj = L2_project_func(W_ref, V_ref, kwargs["kappa_ref"])
        else:
            raise KeyError("kappa must be provided")
        return norm_WSIP(W_ref, e_W, kappa_proj)


def error_norm(
    uh: dfx.fem.function.Function,
    u_ex: Callable[[np.ndarray | float], float] | ufl.core.expr.Expr,
    norm_str: str,
    degree_raise=2,
    relative: bool = False,
    **kwargs
):
    # Create higher order function space
    dg = uh.function_space.ufl_element().discontinuous
    family_name = uh.function_space.ufl_element().family_name
    if dg and family_name == "P":
        family = "DG"
    else:
        family = uh.function_space.ufl_element().element_family

    degree = uh.function_space.ufl_element().degree
    mesh = uh.function_space.mesh
    V = dfx.fem.functionspace(mesh, (family, degree))
    W = dfx.fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = dfx.fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dfx.fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = dfx.fem.Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = dfx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    if relative:
        if norm_str == "L2":
            ref_norm = norm_L2(W, u_ex_W)
        elif norm_str == "H10":
            ref_norm = norm_H10(W, u_ex_W)
        elif norm_str == "H1":
            ref_norm = norm_H1(W, u_ex_W)  
    else:
        ref_norm = 1.0    

    if relative:
        if norm_str == "L2":
            return norm_L2(W, e_W) / ref_norm, ref_norm
        elif norm_str == "H10":
            return norm_H10(W, e_W) / ref_norm, ref_norm
        elif norm_str == "H1":
            return norm_H1(W, e_W) / ref_norm, ref_norm
    else: 
        if norm_str == "L2":
            return norm_L2(W, e_W) / ref_norm
        elif norm_str == "H10":
            return norm_H10(W, e_W) / ref_norm
        elif norm_str == "H1":
            return norm_H1(W, e_W) / ref_norm


def interpolate_nonmatching(
    V_to: dfx.fem.FunctionSpace, V_from: dfx.fem.FunctionSpace, func: dfx.fem.Function
) -> dfx.fem.Function:
    tdim = V_to.mesh.topology.dim
    cells_V_to = np.arange(
        V_to.mesh.topology.index_map(tdim).size_global, dtype=np.int32
    )
    interpolation_data = dfx.fem.create_interpolation_data(V_to, V_from, cells_V_to)
    func_V_to = dfx.fem.Function(V_to)
    func_V_to.interpolate_nonmatching(func, cells_V_to, interpolation_data)
    return func_V_to
