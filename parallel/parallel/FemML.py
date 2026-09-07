from mpi4py import MPI
import dolfinx as dfx
import dolfinx.fem.petsc
import ufl
from typing import Union, Optional, List, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def mass_form(
    V: dfx.fem.FunctionSpace, 
    u: dfx.fem.function.Function | ufl.argument.Argument
) -> ufl.form.Form:
    mass_measure = ufl.Measure(
        "dx",domain=V.mesh, metadata={"quadrature_degree": 1, "quadrature_rule": "vertex"}
    )
    v = ufl.TestFunction(V)
    mass_form = u * v * mass_measure
    return mass_form


def stiffness_form(
    V: dfx.fem.FunctionSpace,
    u: dfx.fem.function.Function | ufl.argument.Argument,
    kappa_function: dfx.fem.function.Function,
) -> ufl.form.Form:
    stiffness_measure = ufl.dx(domain=V.mesh)
    v = ufl.TestFunction(V)
    stiffness_form = kappa_function * ufl.dot(ufl.grad(u), ufl.grad(v)) * stiffness_measure
    return stiffness_form
