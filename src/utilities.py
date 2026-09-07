import numpy as np
import scipy.sparse as scp
import scipy.sparse.linalg as scplin
from mpi4py import MPI
import dolfinx as dfx
import ufl
import dolfinx.fem.petsc
from petsc4py import PETSc
from ProblemDataUFL import ProblemData
from SpaceDiscretization import SpaceDiscretization
from Domain import Domain
from Matrices import get_scipy_sparse_matrix, sum_rows_to_diagonal
import warnings
from DolfinVersion import PETSC_PATH


def deprecated_function():
    # Issue a deprecation warning
    warnings.warn(
        "This function is deprecated and will be removed in future versions.",
        DeprecationWarning,
    )


def MWE_mass_lumping():
    mesh_2d = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V_2d = dfx.fem.FunctionSpace(mesh_2d, ("Lagrange", 1))

    u = ufl.TrialFunction(V_2d)
    v = ufl.TestFunction(V_2d)

    lumped_integration_dx = ufl.Measure("dx", metadata={"quadrature_rule": "vertex"})
    mass_form = dfx.fem.form(ufl.inner(u, v) * lumped_integration_dx)
    MassMatrix = PETSC_PATH.assemble_matrix(mass_form, bcs=[])
    MassMatrix.assemble()
    print(get_scipy_sparse_matrix(MassMatrix).toarray())

    mesh_1d = dfx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V_1d = dfx.fem.FunctionSpace(mesh_1d, ("Lagrange", 1))

    u = ufl.TrialFunction(V_1d)
    v = ufl.TestFunction(V_1d)

    lumped_integration_dx = ufl.Measure("dx", metadata={"quadrature_rule": "vertex"})
    mass_form = dfx.fem.form(ufl.inner(u, v) * lumped_integration_dx)
    MassMatrix = PETSC_PATH.assemble_matrix(mass_form, bcs=[])
    MassMatrix.assemble()
    print(get_scipy_sparse_matrix(MassMatrix).toarray())


def MWE_mass_lumping2():
    import numpy as np
    from mpi4py import MPI
    from dolfinx import mesh
    from dolfinx.fem.petsc import assemble_matrix
    from dolfinx.fem import VectorFunctionSpace, Function

    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 8)
    V = VectorFunctionSpace(domain, ("Lagrange", 1), dim=2)
    from dolfinx import fem
    import numpy
    import ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # Mass Lumping
    dxL = ufl.Measure("dx", metadata={"quadrature_rule": "vertex"})
    Eqa = ufl.inner(u, v) * ufl.dx
    Eqb = ufl.inner(u, v) * dxL

    Xa = assemble_matrix(fem.form(Eqa))
    Xb = assemble_matrix(fem.form(Eqb))
    Xa.assemble()
    Xb.assemble()
    Xa.convert("dense")
    Ca = Xa.getDenseArray()
    Xb.convert("dense")
    Cb = Xb.getDenseArray()
    print("w/o mass lumping \n", Ca)
    print("w/ mass lumping \n", Cb)
    print("difference: ", np.max(np.abs(Ca - Cb)))
    only_diagonal = True
    for i in range(len(Cb)):
        for j in range(len(Cb[0])):
            if i != j:
                if Cb[i][j] != 0.0:
                    only_diagonal = False

    print("only values on the diagonal: ", only_diagonal)


if __name__ == "__main__":
    print(dfx.__version__)
    MWE_mass_lumping()
