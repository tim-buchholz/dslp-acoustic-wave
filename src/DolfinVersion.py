from mpi4py import MPI
import dolfinx as dfx
import dolfinx.fem.petsc
import typing
import petsc4py
import warnings
from packaging.version import Version

if MPI.COMM_WORLD.rank == 0:
    print(f"===Found dolfinx version {dfx.__version__}===")
if Version(dfx.__version__) < Version("0.10.0"):
    raise ModuleNotFoundError("Version of dolfinx must be at least 0.10.0")

FUNCTION = dfx.fem.function.Function
LA_VECTOR = petsc4py.PETSc.Vec
DIRICHLET_BC_TYPE = dfx.fem.bcs.DirichletBC
DFX_FORM = dfx.fem.forms.Form
PETSC_PATH = dfx.fem.petsc  # note that additional import is needed
FUNCTION_SPACE = dfx.fem.function.FunctionSpace


def geometric_dimension(mesh_or_ufl_domain) -> int:
    """Return the UFL geometric dimension across dolfinx/UFL API versions."""
    ufl_domain = (
        mesh_or_ufl_domain.ufl_domain()
        if hasattr(mesh_or_ufl_domain, "ufl_domain")
        else mesh_or_ufl_domain
    )
    gdim = ufl_domain.geometric_dimension
    return int(gdim() if callable(gdim) else gdim)
