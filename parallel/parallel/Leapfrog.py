#!/usr/bin/env python3

# usage: mpirun -np N_PROCESSES [--oversubscribe] Leapfrog
#        [-h] [--mesh_dir MESH_DIR] [--hdf5_source HDF5_SOURCE]
#        [--ic {pulse,mms}] [--pulse_mu PULSE_MU]
#        [--pulse_s PULSE_S] [--pulse_b PULSE_B]
#        [--pulse_amplitude_factor PULSE_AMPLITUDE_FACTOR]
#        [--rhs {zero,mms}] [--kappa {const}]
#        [--kappa_value KAPPA_VALUE]
#        [--solvingType {direct,iterative}]
#        [--direct_method DIRECT_METHOD]
#        [--iterative_method ITERATIVE_METHOD]
#        [--preconditioner PRECONDITIONER]
#        [--max_iterations MAX_ITERATIONS] [--tol TOL]
#        [--output_mode {last,equidistant,all}]
#        [--output_equidistant_N OUTPUT_EQUIDISTANT_N]
#        [--solution_dir SOLUTION_DIR]
#        [--solution_name SOLUTION_NAME] [-v]
#        tau T

import os

# Prevent thread explosion in BLAS / LAPACK / MKL / OpenMP:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc
from typing import Optional, Callable, Tuple
import numpy as np
import sys
import logging
from time import time, sleep
from pathlib import Path
from datetime import datetime
from Domain import Domain, SubDomain
import adios4dolfinx
import ufl
import basix
import shutil
import os
from Norms import L2_project, L2_project_func
from Meshing import get_h
import Fem
import FemML
from Simulation import simulation_summary
import tqdm
import math
from Functions import interpolate_ufl_expression, ic_pulse2D

###
# CLI definition
###
import argparse

parser = argparse.ArgumentParser(
    prog=f"mpirun -np N_PROCESSES [--oversubscribe] Leapfrog",
    description=">>> LF-FEM: Simulation Script of Leapfrog applied to the wave equation with FEM space discretization (%(prog)s tau T) <<<",
    epilog=">>>  DS-LF Script  <<<",
)
# time stepping arguments
time_stepping = parser.add_argument_group("time stepping")
time_stepping.add_argument("tau", type=float, help="time step size")
time_stepping.add_argument("T", type=float, help="final time")
# mesh dir
mesh_IO = parser.add_argument_group("Mesh IO")
mesh_IO.add_argument(
    "--mesh_dir",
    type=str,
    default="meshes",
    help="directory, where mesh files are saved",
)
mesh_IO.add_argument(
    "--hdf5_source",
    type=str,
    default="Omega",
    help="main hdf5 file naming with submeshing information",
)
simulation_data = parser.add_argument_group("simulation data")
simulation_data.add_argument(
    "--ic", type=str, choices=["pulse", "mms"], default="pulse"
)
simulation_data.add_argument("--pulse_mu", type=float, default=0.5)
simulation_data.add_argument("--pulse_s", type=float, default=0.2)
simulation_data.add_argument("--pulse_b", type=float, default=1.0)
simulation_data.add_argument("--pulse_amplitude_factor", type=float, default=1.0)
simulation_data.add_argument("--rhs", type=str, choices=["zero", "mms"], default="zero")
simulation_data.add_argument("--kappa", type=str, choices=["const"], default="const")
simulation_data.add_argument(
    "--kappa_value",
    type=float,
    default=1.0,
    help="value of kappa (if kappa const)",
)
fem_options = parser.add_argument_group("finite element forms")
fem_options.add_argument(
    "--mass-lump",
    "--mass_lump",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="use vertex-quadrature mass lumping (disable with --no-mass-lump)",
)
# Linear Solvers
linear_solver = parser.add_argument_group("linear solver")
linear_solver.add_argument(
    "--solvingType", type=str, choices=["direct", "iterative"], default="direct"
)
linear_solver.add_argument(
    "--direct_method", type=str, default="cholesky", help="choose direct petsc solver"
)
linear_solver.add_argument(
    "--iterative_method",
    type=str,
    default="cg",
    help="choose underlying iterative petsc solver",
)
linear_solver.add_argument(
    "--preconditioner", type=str, default="icc", help="choose preconditioner"
)
linear_solver.add_argument(
    "--max_iterations", type=int, default=1000, help="maximum number of iterations"
)
linear_solver.add_argument(
    "--tol", type=float, default=1e-12, help="tolerance for linear solver"
)
# output options
output = parser.add_argument_group("output")
output.add_argument(
    "--output_mode", type=str, choices=["last", "equidistant", "all"], default="last"
)
output.add_argument("--output_equidistant_N", type=int, default=100)
output.add_argument("--solution_dir", type=str, default="solutions")
output.add_argument("--solution_name", type=str, default="solLF")
# version
parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")
parser.add_argument(
    "-p", "--plot", help="activate optional plotting", action="store_true"
)
args = parser.parse_args()
fem = FemML if args.mass_lump else Fem
mass_form = fem.mass_form
stiffness_form = fem.stiffness_form
JIT_OPTIONS = {
    "cffi_extra_compile_args": ["-Ofast", "-march=native"],
    "cache_dir": f"{str(Path.cwd())}/.cache",
    "cffi_libraries": ["m"],
}
###
# setup arguments
###
t_before_script = time()
mesh_dir = Path(args.mesh_dir)
hdf5_sourcefile = Path(args.hdf5_source)
solvingType = args.solvingType
direct_method = args.direct_method
iterative_method = args.iterative_method
preconditioner = args.preconditioner
max_iterations = args.max_iterations
abs_tolerance = args.tol
rel_tolerance = args.tol

class MassLumping_solver:
    def __init__(self, V: dfx.fem.function.FunctionSpace, diagM: PETSc.Vec) -> None:
        self.V = V
        self.diagM = diagM

    def lumpedProject(self, rhs):
        u = dfx.fem.Function(self.V)
        u.x.petsc_vec.pointwiseDivide(rhs, self.diagM)
        return u

    def solve(self, b: PETSc.Vec, x: PETSc.Vec) -> None:
        u = self.lumpedProject(b)
        # new syntax
        dfx.fem.petsc.assign(u, x)
        # PETSC_PATH.set_bc(b, bcs=bcs)


###
# setup MPI communicators
###
comm = MPI.COMM_WORLD
subdomain_rank = comm.Get_rank()
N_subdomains = comm.Get_size()
###
# setup Logger
###
_name = sys.argv[0].split("/")[-1].split(".")[-2]
log_file = Path(f"log/{_name}.log")
if subdomain_rank == 0:
    logging.basicConfig(
        level=logging.DEBUG, datefmt="%m-%d %H:%M", filename=log_file, filemode="w"
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        datefmt="%m-%d %H:%M",
        filename=log_file,
        filemode="a",  # append for others
    )
logger = logging.getLogger(f"{_name} rank {subdomain_rank}")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)
#
if subdomain_rank == 0:
    now = datetime.now()
    logger.info(f"Started {_name} with {args}")
    for key in vars(args):
        logger.info(f"# {key} : {vars(args)[f'{key}']}")
    logger.info(f"now = {now}")
    logger.info(f"== Starting parallel run with {N_subdomains} processes")

###
#  Read in domains
###
mesh_dir = Path(args.mesh_dir)
hdf5_source_global = mesh_dir.joinpath((args.hdf5_source + ".hdf5"))
Omega = Domain(hdf5_source_global, comm, ghost_mode=dfx.mesh.GhostMode.none)
degree = Omega.V.element.basix_element.degree
if args.mass_lump and degree != 1:
    raise ValueError(
        "Mass lumping is only supported for polynomial degree 1. "
        "Use --no-mass-lump for higher-order FEM."
    )
u = dfx.fem.Function(Omega.V)
logger.debug(f"Number of local DoF's: {len(u.x.array)}")
boundary_tags = Omega.boundary_tags
material_tags = Omega.material_tags
kappa_value: float = args.kappa_value
if args.kappa == "const":
    kappa_expr = dfx.fem.Expression(
        dfx.fem.Constant(Omega.V.mesh, PETSc.ScalarType(kappa_value)),
        Omega.V.element.interpolation_points,
    )
    kappa = dfx.fem.Function(Omega.V)
    kappa.interpolate(kappa_expr)
else:
    raise NotImplementedError()
dim = Omega.V.mesh.topology.dim
fdim = dim - 1
boundary_facets_global = dfx.mesh.exterior_facet_indices(Omega.V.mesh.topology)
boundary_dofs = dfx.fem.locate_dofs_topological(Omega.V, fdim, boundary_facets_global)

# output options
solution_dir = Path(args.solution_dir)

identifier_parameters = {
    "TI": "LF",
    "N_SD": int(N_subdomains),
    "h": float(Omega.h_gmsh),
    "degree": int(Omega.degree),
    "mass": "lumped" if args.mass_lump else "consistent",
    "tau": float(args.tau),
    "T": float(args.T),
    "solve_type": "ML_solver" if args.mass_lump else solvingType,
    "prec": preconditioner,
}

local_output_folder = solution_dir.joinpath(repr(identifier_parameters))
local_output_folder.mkdir(exist_ok=True, parents=True)
local_output_filename = local_output_folder.joinpath(f"{args.solution_name}.bp")


output_equidistant_N = args.output_equidistant_N
if args.output_mode == "all":
    adios4dolfinx.write_mesh(local_output_filename, Omega.V.mesh)
    adios4dolfinx.write_meshtags(
        local_output_filename, Omega.V.mesh, material_tags, meshtag_name="material_tag"
    )

    def output(n, tn, un, vn):
        adios4dolfinx.write_function(local_output_filename, un, time=tn, name="u")
        adios4dolfinx.write_function(local_output_filename, vn, time=tn, name="v")

elif args.output_mode == "equidistant":
    adios4dolfinx.write_mesh(local_output_filename, Omega.V.mesh)
    adios4dolfinx.write_meshtags(
        local_output_filename, Omega.V.mesh, material_tags, meshtag_name="material_tag"
    )

    def output(n, tn, un, vn):
        if n % output_equidistant_N == 0:
            adios4dolfinx.write_function(local_output_filename, un, time=tn, name="u")
            adios4dolfinx.write_function(local_output_filename, vn, time=tn, name="v")

else:  # (args.output_mode == 'last')

    def output(n, tn, un, vn):
        pass


# Simulation data
t0 = 0.0
tau = args.tau
T = args.T
N_t = int(round(T / tau))
assert np.isclose(N_t * tau, T, rtol=0.0, atol=1e-10, equal_nan=False)

###
#  define Functions and Variables
###
x = ufl.SpatialCoordinate(Omega.V.mesh)
if args.ic == "pulse":
    mu = args.pulse_mu
    s = args.pulse_s
    b = args.pulse_b
    factor = args.pulse_amplitude_factor
    u0_expr, v0_expr = ic_pulse2D(mu, s, b, factor, kappa_value, x[0], x[1])
elif args.ic == "mms":
    u0_expr = (
        lambda x: np.sin(np.pi * x[0]) ** 2 * np.sin(np.pi * x[1]) ** 2 * np.exp(0)
    )
    v0_expr = (
        lambda x: np.sin(np.pi * x[0]) ** 2 * np.sin(np.pi * x[1]) ** 2 * np.exp(0)
    )
else:
    raise NotImplementedError()
# tn
tn = dfx.fem.Constant(Omega.V.mesh, t0)
tn_plus = dfx.fem.Constant(Omega.V.mesh, t0 + tau)


# dirichlet bc func
def bnd_func(t_eval):
    return lambda x: 0 * x[0]


gn = bnd_func(tn)(x)
gn_plus = bnd_func(tn_plus)(x)
# rhs
if args.rhs == 'zero':
    u_ufl = None
    def f(t_eval):
        return lambda x: 0 * x[0]
elif args.rhs == 'mms':
    def u_ufl(t_eval):
        return lambda x: ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(ufl.pi * x[1]) ** 2 * ufl.exp(t_eval)
    def f(t_eval):
        ex_sol = u_ufl(t_eval)
        return lambda x: ex_sol(x) - ufl.div(kappa * ufl.grad(ex_sol(x)))
else:
    raise NotImplementedError
fn = f(tn)(x)
fn_plus = f(tn_plus)(x)
# tau const
tau_ufl = dfx.fem.Constant(Omega.V.mesh, tau)
# solution vectors
un_plus = dfx.fem.Function(Omega.V)
vn_plus = dfx.fem.Function(Omega.V)
if args.ic == "pulse":
    un = interpolate_ufl_expression(Omega.V, u0_expr)
    vn = interpolate_ufl_expression(Omega.V, v0_expr)
elif args.ic == "mms":
    un = dfx.fem.Function(Omega.V)
    un.interpolate(u0_expr)
    vn = dfx.fem.Function(Omega.V)
    vn.interpolate(v0_expr)

###
#  Prepare Leapfrog "solve"
###
if subdomain_rank == 0:
    t_setup_0 = time()
    logger.info("preparing mass solver and rhs for leapfrog")
u = ufl.TrialFunction(Omega.V)
phi = ufl.TestFunction(Omega.V)
u_g = dfx.fem.Function(Omega.V)
bc = dfx.fem.dirichletbc(u_g, boundary_dofs)
mass_form_sys = dfx.fem.form(
    mass_form(Omega.V, u), jit_options=JIT_OPTIONS
)
MassMatrix = dfx.fem.petsc.assemble_matrix(mass_form_sys, bcs=[bc])
MassMatrix.assemble()
if args.mass_lump:
    diagonalMass = MassMatrix.getDiagonal()
    solver_M = MassLumping_solver(Omega.V, diagonalMass)
else:
    solver_M = PETSc.KSP().create(Omega.V.mesh.comm)
    solver_M.setOperators(MassMatrix)
    if solvingType == "direct":
        solver_M.setType(PETSc.KSP.Type.PREONLY)
        solver_M.getPC().setType(direct_method)
    elif solvingType == "iterative":
        solver_M.setType(iterative_method)
        solver_M.getPC().setType(preconditioner)
        solver_M.setTolerances(
            rtol=rel_tolerance, atol=abs_tolerance, max_it=max_iterations
        )
    else:
        raise NotImplementedError
    solver_M.setUp()
rhs_form = dfx.fem.form(
    mass_form(Omega.V, un)
    + tau_ufl * mass_form(Omega.V, vn)
    - tau_ufl**2 / 2 * stiffness_form(Omega.V, un, kappa)
    - tau_ufl**3 / 4 * stiffness_form(Omega.V, vn, kappa)
    + tau_ufl**2
    / 4
    * (
        mass_form(Omega.V, fn)
        + mass_form(Omega.V, fn_plus)
    ),
    jit_options=JIT_OPTIONS,
)
b = dfx.fem.petsc.create_vector(Omega.V)  #rhs form
comm.Barrier()
if subdomain_rank == 0:
    t_setup_1 = time()
    logger.info(f"System solver setup time: {t_setup_1-t_setup_0}")


###
#  prepare KSP objects
###
if subdomain_rank == 0:
    logger.info("Initialization done")
    logger.info("Starting time loop")
    t_time_loop_0 = time()

###
#  Time Loop
###
if subdomain_rank == 1:
    pbar = tqdm.tqdm(total=N_t)
n_info = max(1, int(round(N_t / 10)))

for n in range(N_t):
    tn_float = n * tau
    tn.value = tn_float
    tn_plus.value = tn_float + tau

    # leapfrog step on Omega.V
    with b.localForm() as loc_b:
        loc_b.set(0)
    dfx.fem.petsc.assemble_vector(b, rhs_form)
    bc = dfx.fem.dirichletbc(u_g, boundary_dofs)
    dfx.fem.petsc.apply_lifting(b, [mass_form_sys], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b, [bc])

    solver_M.solve(b, un_plus.x.petsc_vec)
    un_plus.x.scatter_forward()
    vn_plus.x.array[:] = 2 / tau * (un_plus.x.array[:] - un.x.array[:]) - vn.x.array[:]

    # optional output
    if subdomain_rank == 1 and (n + 1) % n_info == 0:
        pbar.update(n_info)
    output(n, tn_float, un, vn)

    # update
    un.x.array[:] = un_plus.x.array
    vn.x.array[:] = vn_plus.x.array

comm.Barrier()
if subdomain_rank == 1:
    pbar.close()
if subdomain_rank == 0:
    t_time_loop_End = time()
    t_timeloop = t_time_loop_End - t_time_loop_0
    logger.info(f"Time loop finished after time: {t_timeloop}")
logger.info(f"Max DoF value un: {np.max(un.x.array)}")

# output final solution
if args.output_mode == "last":
    adios4dolfinx.write_mesh(local_output_filename, Omega.V.mesh)
adios4dolfinx.write_function(local_output_filename, un, time=T, name="u")
adios4dolfinx.write_function(local_output_filename, vn, time=T, name="v")
#
if subdomain_rank == 0:
    simulation_summary(t_timeloop, Omega.num_global_dofs, N_t, N_subdomains, 2)

###
#  copy log files
###
if subdomain_rank == 0:
    shutil.copy(log_file, local_output_folder)
    submeshing_log_file = Path("log/submeshing.log")
    shutil.copy(submeshing_log_file, local_output_folder)

###
#  destroy petsc objects
###
if args.mass_lump:
    solver_M.diagM.destroy()
    del solver_M
else:
    solver_M.destroy()
MassMatrix.destroy()
b.destroy()


###
#  optional error calculation
###
if args.rhs == "mms" and args.ic == "mms":
    from Norms import error_norm
    def u_exakt(t_eval):
        return (
            lambda x: np.sin(np.pi * x[0]) ** 2
            * np.sin(np.pi * x[1]) ** 2
            * np.exp(t_eval)
        )
    def v_exakt(t_eval):
        return (
            lambda x: np.sin(np.pi * x[0]) ** 2
            * np.sin(np.pi * x[1]) ** 2
            * np.exp(t_eval)
        )

    error_L2_un, norm_L2_un_ref = error_norm(
        un, u_exakt(T), "L2", degree_raise=1, relative=True
    )
    error_H1_un, norm_H1_un_ref = error_norm(
        un, u_exakt(T), "H10", degree_raise=1, relative=True
    )
    error_L2_vn, norm_L2_vn_ref = error_norm(
        vn, v_exakt(T), "L2", degree_raise=1, relative=True
    )

    if subdomain_rank == 0:
        error_Xh = np.sqrt(error_H1_un**2 + error_L2_vn**2)
        norm_Xh = np.sqrt(norm_H1_un_ref**2 + norm_L2_vn_ref**2)
        logger.info(f"L2 norm of un reference:{norm_L2_un_ref}")
        logger.info(f"H1 norm of un reference:{norm_H1_un_ref}")
        logger.info(f"L2 norm of vn reference:{norm_L2_vn_ref}")
        logger.info(f"Xh norm of reference:{norm_Xh}")
        ###
        logger.info(f"rel L2 error of un on Omega: {error_L2_un}")
        logger.info(f"rel H1 error of un on Omega: {error_H1_un}")
        logger.info(f"rel L2 error of vn on Omega: {error_L2_vn}")
        logger.info(f"rel Xh error: {error_Xh}")

elif subdomain_rank == 0:
    logger.info("No error calculation performed as example is not mms")


###
#  optional plot
###
if args.plot:
    comm_self = MPI.COMM_SELF
    from Plotting import plot_sol_pyvista

    if subdomain_rank == 0:
        read_mesh = adios4dolfinx.read_mesh(local_output_filename, comm_self)
        read_V = dfx.fem.functionspace(read_mesh, ("Lagrange", degree))
        read_u = dfx.fem.Function(read_V)
        adios4dolfinx.read_function(local_output_filename, read_u, time=T, name="u")
        filename = Path(f"plot/LeapfrogPlot.png")
        plot_sol_pyvista(read_V, read_u, filename)
