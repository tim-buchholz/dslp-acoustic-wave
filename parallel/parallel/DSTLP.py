#!/usr/bin/env python3

# usage: mpirun -np N_PROCESSES [--oversubscribe] DSTLP
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
from Fem import mass_form, stiffness_form
from Simulation import simulation_summary
import tqdm
import math
from Functions import interpolate_ufl_expression, ic_pulse2D
###
# CLI definition
###
import argparse

parser = argparse.ArgumentParser(
    prog="mpirun -np N_PROCESSES [--oversubscribe] DSTLP",
    description=">>> DSTLP-FEM: Parallel domain splitting with test-localized prediction (%(prog)s tau T) <<<",
    epilog=">>> DSTLP-FEM Script <<<",
)
# time stepping arguments
time_stepping = parser.add_argument_group("time stepping")
time_stepping.add_argument("tau", type=float, help="time step size")
time_stepping.add_argument("T", type=float, help="final time")
prediction = parser.add_argument_group("test-localized prediction")
prediction.add_argument(
    "--ell_p1",
    type=int,
    required=True,
    help="number of prediction layers on which the test function is active",
)
prediction.add_argument(
    "--ell_p2",
    type=int,
    required=True,
    help="number of outer buffer layers in the prediction domain",
)
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
simulation_data.add_argument("--ic", type=str, choices=["pulse", "mms"], default="pulse")
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
output.add_argument("--solution_name", type=str, default="solDSTLP")
output.add_argument(
    "--write_global_solution",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="write a root-owned reconstructed global solution next to the rank-local outputs",
)
# version
parser.add_argument(
    "--use_alloc_free_comms",
    action="store_true",
    default=False,
    help="use allocation-free communication (persistent MPI requests)",
)
parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")
parser.add_argument(
    "-p", "--plot", help="activate optional plotting", action="store_true"
)
args = parser.parse_args()
if args.ell_p1 < 1:
    parser.error("--ell_p1 must be at least 1")
if args.ell_p2 < 0:
    parser.error("--ell_p2 must be non-negative")
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
# Linear Solver arguments
solvingType = args.solvingType
direct_method = args.direct_method
iterative_method = args.iterative_method
preconditioner = args.preconditioner
max_iterations = args.max_iterations
abs_tolerance = args.tol
rel_tolerance = args.tol


def LinearSolver(
    form: dfx.fem.forms.Form,
    V: dfx.fem.function.FunctionSpace,
    bcs: list[dfx.fem.bcs.DirichletBC] = [],
) -> PETSc.Mat | PETSc.KSP:
    A = dfx.fem.petsc.assemble_matrix(form, bcs=bcs)
    A.assemble()
    solver = PETSc.KSP().create(V.mesh.comm)
    solver.setOperators(A)
    if solvingType == "direct":
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(direct_method)
    elif solvingType == "iterative":
        solver.setType(iterative_method)
        solver.getPC().setType(preconditioner)
        solver.setTolerances(
            rtol=rel_tolerance, atol=abs_tolerance, max_it=max_iterations
        )
    else:
        raise NotImplementedError
    return A, solver


###
# setup MPI communicators
###
comm = MPI.COMM_WORLD
comm_self = MPI.COMM_SELF
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
    logger.info(f"Started parallel DSTLP method with {args}")
    for key in vars(args):
        logger.info(f"# {key} : {vars(args)[f'{key}']}")
    logger.info(f"now = {now}")
    logger.info(f"== Starting parallel run with {N_subdomains} processes")

###
#  Read in domains
###
mesh_dir = Path(args.mesh_dir)
hdf5_source_global = mesh_dir.joinpath((args.hdf5_source + ".hdf5"))
hdf5_source_local = mesh_dir.joinpath(
    (args.hdf5_source + f"_{subdomain_rank}^delta.hdf5")
)
logger.info("Load and initialize SubDomains")
Omega_i_delta = SubDomain(str(hdf5_source_local), comm, comm_self)
requested_prediction_ell = args.ell_p1 + args.ell_p2
if requested_prediction_ell != Omega_i_delta.prediction_ell:
    raise ValueError(
        "DSTLP prediction widths do not match the preprocessed mesh: "
        f"ell_p1 + ell_p2 = {requested_prediction_ell}, "
        f"prediction_ell = {Omega_i_delta.prediction_ell}. "
        "Rerun submeshing.py with --pred_ell equal to ell_p1 + ell_p2."
    )
degree = Omega_i_delta.V.element.basix_element.degree
# material on Omega_i_delta
boundary_tags = Omega_i_delta.boundary_tags
material_tags = Omega_i_delta.material_tags

kappa_value : float = args.kappa_value
if args.kappa == "const":
    kappa_expr = dfx.fem.Expression(
        dfx.fem.Constant(Omega_i_delta.V.mesh, PETSc.ScalarType(kappa_value)),
        Omega_i_delta.V.element.interpolation_points,
    )
    kappa = dfx.fem.Function(Omega_i_delta.V)
    kappa.interpolate(kappa_expr)
else:
    raise NotImplementedError()
# material in \Omega_{\Gamma,i}^{\delta}
prediction_boundary_tags = Omega_i_delta.prediction_boundary_tags
prediction_material_tags = Omega_i_delta.prediction_material_tags
if args.kappa == "const":
    kappa_pred_expr = dfx.fem.Expression(
        dfx.fem.Constant(Omega_i_delta.pred_V.mesh, PETSc.ScalarType(kappa_value)),
        Omega_i_delta.pred_V.element.interpolation_points,
    )
    kappa_pred = dfx.fem.Function(Omega_i_delta.pred_V)
    kappa_pred.interpolate(kappa_pred_expr, cells0=prediction_material_tags.find(1))
else:
    raise NotImplementedError()
comm.Barrier()
logger.info(f"SubDomain {subdomain_rank} ready")
_sync = (
    Omega_i_delta.communicate_and_sync_allocation_free
    if args.use_alloc_free_comms
    else Omega_i_delta.communicate_and_sync
)

# check h_min and CFL
hmin_loc = get_h(Omega_i_delta.mesh, mode="min")
hmin = comm.allreduce(hmin_loc, op=MPI.MIN)
if subdomain_rank == 0:
    logger.info(f"Minimum h in all meshes: {hmin}")

tauCfl_heuristic = (
    Omega_i_delta.ell * 2 * hmin / (args.kappa_value * degree)
)
if args.tau > tauCfl_heuristic:
    logger.warning("Make sure, that the CFL is satisfied. Heuristical it is not.")
propagation_layers = math.ceil(args.tau * args.kappa_value / hmin)
if args.ell_p1 < propagation_layers or args.ell_p2 < propagation_layers:
    logger.warning(
        "Prediction widths (%d, %d) are below the propagation heuristic (%d, %d).",
        args.ell_p1,
        args.ell_p2,
        propagation_layers,
        propagation_layers,
    )

# output options
solution_dir = Path(args.solution_dir)

identifier_parameters = {
    "TI": "DSTLP",
    "N_SD": int(N_subdomains),
    "h": float(Omega_i_delta.h_gmsh),
    "degree": int(Omega_i_delta.degree),
    "ell": int(Omega_i_delta.ell),
    "p_ell": int(Omega_i_delta.prediction_ell),
    "ell_p1": int(args.ell_p1),
    "ell_p2": int(args.ell_p2),
    "tau": float(args.tau),
    "T": float(args.T),
    "solve_type": solvingType,
    "prec": preconditioner
}

local_output_folder = solution_dir.joinpath(repr(identifier_parameters))
local_output_folder.mkdir(exist_ok=True, parents=True)
local_output_filename = local_output_folder.joinpath(
    f"{args.solution_name}_{subdomain_rank}.bp"
)
output_equidistant_N = args.output_equidistant_N
if args.output_mode == "all":
    adios4dolfinx.write_mesh(local_output_filename, Omega_i_delta.V.mesh)
    def output(n, tn, un, vn):
        adios4dolfinx.write_function(local_output_filename, un, time=tn, name="u")
        adios4dolfinx.write_function(local_output_filename, vn, time=tn, name="v")
elif args.output_mode == "equidistant":
    adios4dolfinx.write_mesh(local_output_filename, Omega_i_delta.V.mesh)
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
assert np.isclose(N_t * tau, T, rtol=0.0, atol=args.tol, equal_nan=False)


###
#  define Functions and Variables
###
x = ufl.SpatialCoordinate(Omega_i_delta.V.mesh)
x_pred = ufl.SpatialCoordinate(Omega_i_delta.pred_V.mesh)
# ic
if args.ic == "pulse":
    mu = args.pulse_mu
    s = args.pulse_s
    b = args.pulse_b
    factor = args.pulse_amplitude_factor
    u0_expr, v0_expr = ic_pulse2D(mu, s, b, factor, kappa_value, x[0], x[1])
    u0_pred_expr, v0_pred_expr = ic_pulse2D(mu, s, b, factor, kappa_value, x_pred[0], x_pred[1])
elif args.ic == 'mms':
    u0_expr = (
        lambda x: np.sin(np.pi * x[0]) ** 2 * np.sin(np.pi * x[1]) ** 2 * np.exp(0)
    )
    v0_expr = (
        lambda x: np.sin(np.pi * x[0]) ** 2 * np.sin(np.pi * x[1]) ** 2 * np.exp(0)
    )
else:
    raise NotImplementedError()
# tn
tn = dfx.fem.Constant(Omega_i_delta.V.mesh, t0)
tn_plus = dfx.fem.Constant(Omega_i_delta.V.mesh, t0 + tau)
tn_pred = dfx.fem.Constant(Omega_i_delta.pred_V.mesh, t0)
tn_plus_pred = dfx.fem.Constant(Omega_i_delta.pred_V.mesh, t0 + tau)
# dirichlet bc func
def bnd_func(t_eval):
    return lambda x: 0 * x[0]
gn = bnd_func(tn)(x)
gn_plus = bnd_func(tn_plus)(x)
gn_pred = bnd_func(tn_pred)(x_pred)
gn_plus_pred = bnd_func(tn_plus_pred)(x_pred)
# rhs
if args.rhs == 'zero':
    u_ufl = None
    def f(t_eval):
        return lambda x: 0 * x[0]
    def f_pred(t_eval):
        return lambda x: 0 * x[0]
elif args.rhs == 'mms':
    def u_ufl(t_eval):
        return lambda x: ufl.sin(ufl.pi * x[0]) ** 2 * ufl.sin(ufl.pi * x[1]) ** 2 * ufl.exp(t_eval)
    def f(t_eval):
        ex_sol = u_ufl(t_eval)
        return lambda x: ex_sol(x) - ufl.div(kappa * ufl.grad(ex_sol(x)))
    def f_pred(t_eval):
        ex_sol = u_ufl(t_eval)
        return lambda x: ex_sol(x) - ufl.div(kappa_pred * ufl.grad(ex_sol(x)))
else:
    raise NotImplementedError

fn = f(tn)(x)
fn_plus = f(tn_plus)(x)
fn_pred = f_pred(tn_pred)(x_pred)
fn_plus_pred = f_pred(tn_plus_pred)(x_pred)


# tau const
tau_ufl = dfx.fem.Constant(Omega_i_delta.V.mesh, tau)
tau_pred_ufl = dfx.fem.Constant(Omega_i_delta.pred_V.mesh, tau)
# solution vectors
un_plus_pred = dfx.fem.Function(Omega_i_delta.pred_V)
un_plus = dfx.fem.Function(Omega_i_delta.V)
vn_plus = dfx.fem.Function(Omega_i_delta.V)

if args.ic == "pulse":
    un = interpolate_ufl_expression(Omega_i_delta.V, u0_expr)
    vn = dfx.fem.Function(Omega_i_delta.V)
    vn = interpolate_ufl_expression(Omega_i_delta.V, v0_expr)
    un_pred = interpolate_ufl_expression(Omega_i_delta.pred_V, u0_pred_expr)
    vn_pred = dfx.fem.Function(Omega_i_delta.pred_V)
    vn_pred = interpolate_ufl_expression(Omega_i_delta.pred_V, v0_pred_expr)
elif args.ic == 'mms':
    un = dfx.fem.Function(Omega_i_delta.V)
    un.interpolate(u0_expr)
    vn = dfx.fem.Function(Omega_i_delta.V)
    vn.interpolate(v0_expr)
    un_pred = dfx.fem.Function(Omega_i_delta.pred_V)
    un_pred.interpolate(u0_expr)
    vn_pred = dfx.fem.Function(Omega_i_delta.pred_V)
    vn_pred.interpolate(v0_expr)
###
#  Prepare test-localized Crank-Nicolson increment
###
if subdomain_rank == 0:
    logger.info("preparing test-localized Crank-Nicolson prediction")
u_pred = ufl.TrialFunction(Omega_i_delta.pred_V)
zero_pred = dfx.fem.Function(Omega_i_delta.pred_V)
bc_pred = dfx.fem.dirichletbc(zero_pred, Omega_i_delta.pred_boundary_dofs)
system_form_pred = dfx.fem.form(
    mass_form(Omega_i_delta.pred_V, u_pred)
    + tau_pred_ufl**2
    / 4
    * stiffness_form(Omega_i_delta.pred_V, u_pred, kappa_pred),
    jit_options=JIT_OPTIONS,
)
A_pred, solverPrediction = LinearSolver(
    system_form_pred, Omega_i_delta.pred_V, bcs=[bc_pred]
)
rhs_form_pred = dfx.fem.form(
    tau_pred_ufl * mass_form(Omega_i_delta.pred_V, vn_pred)
    + tau_pred_ufl**2
    / 4
    * (
        mass_form(Omega_i_delta.pred_V, fn_pred)
        + mass_form(Omega_i_delta.pred_V, fn_plus_pred)
    )
    - tau_pred_ufl**2
    / 2
    * stiffness_form(Omega_i_delta.pred_V, un_pred, kappa_pred),
    jit_options=JIT_OPTIONS,
)
b_pred = dfx.fem.petsc.create_vector(Omega_i_delta.pred_V)
prediction_increment = dfx.fem.Function(Omega_i_delta.pred_V)
prediction_cutoff = Omega_i_delta.prediction_cutoff(args.ell_p1)
###
#  Prepare local Crank Nicolson solves
###
if subdomain_rank == 0:
    logger.info("preparing system solver and rhs for local Crank-Nicolson solves")
u = ufl.TrialFunction(Omega_i_delta.V)
phi = ufl.TestFunction(Omega_i_delta.V)
u_g = dfx.fem.Function(Omega_i_delta.V)

outer_bc = dfx.fem.dirichletbc(u_g, Omega_i_delta.ov_boundary_without_interface_dofs_local)

pred = dfx.fem.Function(Omega_i_delta.V)
interface_bc = dfx.fem.dirichletbc(
    pred, Omega_i_delta.ov_interface_dofs_local
)
theta_ufl = dfx.fem.Constant(Omega_i_delta.mesh, 1 / 4)
system_form = dfx.fem.form(
    mass_form(Omega_i_delta.V, u)
    + theta_ufl * tau_ufl**2 * stiffness_form(Omega_i_delta.V, u, kappa),
    jit_options=JIT_OPTIONS,
)
A_sys, solverSystem = LinearSolver(system_form, Omega_i_delta.V, bcs = [outer_bc, interface_bc])
rhs_form = dfx.fem.form(
    mass_form(Omega_i_delta.V, un)
    + tau_ufl * mass_form(Omega_i_delta.V, vn)
    + tau_ufl**2 / 4 * (mass_form(Omega_i_delta.V, fn) + mass_form(Omega_i_delta.V, fn_plus))
    - tau_ufl**2 / 4 * stiffness_form(Omega_i_delta.V, un, kappa),
    jit_options=JIT_OPTIONS,
)
b = dfx.fem.petsc.create_vector(Omega_i_delta.V)  # rhs_form
comm.Barrier()


###
#  prepare KSP objects
###
if subdomain_rank == 0:
    logger.info("Initialization done")
    t_begin = time()
    logger.info("Calling setUp on KSP solver objects:")
solverSystem.setUp()
solverPrediction.setUp()
if subdomain_rank == 0:
    t_system_setUp = time()
    logger.info(f"System solver setup time: {t_system_setUp-t_begin}")
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
    tn_pred.value = tn_float
    tn_plus_pred.value = tn_float + tau

    # step 1: test-localized CN increment on Omega_i_delta.pred_V
    with b_pred.localForm() as loc_b_pred:
        loc_b_pred.set(0)
    dfx.fem.petsc.assemble_vector(b_pred, rhs_form_pred)
    b_pred.array[:] *= prediction_cutoff.x.array
    dfx.fem.petsc.apply_lifting(
        b_pred, [system_form_pred], bcs=[[bc_pred]]
    )
    b_pred.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    dfx.fem.set_bc(b_pred, [bc_pred])
    solverPrediction.solve(b_pred, prediction_increment.x.petsc_vec)
    prediction_increment.x.scatter_forward()
    un_plus_pred.x.array[:] = (
        un_pred.x.array + prediction_increment.x.array
    )

    pred.x.array[:] = 0.0
    pred.x.array[Omega_i_delta.ov_interface_dofs_local] = un_plus_pred.x.array[
        Omega_i_delta.pred_dofs_local
    ]
    interface_bc = dfx.fem.dirichletbc(
        pred, Omega_i_delta.ov_interface_dofs_local
    )
    # un_plus_pred_proj = Omega_i_delta.restrict_from_pred_to_ov_SD(un_plus_pred)
    # interface_bc = dfx.fem.dirichletbc(
    #     un_plus_pred_proj, Omega_i_delta.ov_interface_dofs_local
    # )

    # step 2: local calculation by Crank Nicolson on Omega_i_delta.V
    with b.localForm() as loc_b:
        loc_b.set(0)
    dfx.fem.petsc.assemble_vector(b, rhs_form)
    # u_g.interpolate(gn)
    outer_bc = dfx.fem.dirichletbc(u_g, Omega_i_delta.ov_boundary_without_interface_dofs_local)
    dfx.fem.petsc.apply_lifting(b, [system_form], bcs=[[outer_bc, interface_bc]])
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    dfx.fem.set_bc(b, [outer_bc, interface_bc])
    solverSystem.solve(b, un_plus.x.petsc_vec)
    vn_plus.x.array[:] = 2/tau * (un_plus.x.array[:] - un.x.array[:]) - vn.x.array[:]

    # step 3: update the overlapping regions by communicate_and_sync
    # With --use_alloc_free_comms returned functions are double-buffered
    # pre-allocated aliases; slot alternates each call so un_p_sync and
    # vn_p_sync occupy different buffers and remain valid until the update
    # block below.
    un_p_sync, un_p_sync_pred = _sync(un_plus)
    vn_p_sync, vn_p_sync_pred = _sync(vn_plus)

    # optional output
    if subdomain_rank == 1 and (n + 1) % n_info == 0:
        pbar.update(n_info)
    output(n, tn_float, un, vn)

    # update
    un.x.array[:] = un_p_sync.x.array
    vn.x.array[:] = vn_p_sync.x.array
    Omega_i_delta.merge_prediction_state(
        un_pred, un_p_sync, un_p_sync_pred
    )
    Omega_i_delta.merge_prediction_state(
        vn_pred, vn_p_sync, vn_p_sync_pred
    )


comm.Barrier()
Omega_i_delta.free_persistent_requests()
if subdomain_rank == 1:
    pbar.close()
if subdomain_rank == 0:
    t_time_loop_End = time()
    t_timeloop = t_time_loop_End - t_time_loop_0
    logger.info(f"Time loop finished after time: {t_timeloop}")
logger.info(f"Max DoF value un: {np.max(un.x.array)}")

# destroy petsc objects
A_sys.destroy()
solverSystem.destroy()
A_pred.destroy()
solverPrediction.destroy()
b.destroy()
b_pred.destroy()


# output final solution
if args.output_mode == "last":
    adios4dolfinx.write_mesh(local_output_filename, Omega_i_delta.V.mesh)
adios4dolfinx.write_function(local_output_filename, un, time=T, name="u")
adios4dolfinx.write_function(local_output_filename, vn, time=T, name="v")
#
if subdomain_rank == 0:
    simulation_summary(t_timeloop, Omega_i_delta.num_global_dofs, N_t, N_subdomains, 2)
###
#  copy log files
###
if subdomain_rank == 0:
    shutil.copy(log_file, local_output_folder)
    submeshing_log_file = Path("log/submeshing.log")
    shutil.copy(submeshing_log_file, local_output_folder)

###
#  optional error calculation
###
if args.rhs == 'mms' and args.ic == 'mms':
    from Norms import error_norm, norm_L2, norm_H10
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
    Ri_un = Omega_i_delta.restrict_from_ov_to_non_ov_SD(un)
    Ri_vn = Omega_i_delta.restrict_from_ov_to_non_ov_SD(vn)

    error_L2_un_Omega_i = error_norm(
        Ri_un, u_exakt(T), "L2", degree_raise=1, relative=False
    )
    error_H1_un_Omega_i = error_norm(
        Ri_un, u_exakt(T), "H10", degree_raise=1, relative=False
    )
    error_L2_vn_Omega_i = error_norm(
        Ri_vn, v_exakt(T), "L2", degree_raise=1, relative=False
    )
    error_L2_un_abs = np.sqrt(comm.allreduce(error_L2_un_Omega_i**2, op=MPI.SUM))
    error_H1_un_abs = np.sqrt(comm.allreduce(error_H1_un_Omega_i**2, op=MPI.SUM))
    error_L2_vn_abs = np.sqrt(comm.allreduce(error_L2_vn_Omega_i**2, op=MPI.SUM))
    u_ref = dfx.fem.Function(Omega_i_delta.nonov_V)
    u_ref.interpolate(u_exakt(T))
    v_ref = dfx.fem.Function(Omega_i_delta.nonov_V)
    v_ref.interpolate(v_exakt(T))
    norm_L2_un_ref_local = norm_L2(Omega_i_delta.nonov_V, u_ref)
    norm_H1_un_ref_local = norm_H10(Omega_i_delta.nonov_V, u_ref)
    norm_L2_vn_ref_local = norm_L2(Omega_i_delta.nonov_V, v_ref)
    norm_L2_un_ref = np.sqrt(comm.allreduce(norm_L2_un_ref_local**2, op=MPI.SUM))
    norm_H1_un_ref = np.sqrt(comm.allreduce(norm_H1_un_ref_local**2, op=MPI.SUM))
    norm_L2_vn_ref = np.sqrt(comm.allreduce(norm_L2_vn_ref_local**2, op=MPI.SUM))

    if subdomain_rank == 0:
        error_Xh_abs = np.sqrt(error_H1_un_abs**2 + error_L2_vn_abs**2)
        norm_Xh = np.sqrt(norm_H1_un_ref**2 + norm_L2_vn_ref**2)
        error_L2_un_rel = error_L2_un_abs / norm_L2_un_ref
        error_H1_un_rel = error_H1_un_abs / norm_H1_un_ref
        error_L2_vn_rel = error_L2_vn_abs / norm_L2_vn_ref
        error_Xh_rel = error_Xh_abs / norm_Xh

        logger.info(f"abs L2 error of qn on Omega: {error_L2_un_abs}")
        logger.info(f"abs H1 error of qn on Omega: {error_H1_un_abs}")
        logger.info(f"abs L2 error of pn on Omega: {error_L2_vn_abs}")
        logger.info(f"abs Xh error: {error_Xh_abs}")
        logger.info(f"L2 norm of un reference:{norm_L2_un_ref}")
        logger.info(f"H1 norm of un reference:{norm_H1_un_ref}")
        logger.info(f"L2 norm of vn reference:{norm_L2_vn_ref}")
        logger.info(f"Xh norm of reference:{norm_Xh}")
        logger.info(f"rel L2 error of un on Omega: {error_L2_un_rel}")
        logger.info(f"rel H1 error of un on Omega: {error_H1_un_rel}")
        logger.info(f"rel L2 error of vn on Omega: {error_L2_vn_rel}")
        logger.info(f"rel Xh error: {error_Xh_rel}")
elif subdomain_rank == 0:
    logger.info("No error calculation performed as example is not mms")


###
#  directly write or plot global solution (if desired)
###
if args.plot or args.write_global_solution:
    Omega = None
    if subdomain_rank == 0:
        logger.info("Performing global reconstruction")
        Omega = Domain(hdf5_source_global, comm_self)
    comm.Barrier()

    # global reconstruction on rank 0
    global_un = Omega_i_delta.reconstruct_global_approximation_on_root(comm, Omega, un)
    global_vn = Omega_i_delta.reconstruct_global_approximation_on_root(comm, Omega, vn)
    global_output_filename = local_output_folder.joinpath(f"{args.solution_name}.bp")
    comm.Barrier()
    if subdomain_rank == 0:
        adios4dolfinx.write_mesh(global_output_filename, Omega.V.mesh)
        adios4dolfinx.write_function(global_output_filename, global_un, time=T, name="u")
        adios4dolfinx.write_function(global_output_filename, global_vn, time=T, name="v")
    comm.Barrier()
    if subdomain_rank == 0:
        logger.info("Wrote global reconstruction to file")

    if args.plot:
        from Plotting import plot_sol_pyvista, plot_sol_with_object
        if subdomain_rank == 0:
            filename = Path("plot/DSTLPFirstSol.png")
            plot_sol_pyvista(Omega.V, global_un, filename)

            if args.rhs == 'mms' and args.ic == 'mms':
                filename = Path(f"plot/u_exakt.png")
                un_glob_exakt = dfx.fem.Function(Omega.V)
                un_glob_exakt.interpolate(u_exakt(T))
                plot_sol_pyvista(Omega.V, un_glob_exakt, filename)

# logging.shutdown()
comm.Barrier()
# MPI.Finalize()
