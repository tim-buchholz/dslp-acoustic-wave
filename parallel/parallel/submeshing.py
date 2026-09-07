#!/usr/bin/env python3

# usage: mpirun -np N_PROCESSES [--oversubscribe] submeshing.py
#       [-h] [--width WIDTH] [--height HEIGHT]
#       [--mesh-refinement MESH_REFINEMENT]
#      [--partitioner {scotch,kahip,parmetis}] [--pred_ell PRED_ELL]
#       [--DoF_tol DOF_TOL] [--mesh_dir MESH_DIR] [-p] [--plot_dir PLOT_DIR]
#       [-v]
#       h polynomial_degree ell

import os

# Prevent thread explosion in BLAS / LAPACK / MKL / OpenMP:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from mpi4py import MPI
import dolfinx as dfx
import numpy as np
from time import time, sleep
import sys
import os
import h5py
from datetime import datetime
import pyvista

sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
sys.path.append("src")
from Plotting import plot_sol_pyvista, plot_mesh_with_material, plot_mesh_with_face_marker
from Meshing import *
from CommunicationPlan import schedule_from_comms, flatten_schedule, unflatten_schedule
from typing import Any
import logging

###
# CLI definition
###
import argparse

parser = argparse.ArgumentParser(
    prog=f"mpirun -np N_PROCESSES [--oversubscribe] submeshing.py ",
    description=">>> submeshing help (%(prog)s) <<<",
    epilog=">>>  SubMeshing Script  <<<",
)
# global mesh group
global_mesh = parser.add_argument_group("global mesh")

global_mesh.add_argument("h", type=float, help="discretization parameter")

global_mesh.add_argument(
    "polynomial_degree", type=int, default=1, help="polynomial degree of discretization"
)
global_mesh.add_argument(
    "ell", type=int, help="overlap parameter (number of additional layers)"
)
global_mesh.add_argument(
    "--width", type=float, default=1.0, help="width of the underlying rectangle"
)
global_mesh.add_argument(
    "--height", type=float, default=1.0, help="height of the underlying rectangle"
)
global_mesh.add_argument(
    "--mesh-refinement", type=float, default=1, help="mesh refinement factor"
)
global_mesh.add_argument(
    "--refinement-mode", choices=["center", "axes"],
    default="center", help="mesh refinement mode"
)
# partitioner
global_mesh.add_argument(
    "--partitioner",
    choices=["scotch", "kahip", "parmetis"],
    default="scotch",
    help="name of the partitioner",
)
# prediction domain overlap parameter
parser.add_argument(
    "--pred_ell", type=int, default=1, help="overlap parameter predicition domain"
)
# DoF tolerance
parser.add_argument(
    "--DoF_tol",
    type=float,
    default=1e-8,
    help="DoF (degree of freedom) tolerance for boundary checks etc",
)
# mesh_dir
parser.add_argument("--mesh_dir", default="meshes", help="destination to store meshes")
# plt_meshes
plotting_meshes = parser.add_argument_group("plotting meshes")
plotting_meshes.add_argument(
    "-p", "--plot", help="activate optional plotting", action="store_true"
)
plotting_meshes.add_argument(
    "--plot_dir", default="plot", help="destination to plot meshes to"
)
# version
parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")
args = parser.parse_args()

###
# setup arguments
###
t_before_script = time()
h = args.h
degree = args.polynomial_degree
ell = args.ell
DoF_tol = args.DoF_tol
plot_dir = Path(args.plot_dir)
mesh_dir = Path(args.mesh_dir)

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
    logger.info(f"Started submeshing routine with {args}")
    for key in vars(args):
        logger.info(f"# {key} : {vars(args)[f'{key}']}")
    logger.info(f"now = {now}")
    logger.info(f"== Starting parallel script with {N_subdomains} processes")

######
###
###   Submeshing phase: build all meshes, transfer meshtags and generate
###                     local-to-global DoFmaps
###
######

###
# Create fulldomain and write to file with material tags
###
t0 = time()
if subdomain_rank == 0:
    logger.info(f"Creating global mesh")
global_mesh_filename, n_total_dofs = write_global_mesh(comm, args, mesh_dir)
comm.Barrier()
t1 = time()


if subdomain_rank == 0:
    logger.info(f"Total time for global meshing and partitioning: {t1-t0}")


###
# Prepare global mesh on each rank within comm_self
###
global_mesh, read_meshtags = read_mesh_with_meshtags(
    Path(mesh_dir.joinpath("Omega.bp")), comm_self, ["material_tag", "subdomain_tag", "boundary_tag"]
)
dim = global_mesh.topology.dim
fdim = dim - 1
vdim = 0
global_mesh.topology.create_connectivity(fdim, dim)
global_mesh.topology.create_connectivity(vdim, dim)
material_tags = read_meshtags["material_tag"]
subdomain_tags = read_meshtags["subdomain_tag"]
boundary_tags = read_meshtags["boundary_tag"]
V_global = dfx.fem.functionspace(global_mesh, ('Lagrange', degree))
boundary_facets_global = dfx.mesh.exterior_facet_indices(global_mesh.topology)
boundary_dofs_global = dfx.fem.locate_dofs_topological(
    V_global, fdim, boundary_facets_global, remote=True
)


###
# extract non-overlapping subdomain from partition via subdomain tags
###
subdomain_entities = subdomain_tags.find(subdomain_rank)
non_overlapping_submesh, non_ov_cell_map, non_ov_vertex_map, non_ov_node_map = dfx.mesh.create_submesh(
    global_mesh, dim, subdomain_entities
)
non_overlapping_submesh.topology.create_connectivity(fdim, dim)
# non_overlapping_submesh.topology.create_connectivity(vdim, dim) # not needed
non_overlapping_material_tags = get_material_marker_for_submesh(
    global_mesh, material_tags, non_overlapping_submesh, non_ov_cell_map, non_ov_vertex_map
)
non_overlapping_boundary_tags = get_boundary_marker_for_submesh(
    global_mesh,
    boundary_tags,
    (100, 101),
    non_overlapping_submesh,
    non_ov_cell_map,
    non_ov_vertex_map,
)

hmin_loc = get_h(non_overlapping_submesh, mode="min")
hmin = comm.allreduce(hmin_loc, op=MPI.MIN)

comm.Barrier()
t2 = time()
if subdomain_rank == 0:
    logger.info(
        f"Global mesh has {V_global.dofmap.index_map.size_global} DoFs (per component)"
    )
    logger.info(f"h_min in global mesh: {hmin}")
    logger.info(f"time to read global mesh on each rank: {t2-t1}")

# # fix reordering after read
non_overlapping_mesh_file = Path(mesh_dir.joinpath(f"SD_{subdomain_rank}.bp"))
non_overlapping_submesh, tag_dict = save_and_load_mesh_with_meshtags(
    non_overlapping_mesh_file,
    comm_self,
    non_overlapping_submesh,
    [non_overlapping_material_tags, non_overlapping_boundary_tags],
    ["material_tag", "boundary_tag"],
)
non_overlapping_material_tags = tag_dict["material_tag"]
non_overlapping_boundary_tags = tag_dict["boundary_tag"]


cell_index_map = non_overlapping_submesh.topology.index_map(dim)
num_cells_non_ov = cell_index_map.size_local + cell_index_map.num_ghosts 

np_non_ov_cell_map = non_ov_cell_map.sub_topology_to_topology(
    np.arange(num_cells_non_ov), inverse=False
)[non_overlapping_submesh.topology.original_cell_index]

V_SD_i = dfx.fem.functionspace(non_overlapping_submesh, ("Lagrange", degree))
non_ov_l2gmap = generate_dofmap(V_global, V_SD_i, np_non_ov_cell_map)
n_nonov_dofs = len(non_ov_l2gmap)

boundary_facets_nonov = dfx.mesh.exterior_facet_indices(non_overlapping_submesh.topology)
boundary_dofs_nonov = dfx.fem.locate_dofs_topological(
    V_SD_i, fdim, boundary_facets_nonov
) 
interface_dofs_nonov_global = np.setdiff1d(
    non_ov_l2gmap[boundary_dofs_nonov], boundary_dofs_global, assume_unique=True
)

comm.Barrier()
t3 = time()
if args.plot:
    plot_mesh_with_material(
        non_overlapping_submesh,
        non_overlapping_material_tags,
        plot_dir.joinpath(f"mesh_{subdomain_rank}.png"),
    )
if subdomain_rank == 0:
    logger.info(f"time for non overlapping subdomain generation: {t3-t2}")
comm.Barrier()
t4 = time()
# ###
# # build overlapping subdomains as cell extension from non-overlapping subdomains
# ###
overlapping_cells = extend_cells_by_layers(
    global_mesh=global_mesh, reference_cells=np_non_ov_cell_map, num_layers=ell
)
overlapping_submesh, ov_cell_map, ov_vertex_map, ov_node_map = dfx.mesh.create_submesh(
    global_mesh, dim, overlapping_cells
)
overlapping_submesh.topology.create_connectivity(fdim, dim)
overlapping_submesh.topology.create_connectivity(vdim, dim)
overlapping_material_tags = get_material_marker_for_submesh(
    global_mesh, material_tags, overlapping_submesh, ov_cell_map, ov_vertex_map
)
overlapping_boundary_tags = get_boundary_marker_for_submesh(
    global_mesh,
    boundary_tags,
    (100, 101),
    overlapping_submesh,
    ov_cell_map,
    ov_vertex_map,
)
# fix reordering of overlapping subdomain after read
overlapping_mesh_file = Path(mesh_dir.joinpath(f"SD_Ov_{subdomain_rank}.bp"))
overlapping_submesh, tag_dict = save_and_load_mesh_with_meshtags(
    overlapping_mesh_file,
    comm_self,
    overlapping_submesh,
    [overlapping_material_tags, overlapping_boundary_tags],
    ["material_tag", "boundary_tag"],
)
overlapping_material_tags = tag_dict["material_tag"]
overlapping_boundary_tags = tag_dict["boundary_tag"]

cell_index_map = overlapping_submesh.topology.index_map(dim)
num_cells_ov = cell_index_map.size_local + cell_index_map.num_ghosts

np_ov_cell_map = ov_cell_map.sub_topology_to_topology(
    np.arange(num_cells_ov), inverse=False
)[overlapping_submesh.topology.original_cell_index]

vertex_index_map = overlapping_submesh.topology.index_map(vdim)
num_vetices_local = vertex_index_map.size_local + vertex_index_map.num_ghosts

np_ov_vertex_map = ov_vertex_map.sub_topology_to_topology(
    np.arange(num_vetices_local), inverse=False
)[overlapping_submesh.geometry.input_global_indices]

V_SDov_i = dfx.fem.functionspace(overlapping_submesh, ("Lagrange", degree))
boundary_facets_ov = dfx.mesh.exterior_facet_indices(overlapping_submesh.topology)
boundary_dofs_ov = dfx.fem.locate_dofs_topological(V_SDov_i, fdim, boundary_facets_ov)

ov_l2gmap = generate_dofmap(V_global, V_SDov_i, np_ov_cell_map)
interface_dofs_ov_global = np.setdiff1d(
    ov_l2gmap[boundary_dofs_ov], boundary_dofs_global, assume_unique=True
)
n_ov_dofs = len(ov_l2gmap)
comm.Barrier()
t5 = time()
if args.plot:
    plot_mesh_with_material(
        overlapping_submesh, overlapping_material_tags, plot_dir.joinpath(f"OVmesh_{subdomain_rank}.png")
    )
if subdomain_rank == 0:
    logger.info(f"time for overlapping subdomain generation: {t5-t4}")
comm.Barrier()
t6 = time()

# ### Next is probably simpler in CG see submeshing-communication version
# # Prediction domain generation: interface faces -> interface vertices-> global interface vertices -> vertex extension (-> cell extension)
# ###
if args.plot:
    plot_mesh_with_face_marker(
        overlapping_submesh,
        overlapping_boundary_tags,
        102,
        plot_dir.joinpath(Path(f"Interface_{subdomain_rank}")),
    )
# # fix reordering after read


# derive interface vertices from faces
prediction_domain_size = args.pred_ell
interface_faces = overlapping_boundary_tags.find(102)
f_to_v = overlapping_submesh.topology.connectivity(fdim, vdim)

interface_vertices = np.unique(
    np.array([f_to_v.links(f) for f in interface_faces]).flatten().astype(np.int32)
)
global_interface_vertices = np_ov_vertex_map[interface_vertices]

global_cells_around_interface = get_cells_around_interface_vertices(
    global_mesh, global_interface_vertices
)
if prediction_domain_size > 1:
    prediction_cells = extend_cells_by_layers(
        global_mesh=global_mesh,
        reference_cells=global_cells_around_interface,
        num_layers=prediction_domain_size - 1,
    )
else:
    prediction_cells = global_cells_around_interface

prediction_submesh, pred_cell_map, pred_vertex_map, pred_node_map = (
    dfx.mesh.create_submesh(global_mesh, dim, prediction_cells)
)
prediction_submesh.topology.create_connectivity(fdim, dim)
prediction_material_tags = get_material_marker_for_submesh(
    global_mesh, material_tags, prediction_submesh, pred_cell_map, pred_vertex_map
)
prediction_boundary_tags = get_boundary_marker_for_submesh(
    global_mesh,
    boundary_tags,
    (100, 101),
    prediction_submesh,
    pred_cell_map,
    pred_vertex_map,
)

prediction_mesh_file = Path(mesh_dir.joinpath(f"PSD_{subdomain_rank}.bp"))
prediction_submesh, tag_dict = save_and_load_mesh_with_meshtags(
    prediction_mesh_file,
    comm_self,
    prediction_submesh,
    [prediction_material_tags, prediction_boundary_tags],
    ["material_tag", "boundary_tag"],
)
prediction_material_tags = tag_dict["material_tag"]
prediction_boundary_tags = tag_dict["boundary_tag"]

cell_index_map = prediction_submesh.topology.index_map(dim)
num_cells_pred = cell_index_map.size_local + cell_index_map.num_ghosts

np_pred_cell_map = pred_cell_map.sub_topology_to_topology(
    np.arange(num_cells_pred), inverse=False
)[prediction_submesh.topology.original_cell_index]


V_Pred_i = dfx.fem.functionspace(prediction_submesh, ("Lagrange", degree))
pred_l2gmap = generate_dofmap(V_global, V_Pred_i, np_pred_cell_map)
n_pred_dofs = len(pred_l2gmap)

boundary_facets_prediction = dfx.mesh.exterior_facet_indices(
    prediction_submesh.topology
)
boundary_dofs_prediction = dfx.fem.locate_dofs_topological(
    V_Pred_i, fdim, boundary_facets_prediction
)

comm.Barrier()
t7 = time()
if args.plot:
    plot_mesh_with_material(
        prediction_submesh,
        prediction_material_tags,
        plot_dir.joinpath(f"Prediction_mesh_{subdomain_rank}.png"),
    )
if subdomain_rank == 0:
    logger.info(f"time for prediction domain generation: {t7-t6}")
comm.Barrier()
t8 = time()

# ###
# #  Output, synchronize and summarize meshing phase
# ###
if args.plot:
    u_glob = dfx.fem.Function(V_global)
    u_glob.x.array[ov_l2gmap] = 1.0
    u_glob.x.array[non_ov_l2gmap] = 2.0
    u_glob.x.array[pred_l2gmap] = 3.0
    u_glob.x.array[interface_dofs_ov_global] = 4.0
    plot_sol_pyvista(V_global, u_glob, filename=plot_dir.joinpath(f"Omega_{subdomain_rank}"),show_edges=True)
logger.debug(f"# non-overlapping cells: {len(np_non_ov_cell_map)}")
logger.debug(f"# overlapping cells with ell={ell}: {len(np_ov_cell_map)}")
logger.debug(
    f"# prediction cells with {prediction_domain_size} layers: {len(np_pred_cell_map)}"
)
logger.info(f"Initialization of subdomain {subdomain_rank} done")
comm.Barrier()
t9 = time()
if subdomain_rank == 0:
    logger.info(f"total time taken for (sub)meshing phase {t6-t0}")


# ######
# ###
# ###   Communication phase: establish local-to-local communcation inbetween the
# ###                        various subdomains
# ###
# ######

# # difference to old submeshing: no interface dofs needed, as no averaging

# ###
# #  Phase 1: Number of DoF
# ###
if subdomain_rank == 0:
    t0_comm = time()
    logger.info("== Start evaluating communciation dependencies ==")

# own DoD's = non_ov_l2gmap
unowned_interior_dofs = np.setdiff1d(ov_l2gmap, non_ov_l2gmap, assume_unique=True)
unowned_pred_dofs = np.setdiff1d(pred_l2gmap, non_ov_l2gmap, assume_unique=True)
own_pred_dofs = np.setdiff1d(pred_l2gmap, unowned_pred_dofs, assume_unique=True)
number_unowned_dofs = np.array(
    [len(unowned_interior_dofs), len(unowned_pred_dofs), len(interface_dofs_nonov_global)], dtype=np.int32
)
# # Allgather number of unowned DoF's
number_unowned_dofs_all = np.empty((3 * N_subdomains,), dtype=np.int32)
comm.Allgather(
    [number_unowned_dofs, MPI.INT32_T], [number_unowned_dofs_all, MPI.INT32_T]
)
number_needed_dofs_all = number_unowned_dofs_all.reshape((N_subdomains, 3))
if subdomain_rank == 0:
    logger.info(f"number_needed_dofs_all: {number_needed_dofs_all}")
# ###
# #  Phase 2: gather outgoing and incoming communication
# ###
communication_outgoing = {}
communication_incoming = {}
# for each rank we go through all ranks and check for dependencies
for rank in range(N_subdomains):
    # 1) Broadcast demand of needed DoF's
    if subdomain_rank == rank:
        total_needed_dofs = np.concatenate(
            (
                unowned_interior_dofs,
                unowned_pred_dofs,
                interface_dofs_nonov_global
            ),
            dtype=np.int32,
        )
    else:
        total_needed_dofs = np.empty(
            (np.sum(number_needed_dofs_all, axis=1)[rank],), dtype=np.int32
        )
    comm.Bcast([total_needed_dofs, MPI.INT32_T], root=rank)
    # 2) check which DoF's can be provided from each rank
    if subdomain_rank == rank:
        providable_interior_dofs = np.array([], dtype=np.int32)
        providable_prediction_dofs = np.array([], dtype=np.int32)
        providable_interface_dofs = np.array([], dtype=np.int32)
    else:
        logger.debug(
            f" # rank{rank} requested {number_needed_dofs_all[rank]} dofs from {subdomain_rank}"
        )
        number_of_requested_dofs = number_needed_dofs_all[rank]
        offset = 0
        stride = number_of_requested_dofs[0]
        providable_interior_dofs = np.intersect1d(
            total_needed_dofs[offset : offset + stride], non_ov_l2gmap, assume_unique=True
        )
        offset += stride
        stride = number_of_requested_dofs[1]
        providable_prediction_dofs = np.intersect1d(
            total_needed_dofs[offset : offset + stride],
            non_ov_l2gmap,
            assume_unique=True,
        )
        offset += stride
        stride = number_of_requested_dofs[2]
        providable_interface_dofs = np.intersect1d(
            total_needed_dofs[offset : offset + stride],
            non_ov_l2gmap,
            assume_unique=True,
        )
        if len(providable_interior_dofs) + len(providable_prediction_dofs) + len(providable_interface_dofs) > 0:
            communication_outgoing[rank] = [
                providable_interior_dofs,
                providable_prediction_dofs,
                providable_interface_dofs
            ]
        # these DoF's do still need to get translated to local indexing by np.where(np.isin(self.non_ov_l2gmap, global_dofs))[0] by other subdomain

    # 3) Gatherv the providable DoF's
    sendcounts_interior = np.array(
        comm.gather(len(providable_interior_dofs), root=rank)
    )
    if subdomain_rank == rank:
        all_provided_interior_dofs_root = np.empty(
            (sum(sendcounts_interior),), dtype=np.int32
        )
    else:
        all_provided_interior_dofs_root = None
    comm.Gatherv(
        [providable_interior_dofs.astype(np.int32), MPI.INT32_T],
        [all_provided_interior_dofs_root, sendcounts_interior, MPI.INT32_T],
        root=rank,
    )
    sendcounts_prediction = np.array(
        comm.gather(len(providable_prediction_dofs), root=rank)
    )
    if subdomain_rank == rank:
        all_provided_prediction_dofs_root = np.empty(
            (sum(sendcounts_prediction),), dtype=np.int32
        )
    else:
        all_provided_prediction_dofs_root = None
    comm.Gatherv(
        [providable_prediction_dofs.astype(np.int32), MPI.INT32_T],
        [all_provided_prediction_dofs_root, sendcounts_prediction, MPI.INT32_T],
        root=rank,
    )
    sendcounts_interface = np.array(
        comm.gather(len(providable_interface_dofs), root=rank)
    )

    if subdomain_rank == rank:
        all_provided_interface_dofs_root = np.empty(
            (sum(sendcounts_interface),), dtype=np.int32
        )
    else:
        all_provided_interface_dofs_root = None
    comm.Gatherv(
        [providable_interface_dofs.astype(np.int32), MPI.INT32_T],
        [all_provided_interface_dofs_root, sendcounts_interface, MPI.INT32_T],
        root=rank,
    )
    # 4) split the provided DoF's into interior, prediction and interface DoF's again
    if subdomain_rank == rank:
        # interior dofs
        split_by_count = [
            np.sum(sendcounts_interior[:i]) for i in range(1, len(sendcounts_interior))
        ]
        subarrays_interior = np.split(all_provided_interior_dofs_root, split_by_count)
        # prediction dofs
        split_by_count = [
            np.sum(sendcounts_prediction[:i])
            for i in range(1, len(sendcounts_prediction))
        ]
        subarrays_prediction = np.split(all_provided_prediction_dofs_root, split_by_count)
        # interface dofs
        split_by_count = [
            np.sum(sendcounts_interface[:i])
            for i in range(1, len(sendcounts_interface))
        ]
        subarrays_interface = np.split(all_provided_interface_dofs_root, split_by_count)
        for sender_rank in range(N_subdomains):
            if (
                len(subarrays_interior[sender_rank])
                + len(subarrays_prediction[sender_rank])
                + len(subarrays_interface[sender_rank])
                > 0
            ):
                communication_incoming[sender_rank] = [
                    subarrays_interior[sender_rank],
                    subarrays_prediction[sender_rank],
                    subarrays_interface[sender_rank],
                ]

# # 5) Diagnostics for debugging
for key in communication_outgoing.keys():
    logger.debug(
        f" # rank {subdomain_rank} is going to send {len(communication_outgoing[key][0])} interior and {len(communication_outgoing[key][1])} prediction DoF's to rank {key}"
    )
for key in communication_incoming.keys():
    logger.debug(
        f" # rank {subdomain_rank} is going to receive {len(communication_incoming[key][0])} interior and {len(communication_incoming[key][1])} prediction DoF's from rank {key}"
    )

# ###
# #  Phase 3: Derive a communication schedule based on outgoing communication
# #           This will be done bei rank 0
# ###
N_message_blocks = 3 # (interior, prediction and interface DoF's)
flattened_comms_out = []
for key in communication_outgoing:
    message_count = (
        subdomain_rank,
        key,
        len(communication_outgoing[key][0]),
        len(communication_outgoing[key][1]),
        len(communication_outgoing[key][2]),
    )
    for num in message_count:
        flattened_comms_out.append(num)
flattened_comms_out = np.array(flattened_comms_out, dtype=np.int32)
# Gatherv to rank 0
sendcounts_comms = np.array(comm.gather(len(flattened_comms_out), root=0))
if subdomain_rank == 0:
    comms_all = np.empty((sum(sendcounts_comms),), dtype=np.int32)
else:
    comms_all = None
comm.Gatherv(
    [flattened_comms_out, MPI.INT32_T],
    [comms_all, sendcounts_comms, MPI.INT32_T],
    root=0,
)
# On rank 0 we now derive a min_round_schedule, see Communication.py
if subdomain_rank == 0:
    logger.debug(comms_all.reshape((-1, N_message_blocks + 2)))
    schedule = schedule_from_comms(comms_all.reshape((-1, N_message_blocks + 2)))
    flattened_schedule = flatten_schedule(schedule)
else:
    flattened_schedule = np.array([])
comm.Barrier()

# ###
# #  Phase 4: Broadcast the communication schedule to all other ranks
# ###
# bcast to all
bcasted_schedule = np.empty(comm.bcast(len(flattened_schedule), root=0), dtype=np.int32)
if subdomain_rank == 0:
    bcasted_schedule[:] = flattened_schedule
comm.Bcast([bcasted_schedule, MPI.INT32_T], root=0)
unflattened_schedule = unflatten_schedule(bcasted_schedule)
if subdomain_rank == 0:
    logger.info("== Built and Bcast'ed the following communication schedule")
    for i, round in enumerate(unflattened_schedule):
        logger.info(f"{i} # {[(r[0],r[1]) for r in round]}")
comm.Barrier()
if subdomain_rank == 0:
    t_end_comm = time()
    logging.info(
        f"== Evaluating of communciation dependencies finished after {t_end_comm-t0_comm} =="
    )
######
###
###   Writing phase: write all information about subdomains and scheduling
###                  to h5py files
###
######
if subdomain_rank == 0:
    logging.info("== Storing DoFmaps and communication dependencies to h5")
t_start_IO = time()
# 1) store global mesh
if subdomain_rank == 0:
    logging.info("... Storing fulldomain")
    with h5py.File(mesh_dir.joinpath(f"Omega.hdf5"), "w") as f:
        mesh_grp = f.create_group("Meshes")
        mesh_grp.attrs["mesh"] = str(global_mesh_filename)
        mesh_grp.attrs["h_gmsh"] = h
        mesh_grp.attrs["fem-type"] = 'Lagrange'
        mesh_grp.attrs["degree"] = degree
        mesh_grp.attrs["num_dofs_global"] = n_total_dofs
        dset_boundary_dofs = f.create_dataset(
            "Meshes/mesh/boundary",
            boundary_dofs_global.shape,
            dtype=boundary_dofs_global.dtype,
        )
        dset_boundary_dofs[:] = boundary_dofs_global
# 2) store subdomains
if subdomain_rank == 0:
    logging.info("... Storing subdomains")
# on each rank
with h5py.File(mesh_dir.joinpath(f"Omega_{subdomain_rank}^delta.hdf5"), "w") as f:
    mesh_grp = f.create_group("Meshes")
    mesh_grp.attrs["non-overlapping"] = str(non_overlapping_mesh_file)
    mesh_grp.attrs["overlapping"] = str(overlapping_mesh_file)
    mesh_grp.attrs["prediction"] = str(prediction_mesh_file)
    mesh_grp.attrs["fem-type"] = "Lagrange"
    mesh_grp.attrs["degree"] = degree
    mesh_grp.attrs["ell"] = ell
    mesh_grp.attrs["h_gmsh"] = h
    mesh_grp.attrs["prediction_ell"] = prediction_domain_size
    mesh_grp.attrs["num_dofs_global"] = n_total_dofs
    mesh_grp.attrs["num_dofs_nonov"] = n_nonov_dofs
    mesh_grp.attrs["num_dofs_ov"] = n_ov_dofs
    mesh_grp.attrs["num_dofs_pred"] = n_pred_dofs
    ###
    #  Non-Overlapping mappings
    ###
    dset_entity_map = f.create_dataset(
        "Meshes/non-overlapping/entity_map", np_non_ov_cell_map.shape, dtype=np_non_ov_cell_map.dtype
    )
    dset_entity_map[:] = np_non_ov_cell_map
    dset_l2gmap = f.create_dataset(
        "Meshes/non-overlapping/l2gmap", non_ov_l2gmap.shape, dtype=non_ov_l2gmap.dtype
    )
    dset_l2gmap[:] = non_ov_l2gmap

    dset_boundary_dofs = f.create_dataset(
        "Meshes/non-overlapping/boundary",
        boundary_dofs_nonov.shape,
        dtype=boundary_dofs_nonov.dtype,
    )
    dset_boundary_dofs[:] = boundary_dofs_nonov

    dset_interface_dofs = f.create_dataset(
        "Meshes/non-overlapping/interface",
        interface_dofs_nonov_global.shape,
        dtype=interface_dofs_nonov_global.dtype,
    )
    dset_interface_dofs[:] = interface_dofs_nonov_global
    ###
    #  Overlapping mappings
    ###
    dset_ov_entity_map = f.create_dataset(
        "Meshes/overlapping/entity_map", np_ov_cell_map.shape, dtype=np_ov_cell_map.dtype
    )
    dset_ov_entity_map[:] = np_ov_cell_map
    dset_ov_l2gmap = f.create_dataset(
        "Meshes/overlapping/l2gmap", ov_l2gmap.shape, dtype=ov_l2gmap.dtype
    )
    dset_ov_l2gmap[:] = ov_l2gmap

    dset_ov_boundary_dofs = f.create_dataset(
        "Meshes/overlapping/boundary",
        boundary_dofs_ov.shape,
        dtype=boundary_dofs_ov.dtype,
    )
    dset_ov_boundary_dofs[:] = boundary_dofs_ov

    dset_ov_interface_dofs = f.create_dataset(
        "Meshes/overlapping/interface",
        interface_dofs_ov_global.shape,
        dtype=interface_dofs_ov_global.dtype,
    )
    dset_ov_interface_dofs[:] = interface_dofs_ov_global   
    ###
    #  Prediction mappings
    ###
    dset_prediction_entity_map = f.create_dataset(
        "Meshes/prediction/entity_map",
        np_pred_cell_map.shape,
        dtype=np_pred_cell_map.dtype,
    )
    dset_prediction_entity_map[:] = np_pred_cell_map
    dset_prediction_l2gmap = f.create_dataset(
        "Meshes/prediction/l2gmap", pred_l2gmap.shape, dtype=pred_l2gmap.dtype
    )
    dset_prediction_l2gmap[:] = pred_l2gmap
    dset_prediction_boundary_dofs = f.create_dataset(
        "Meshes/prediction/boundary",
        boundary_dofs_prediction.shape,
        dtype=boundary_dofs_prediction.dtype,
    )
    dset_prediction_boundary_dofs[:] = boundary_dofs_prediction
    ###
    #  Incoming communication
    ###
    communication_incoming_grp = f.create_group("IncomingCommunication")
    for key in communication_incoming:
        communication_incoming_grp.attrs[str(key)] = np.array(
            [len(message) for message in communication_incoming[key]], dtype=np.int32
        )
        inc_interior_dofs, inc_prediction_dofs, inc_interface_dofs = (
            communication_incoming[key]
        )
        # interior dofs
        ds_interior_dofs = f.create_dataset(
            f"IncomingCommunication/from{key}/interior",
            inc_interior_dofs.shape,
            dtype=inc_interior_dofs.dtype,
        )
        ds_interior_dofs[:] = inc_interior_dofs
        # prediction dofs
        ds_prediction_dofs = f.create_dataset(
            f"IncomingCommunication/from{key}/prediction",
            inc_prediction_dofs.shape,
            dtype=inc_prediction_dofs.dtype,
        )
        ds_prediction_dofs[:] = inc_prediction_dofs
        # interface dofs
        ds_interface_dofs = f.create_dataset(
            f"IncomingCommunication/from{key}/interface",
            inc_interface_dofs.shape,
            dtype=inc_interface_dofs.dtype,
        )
        ds_interface_dofs[:] = inc_interface_dofs
    ###
    #  Outgoing communication
    ###
    communication_outgoing_grp = f.create_group("OutgoingCommunication")
    # flattened schedule
    ds_schedule = f.create_dataset(
        f"OutgoingCommunication/schedule",
        bcasted_schedule.shape,
        bcasted_schedule.dtype,
    )
    ds_schedule[:] = bcasted_schedule
    for key in communication_outgoing:
        communication_outgoing_grp.attrs[str(key)] = np.array(
            [len(message) for message in communication_outgoing[key]], dtype=np.int32
        )
        inc_interior_dofs, inc_prediction_dofs, inc_interface_dofs = (
            communication_outgoing[key]
        )
        # interior dofs
        ds_interior_dofs = f.create_dataset(
            f"OutgoingCommunication/to{key}/interior",
            inc_interior_dofs.shape,
            dtype=inc_interior_dofs.dtype,
        )
        ds_interior_dofs[:] = inc_interior_dofs
        # prediction dofs
        ds_prediction_dofs = f.create_dataset(
            f"OutgoingCommunication/to{key}/prediction",
            inc_prediction_dofs.shape,
            dtype=inc_prediction_dofs.dtype,
        )
        ds_prediction_dofs[:] = inc_prediction_dofs
        # interface dofs
        ds_interface_dofs = f.create_dataset(
            f"OutgoingCommunication/to{key}/interface",
            inc_interface_dofs.shape,
            dtype=inc_interface_dofs.dtype,
        )
        ds_interface_dofs[:] = inc_interface_dofs
comm.Barrier()

###
#  Finish the script
###
if subdomain_rank == 0:
    t_end_IO = time()
    logging.info(
        f"== Storing dofmaps and communication dependencies finished after {t_end_IO-t_start_IO}"
    )
    logging.info("Finished submeshing")
    logging.info(f"Total time of submeshing script {t_end_IO-t_before_script} seconds")

logging.shutdown()
