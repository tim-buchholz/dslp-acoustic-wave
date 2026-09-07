from mpi4py import MPI
import dolfinx as dfx
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Iterable, List, Dict, Any
import gmsh
from pathlib import Path
import adios4dolfinx
import logging
import scifem

logger = logging.getLogger(__name__)

###
#  Global Meshing
###


def build_global_mesh(
    comm: MPI.Intracomm,
    width: float,
    height: float,
    h: float,
    partitioner: Optional[
        Callable[
            [MPI.Intracomm, int, int, dfx.cpp.graph.AdjacencyList_int32],
            dfx.cpp.graph.AdjacencyList_int32,
        ]
    ] = None,
    mesh_refinement_factor: float = 2,
    refinement_mode: str = 'center',
    save_mesh: bool = False,
) -> Tuple[dfx.mesh.Mesh, dfx.mesh.MeshTags, dfx.mesh.MeshTags, np.ndarray]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.model.add("rectangle_with_prism")

    # --- Outer rectangle (one set of boundary lines + one surface) ---
    a1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h)
    a2 = gmsh.model.geo.addPoint(width, 0.0, 0.0, h)
    a3 = gmsh.model.geo.addPoint(width, height, 0.0, h)
    a4 = gmsh.model.geo.addPoint(0.0, height, 0.0, h)
    if mesh_refinement_factor == 1:
        l1 = gmsh.model.geo.addLine(a1, a2)
        l2 = gmsh.model.geo.addLine(a2, a3)
        l3 = gmsh.model.geo.addLine(a3, a4)
        l4 = gmsh.model.geo.addLine(a4, a1)
        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], tag=1) 
        gmsh.model.addPhysicalGroup(
            1, [l1, l2, l3, l4], tag=101
        ) 
    elif refinement_mode == 'axes':
        a12 = gmsh.model.geo.addPoint(0.5 * width, 0.0, 0.0, h / mesh_refinement_factor)
        a23 = gmsh.model.geo.addPoint(width, 0.5 * height, 0.0, h / mesh_refinement_factor)
        a34 = gmsh.model.geo.addPoint(0.5 * width, height, 0.0, h / mesh_refinement_factor)
        a41 = gmsh.model.geo.addPoint(0.0, 0.5 * height, 0.0, h / mesh_refinement_factor)

        l1a = gmsh.model.geo.addLine(a1, a12)
        l1b = gmsh.model.geo.addLine(a12, a2)
        l2a = gmsh.model.geo.addLine(a2, a23)
        l2b = gmsh.model.geo.addLine(a23, a3)
        l3a = gmsh.model.geo.addLine(a3, a34)
        l3b = gmsh.model.geo.addLine(a34, a4)
        l4a = gmsh.model.geo.addLine(a4, a41)
        l4b = gmsh.model.geo.addLine(a41, a1)
        cl_outer = gmsh.model.geo.addCurveLoop([l1a, l1b, l2a, l2b, l3a, l3b, l4a, l4b])
        outer_surface = gmsh.model.geo.addPlaneSurface([cl_outer])

        # --- interior refinement points and lines (create each geometric line once) ---
        m1 = gmsh.model.geo.addPoint(
            0.5 * width, 0.5 * height, 0.0, h / mesh_refinement_factor
        )

        # radial lines from center to mid-edge points (each only once)
        l5 = gmsh.model.geo.addLine(m1, a12)
        l6 = gmsh.model.geo.addLine(m1, a23)
        l7 = gmsh.model.geo.addLine(m1, a34)
        l8 = gmsh.model.geo.addLine(m1, a41)

        # --- synchronize geometry before embedding ---
        gmsh.model.geo.synchronize()

        # embed all internal curve ids (dimension 1) into the outer surface (dimension 2)
        internal_curves = [l5, l6, l7, l8]
        gmsh.model.mesh.embed(1, internal_curves, 2, outer_surface)

        # --- physical groups (tags) ---
        gmsh.model.addPhysicalGroup(2, [outer_surface], tag=1)  # whole domain material
        gmsh.model.addPhysicalGroup(
            1, [l1a, l1b, l2a, l2b, l3a, l3b, l4a, l4b], tag=101
        )  # outer boundary
    elif refinement_mode == 'center': 
        # --- Add central point for refinement ---
        cx, cy = width / 2.0, height / 2.0
        p_center = gmsh.model.geo.addPoint(cx, cy, 0.0, h)  # geometric point, h irrelevant here

        # Outer rectangle geometry (same as in the if-case)
        l1 = gmsh.model.geo.addLine(a1, a2)
        l2 = gmsh.model.geo.addLine(a2, a3)
        l3 = gmsh.model.geo.addLine(a3, a4)
        l4 = gmsh.model.geo.addLine(a4, a1)
        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        # --- Local mesh refinement via fields ---
        # 1) Attractor: distance to central point
        attractor = gmsh.model.mesh.field.add("Attractor")
        gmsh.model.mesh.field.setNumbers(attractor, "NodesList", [p_center])

        # 2) Threshold: small size near the point, regular size elsewhere
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", attractor)
        gmsh.model.mesh.field.setNumber(threshold, "LcMin", h / mesh_refinement_factor)
        gmsh.model.mesh.field.setNumber(threshold, "LcMax", h)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", min(width, height) / 10.0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", min(width, height) / 3.0)

        # 3) Make this field control the mesh sizes
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

        # Physical groups
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], tag=101)

    # --- mesh generation (use gmsh.model.mesh.generate) ---
    gmsh.model.mesh.generate(2)

    # --- convert to dolfinx mesh ---
    mesh_data = dfx.io.gmsh.model_to_mesh(
        gmsh.model, comm, 0, gdim=2, partitioner=partitioner
    )
    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    face_tags = mesh_data.facet_tags

    if save_mesh:
        mesh_folder = Path("meshes")
        mesh_folder.mkdir(exist_ok=True, parents=True)
        filename = mesh_folder / f"mesh_h={h}.msh"
        gmsh.write(str(filename))

    gmsh.finalize()

    return mesh, cell_tags, face_tags


def write_global_mesh(comm: MPI.Intracomm, args: Any, mesh_dir: Path) -> Tuple[np.ndarray, Path, int]:  # args: Namespace object
    subdomain_rank = comm.Get_rank()
    if args.partitioner == "scotch":
        partitioner = dfx.mesh.create_cell_partitioner(dfx.graph.partitioner_scotch())
    elif args.partitioner == "kahip":
        partitioner = dfx.mesh.create_cell_partitioner(dfx.graph.partitioner_kahip())
    elif args.partitioner == "parmetis":
        partitioner = dfx.mesh.create_cell_partitioner(dfx.graph.partitioner_parmetis())
    global_mesh, material_tags, boundary_tags = build_global_mesh(
        comm=comm,
        width=args.width,
        height=args.height,
        h=args.h,
        partitioner=partitioner,
        mesh_refinement_factor=args.mesh_refinement,
        refinement_mode=args.refinement_mode
    )
    V = dfx.fem.functionspace(global_mesh, ("Lagrange", args.polynomial_degree))
    n_total_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if subdomain_rank == 0:
        logger.info(f"Total number of DoFs per component {n_total_dofs}")

    dim = global_mesh.topology.dim
    fdim = dim - 1
    # subdomain tags
    global_mesh.topology.create_entities(dim)
    global_mesh.topology.create_entities(fdim)
    e_map = global_mesh.topology.index_map(dim)
    entities_local = np.arange(e_map.size_local, dtype=np.int32)
    global_mesh.topology.create_connectivity(dim, global_mesh.topology.dim)
    values = np.full_like(np.arange(e_map.size_local, dtype=np.int32), subdomain_rank)
    subdomain_tags = dfx.mesh.meshtags(global_mesh, dim, entities_local, values)
    comm.Barrier()
    filename = Path(mesh_dir.joinpath("Omega.bp"))
    write_mesh_with_meshtags(
        filename,
        global_mesh,
        [material_tags, subdomain_tags, boundary_tags],
        ["material_tag", "subdomain_tag", "boundary_tag"],
    )

    return filename, n_total_dofs

###
#   Meshtag Transfer
###


def get_material_marker_for_submesh(
    global_mesh: dfx.mesh.Mesh,
    global_material_tag: dfx.mesh.MeshTags,
    submesh: dfx.mesh.Mesh,
    entity_map: dfx.mesh.EntityMap,
    vertex_map: dfx.mesh.EntityMap,
) -> dfx.mesh.MeshTags:
    transfered_global_tag, cell_map = scifem.transfer_meshtags_to_submesh(
        global_material_tag, submesh, vertex_map, entity_map
    )
    return transfered_global_tag

def get_boundary_marker_for_submesh(
    global_mesh: dfx.mesh.Mesh,
    global_boundary_tag: dfx.mesh.MeshTags,
    boundary_tag_values: Iterable[int],
    submesh: dfx.mesh.Mesh,
    entity_map: dfx.mesh.EntityMap,
    vertex_map: dfx.mesh.EntityMap,
):
    transfered_global_tag, face_map = scifem.transfer_meshtags_to_submesh(global_boundary_tag, submesh, vertex_map, entity_map)
    fdim = global_mesh.topology.dim -1
    num_faces_submesh = submesh.topology.index_map(fdim).size_local
    new_boundary_marker = np.zeros(num_faces_submesh, dtype=np.int32)
    new_tag_value = max(boundary_tag_values) + 1

    all_boundary_faces = dfx.mesh.exterior_facet_indices(submesh.topology)
    new_boundary_marker[all_boundary_faces] = new_tag_value
    for value in boundary_tag_values:
        boundary_i = transfered_global_tag.find(value)
        new_boundary_marker[boundary_i] = value
    submesh_boundary_tags = dfx.mesh.meshtags(submesh, fdim, np.arange(num_faces_submesh, dtype=np.int32), new_boundary_marker)
    return submesh_boundary_tags


###
#  DoF mappings between subspaces
###
def generate_dofmap(
    V_global: dfx.fem.function.FunctionSpace,
    V_subdomain: dfx.fem.function.FunctionSpace,
    np_cell_map: np.ndarray
) -> np.ndarray:
    local_func = dfx.fem.Function(V_subdomain)
    local_to_global_dofmap = np.zeros_like(local_func.x.array[:]).astype(np.int32)
    for local_cell_index, global_cell_index in enumerate(np_cell_map):
        local_dofs = V_subdomain.dofmap.list[local_cell_index]
        global_dofs = V_global.dofmap.list[global_cell_index]
        local_to_global_dofmap[local_dofs] = global_dofs

    return local_to_global_dofmap.astype(np.int32)


###
#  Cell extensions
###
def extend_cells_by_layers(
    global_mesh: dfx.mesh.Mesh,
    reference_cells: np.ndarray,
    num_layers: int,
):
    dim = global_mesh.topology.dim
    vdim = 0
    global_mesh.topology.create_entities(vdim)
    vertex_map = global_mesh.topology.index_map(vdim)
    num_vertices = vertex_map.size_local + vertex_map.num_ghosts
    vertices = np.arange(num_vertices, dtype=np.int32)
    vertex_marker_values = np.zeros_like(vertices, dtype=np.int32)

    ## connectivity
    global_mesh.topology.create_connectivity(dim, vdim)
    c_to_v = global_mesh.topology.connectivity(dim, vdim)

    # mark values
    for global_cell in reference_cells:
        vertex_marker_values[c_to_v.links(global_cell)] = 1
    vt = dfx.mesh.meshtags(global_mesh, vdim, vertices, vertex_marker_values)
    c_map = global_mesh.topology.index_map(dim)
    num_cells_global = c_map.size_local + c_map.num_ghosts

    extension = reference_cells.copy()
    other_cells = np.arange(num_cells_global).astype(np.int32)

    # for each layer
    for layer in range(num_layers):
        other_cells = np.setdiff1d(other_cells, extension)
        additional_cells = []
        for cell in other_cells:
            for vertex in c_to_v.links(cell):
                if vt.values[vertex] == 1:
                    additional_cells.append(cell)
                    # break
        for cell in additional_cells:
            vertex_marker_values[c_to_v.links(cell)] = 1
        vt = dfx.mesh.meshtags(global_mesh, vdim, vertices, vertex_marker_values)
        extension = np.concatenate((extension, np.array(additional_cells))).astype(
            np.int32
        )

    return np.unique(extension)

def get_cells_around_interface_vertices(domain: dfx.mesh.Mesh, interface_vertices: np.ndarray):  
    tdim = domain.topology.dim
    vdim = 0
    v_to_c = domain.topology.connectivity(vdim, tdim)
    additional_cells = []
    for vertex in interface_vertices:
        for cell in v_to_c.links(vertex):
            additional_cells.append(cell)
    cells_around_interface = np.unique(np.array(additional_cells).astype(np.int32))
    return cells_around_interface


def get_h(mesh: dfx.mesh.Mesh, mode: str = 'max') -> float:
    tdim = mesh.topology.dim
    c_map = mesh.topology.index_map(tdim)
    num_cells_local = c_map.size_local + c_map.num_ghosts
    cells = np.arange(num_cells_local, dtype=np.int32)
    hs = mesh.h(tdim, cells)
    if mode == 'max':
        return np.max(hs)
    elif mode == 'min':
        return np.min(hs) 
    else: 
        return None


def write_mesh_with_meshtags(
    filename: Path,
    mesh: dfx.mesh.Mesh,
    meshtags: Iterable[dfx.mesh.MeshTags],
    meshtag_names: Iterable[str] = ["material_tag"],
):  
    adios4dolfinx.write_mesh(filename, mesh)
    for i, meshtag in enumerate(meshtags):
        adios4dolfinx.write_meshtags(
            filename, mesh, meshtag, meshtag_name=meshtag_names[i]
        )


def read_mesh_with_meshtags(
    filename: Path,
    comm: MPI.Intracomm,
    mesh_tag_names: Iterable[str] = ["material_tag"],
    ghost_mode: Optional[dfx.mesh.GhostMode] = None
) -> Tuple[dfx.mesh.Mesh, Dict[str, dfx.mesh.MeshTags]]:
    if ghost_mode is not None:
        read_mesh = adios4dolfinx.read_mesh(filename, comm, ghost_mode=ghost_mode)
    else:
        read_mesh = adios4dolfinx.read_mesh(filename, comm)
    read_meshtags = {}
    for i, name in enumerate(mesh_tag_names):
        read_meshtags[name] = adios4dolfinx.read_meshtags(filename, read_mesh, meshtag_name=name)
    return read_mesh, read_meshtags


def save_and_load_mesh_with_meshtags(
    filename: Path,
    comm: MPI.Intracomm,
    mesh: dfx.mesh.Mesh,
    meshtags: Iterable[dfx.mesh.MeshTags],
    meshtag_names: Iterable[str] = ["material_tag"],
    
) -> Tuple[dfx.mesh.Mesh, Dict[str, dfx.mesh.MeshTags]]:
    write_mesh_with_meshtags(filename, mesh, meshtags, meshtag_names)
    return read_mesh_with_meshtags(filename, comm, meshtag_names)
