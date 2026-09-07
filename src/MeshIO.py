from mpi4py import MPI
import dolfinx as dfx
import ufl
import numpy as np
from Config import FEM_TYPE, FEM_DEGREE
import warnings
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from dolfinx import cpp
import basix

FLOAT_CONVERSION = 1e8

try:
    import h5py
except ModuleNotFoundError as e:
    print(e)
try:
    import adios4dolfinx
except ModuleNotFoundError as e:
    print(e)
# import gmsh
# import meshio


def _patch_adios4dolfinx_dof_layout_api(mesh: dfx.mesh.Mesh) -> None:
    """Restore the dolfinx 0.10 dof-layout method used by adios4dolfinx."""
    cmap = (
        mesh.geometry.cmaps[0]
        if hasattr(mesh.geometry, "cmaps")
        else mesh.geometry.cmap
    )
    layout = cmap.create_dof_layout()
    layout_type = type(layout)
    if not hasattr(layout_type, "num_entity_closure_dofs"):
        layout_type.num_entity_closure_dofs = lambda self, dim: len(
            self.entity_closure_dofs(dim, 0)
        )


def _patch_adios4dolfinx_partitioner_api() -> None:
    """Allow adios4dolfinx to call the dolfinx 0.10 partitioner signature."""
    create_cell_partitioner = dfx.cpp.mesh.create_cell_partitioner
    if getattr(create_cell_partitioner, "_domainsplitting_compat", False):
        return

    def create_cell_partitioner_compat(*args):
        if len(args) == 1:
            try:
                return create_cell_partitioner(args[0], None)
            except TypeError:
                return create_cell_partitioner(args[0])
        if len(args) == 2:
            try:
                return create_cell_partitioner(args[0], args[1])
            except TypeError:
                return create_cell_partitioner(args[0])
        return create_cell_partitioner(*args)

    create_cell_partitioner_compat._domainsplitting_compat = True
    dfx.cpp.mesh.create_cell_partitioner = create_cell_partitioner_compat


def _patch_adios4dolfinx_api(mesh: dfx.mesh.Mesh) -> None:
    _patch_adios4dolfinx_dof_layout_api(mesh)
    _patch_adios4dolfinx_partitioner_api()


def _adios_path(filename: str | Path) -> Path:
    path = Path(filename)
    if path.suffix != ".bp":
        path = path.with_suffix(".bp")
    return path


def _remove_existing_adios_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _adios_meshtags_sidecar(path: Path) -> Path:
    if path.is_dir():
        return path / "_domainsplitting_meshtags.npz"
    return path.with_suffix(path.suffix + ".meshtags.npz")


def _write_meshtags_sidecar(
    path: Path,
    meshtags: Iterable[dfx.mesh.MeshTags],
    meshtag_names: Iterable[str],
) -> None:
    data = {}
    for meshtag, name in zip(meshtags, meshtag_names):
        data[f"{name}__dim"] = np.array([meshtag.dim], dtype=np.int32)
        data[f"{name}__indices"] = np.asarray(meshtag.indices, dtype=np.int32)
        data[f"{name}__values"] = np.asarray(meshtag.values)
    np.savez(_adios_meshtags_sidecar(path), **data)


def _read_meshtags_sidecar(
    path: Path,
    mesh: dfx.mesh.Mesh,
    mesh_tag_names: Iterable[str],
) -> Optional[Dict[str, dfx.mesh.MeshTags]]:
    sidecar = _adios_meshtags_sidecar(path)
    if not sidecar.exists():
        return None

    data = np.load(sidecar)
    read_meshtags = {}
    for name in mesh_tag_names:
        dim = int(data[f"{name}__dim"][0])
        indices = np.asarray(data[f"{name}__indices"], dtype=np.int32)
        values = np.asarray(data[f"{name}__values"])
        read_meshtags[name] = dfx.mesh.meshtags(mesh, dim, indices, values)
    return read_meshtags


def _fem_type_code(V: dfx.fem.function.FunctionSpace) -> int:
    return 1 if V.element.basix_element.discontinuous else 0


def _fem_type_from_code(code: int) -> str:
    if code == 0:
        return "Lagrange"
    if code == 1:
        return "DG"
    raise ValueError(f"Unknown finite element type code {code}")


def _function_space_metadata(V: dfx.fem.function.FunctionSpace) -> dict[str, np.ndarray]:
    return {
        "fem_type_code": np.array([_fem_type_code(V)], dtype=np.int32),
        "fem_degree": np.array([V.element.basix_element.degree], dtype=np.int32),
    }


def write_mesh_with_meshtags(
    filename: Path,
    mesh: dfx.mesh.Mesh,
    meshtags: Iterable[dfx.mesh.MeshTags],
    meshtag_names: Iterable[str] = ["material_tag"],
):
    path = _adios_path(filename)
    _patch_adios4dolfinx_api(mesh)
    adios4dolfinx.write_mesh(path, mesh)
    meshtags = list(meshtags)
    meshtag_names = list(meshtag_names)
    try:
        for meshtag, meshtag_name in zip(meshtags, meshtag_names):
            adios4dolfinx.write_meshtags(
                path, mesh, meshtag, meshtag_name=meshtag_name
            )
    except (TypeError, ValueError):
        _write_meshtags_sidecar(path, meshtags, meshtag_names)


def read_mesh_with_meshtags(
    filename: Path,
    comm: MPI.Intracomm,
    mesh_tag_names: Iterable[str] = ["material_tag"],
    ghost_mode: Optional[dfx.mesh.GhostMode] = None,
) -> Tuple[dfx.mesh.Mesh, Dict[str, dfx.mesh.MeshTags]]:
    path = _adios_path(filename)
    _patch_adios4dolfinx_partitioner_api()
    if ghost_mode is not None:
        read_mesh = adios4dolfinx.read_mesh(path, comm, ghost_mode=ghost_mode)
    else:
        read_mesh = adios4dolfinx.read_mesh(path, comm)
    read_meshtags = {}
    sidecar_meshtags = _read_meshtags_sidecar(path, read_mesh, mesh_tag_names)
    if sidecar_meshtags is not None:
        return read_mesh, sidecar_meshtags

    for name in mesh_tag_names:
        read_meshtags[name] = adios4dolfinx.read_meshtags(
            path, read_mesh, meshtag_name=name
        )
    return read_mesh, read_meshtags


def write_adios_function(
    filename: str | Path,
    V: dfx.fem.function.FunctionSpace,
    func: dfx.fem.function.Function,
    t: float = 0.0,
    name: str = "f",
) -> Path:
    path = _adios_path(filename)
    _remove_existing_adios_path(path)
    _patch_adios4dolfinx_api(V.mesh)
    adios4dolfinx.write_mesh(path, V.mesh)
    adios4dolfinx.write_attributes(
        path,
        V.mesh.comm,
        f"FunctionSpace/{name}",
        _function_space_metadata(V),
    )
    adios4dolfinx.write_function(path, func, time=t, name=name)
    return path


def append_adios_function(
    filename: str | Path,
    func: dfx.fem.function.Function,
    t: float = 0.0,
    name: str = "f",
) -> Path:
    path = _adios_path(filename)
    _patch_adios4dolfinx_partitioner_api()
    adios4dolfinx.write_function(path, func, time=t, name=name)
    return path


def read_adios_function(
    filename: str | Path,
    t: float = 0.0,
    name: str = "f",
    reconstruct_FunctionSpace: bool = False,
    comm: MPI.Intracomm = MPI.COMM_SELF,
    get_timesteps: bool = False,
):
    path = _adios_path(filename)
    _patch_adios4dolfinx_partitioner_api()
    if get_timesteps:
        return adios4dolfinx.read_timestamps(path, comm, name)

    mesh = adios4dolfinx.read_mesh(path, comm)
    attrs = adios4dolfinx.read_attributes(path, comm, f"FunctionSpace/{name}")
    fem_type = _fem_type_from_code(int(attrs["fem_type_code"][0]))
    fem_degree = int(attrs["fem_degree"][0])
    V = dfx.fem.functionspace(mesh, (fem_type, fem_degree))
    func = dfx.fem.Function(V)
    adios4dolfinx.read_function(path, func, time=t, name=name)
    if reconstruct_FunctionSpace:
        return V, func.x.array.copy()
    return func.x.array.copy()


def custom_1Dmesh_from_np(np_mesh):
    # see https://fenicsproject.discourse.group/t/mesheditor-equivalent/7917
    N = np_mesh.size
    gdim, shape, degree = 1, "interval", 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
    x = np_mesh.reshape((N, 1))
    cells = np.array([[i - 1, i] for i in range(1, N)], dtype=np.int64)

    mesh = dfx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

    V = dfx.fem.functionspace(mesh, ("P", 1))
    print(V.tabulate_dof_coordinates(), mesh.geometry.dim)
    print(mesh.topology.index_map(mesh.topology.dim).size_local)


def custom_2Dmesh_from_np(np_mesh, cells):
    gdim, shape, degree = 2, "triangle", 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

    mesh = dfx.mesh.create_mesh(MPI.COMM_WORLD, cells, np_mesh[:, :gdim], domain)

    V = dfx.fem.functionspace(mesh, ("P", 1))
    print(V.tabulate_dof_coordinates(), mesh.geometry.dim)


def save_vtk(
    V: dfx.fem.function.FunctionSpace,
    uh: dfx.fem.function.Function,
    name="output",
    t: float = 0.0,
):
    if os.path.exists(f"{name}.bp"):
        raise NotImplementedError
    # be aware, that writing multiple times in the same file is not supported
    # we have to write some kind of context handling for that
    with dfx.io.VTXWriter(V.mesh.comm, f"{name}.bp", uh, "BP4") as vtx:
        vtx.write(t)


def save_xdmf(
    V: dfx.fem.function.FunctionSpace, uh: dfx.fem.function.Function, name="output"
):
    with dfx.io.XDMFFile(V.mesh.comm, f"{name}.xdmf", "w") as xdmf:
        if FEM_TYPE == "Lagrange" and FEM_DEGREE == 1:
            xdmf.write_mesh(V.mesh)
            xdmf.write_function(uh)
        else:
            warnings.warn(
                UserWarning(
                    "Solution got interpolated when trying to save, try without saving instead"
                )
            )
            V1 = dfx.fem.functionspace(V.mesh, (FEM_TYPE, 1))
            u1 = dfx.fem.Function(V1)
            u1.interpolate(uh)
            xdmf.write_mesh(V1.mesh)
            xdmf.write_function(u1)


def read_h5(filename):
    file = h5py.File(filename, "r")
    # List all the groups in the file
    # print("Groups in the h5 file:")
    # print(list(file.keys()))
    group = file["Function"]["f"]["0"]
    values = np.array(group)
    geometry = np.array(file["Mesh"]["/Mesh"]["mesh"]["geometry"])
    topology = np.array(file["Mesh"]["/Mesh"]["mesh"]["topology"])
    return values, geometry, topology


def write_hdf5(filename, V, func, t=0.0):
    with h5py.File(f"{filename}.hdf5", "w") as f:
        t_set = f.create_dataset("timesteps", (1,), maxshape=(None,), dtype=np.float64)
        t_set[:] = np.array([t])
        ### mesh group
        mesh_grp = f.create_group("Mesh")
        geometry_x = V.mesh.geometry.x
        cells = V.mesh.geometry.dofmap
        dset_geometry = f.create_dataset(
            "Mesh/geometry/x", geometry_x.shape, dtype=geometry_x.dtype
        )
        dset_geometry[:] = geometry_x
        dset_cells = f.create_dataset(
            "Mesh/geometry/dofmap", cells.shape, dtype=cells.dtype
        )
        dset_cells[:] = cells
        mesh_grp.attrs[
            "mesh_degree"
        ] = V.mesh._ufl_domain._ufl_coordinate_element._degree
        ### FunctionSpace group
        fs_grp = f.create_group("FunctionSpace")

        ### FunctionSpace metadata
        family_name = V.element.basix_element.family.name
        if V.element.basix_element.discontinuous and family_name == "P":
            fem_type = "DG"
        elif family_name == "P":
            fem_type = "CG"
        else:
            raise NotImplementedError
        fs_grp.attrs["family_name"] = family_name
        fs_grp.attrs["fem_type"] = fem_type
        fs_grp.attrs["fem_degree"] = V.element.basix_element.degree
        fs_grp.attrs["cell_name"] = V.element.basix_element.cell_type.name
        fs_grp.attrs["cell_gdim"] = V.ufl_domain()._geometric_dimension
        ### function dataset
        t_conv = int(t * FLOAT_CONVERSION)
        values = func.x.array
        dset_func = f.create_dataset(
            f"Function/f/{str(t_conv)}", (values.size,), dtype=values.dtype
        )
        dset_func[:] = values


def append_hdf5(filename, func, t=0.0):
    with h5py.File(f"{filename}.hdf5", "a") as f:
        # append new timestep
        t_len = f["timesteps"].len()
        f["timesteps"].resize((t_len + 1,))
        f["timesteps"][-1] = t

        ### function dataset
        t_conv = int(t * FLOAT_CONVERSION)
        values = func.x.array
        dset_func = f.create_dataset(
            f"Function/f/{str(t_conv)}", (values.size,), dtype=values.dtype
        )
        dset_func[:] = values


def read_hdf5(
    filename,
    t=0.0,
    reconstruct_FunctionSpace=False,
    comm=MPI.COMM_SELF,
    get_timesteps=False,
):
    with h5py.File(f"{filename}.hdf5", "r") as f:
        if get_timesteps:
            return np.array(f["timesteps"])
        t_conv = int(t * FLOAT_CONVERSION)
        func_values = np.array(f[f"Function/f/{str(t_conv)}"])
        if reconstruct_FunctionSpace:
            x = np.array(f["Mesh/geometry/x"])
            cells = np.array(f["Mesh/geometry/dofmap"])
            mesh_degree = int(f["Mesh"].attrs["mesh_degree"])

            gdim = int(f["FunctionSpace"].attrs["cell_gdim"])
            shape = f["FunctionSpace"].attrs["cell_name"]
            fem_degree = int(f["FunctionSpace"].attrs["fem_degree"])
            fem_type = f["FunctionSpace"].attrs["fem_type"]
            family_name = f["FunctionSpace"].attrs["family_name"]

            recon_mesh = reconstruct_mesh(
                cell_shape=shape,
                cell_degree=mesh_degree,
                family_name=family_name,
                cells=cells,
                x=x,
                gdim=gdim,
                communicator=comm,
            )
            Fs = dfx.fem.functionspace(recon_mesh, (fem_type, fem_degree))
            return Fs, func_values
    return func_values


def reconstruct_mesh(
    cell_shape, cell_degree, family_name, cells, x, gdim, communicator
):
    if cell_shape == "triangle":
        cell_type = dfx.mesh.CellType.triangle
    elif cell_shape == "quadrilateral":
        cell_type = dfx.mesh.CellType.quadrilateral
    elif cell_shape == "interval":
        cell_type = dfx.mesh.CellType.interval
    else:
        raise NotImplementedError

    cmap = cpp.fem.CoordinateElement_float64(cell_type, cell_degree)
    msh = cpp.mesh.create_mesh(
        communicator,
        cpp.graph.AdjacencyList_int64(cells.astype(np.int64)),
        cmap,
        x,
        cpp.mesh.create_cell_partitioner(dfx.mesh.GhostMode.shared_facet),
    )
    # msh.name = name

    domain = ufl.Mesh(
        basix.ufl.element(
            family_name,
            cell_shape,
            cell_degree,
            basix.LagrangeVariant.gll_warped,
            shape=(x.shape[1],),  # (x.shape[1],),
            gdim=x.shape[1], #gdim
            # discontinuous=discontinuous,
        )
    )

    recon_mesh = dfx.mesh.Mesh(msh, domain)
    return recon_mesh


if __name__ == "__main__":
    # a = np.linspace(0, 1, 101)
    # a[-1] = 1.05
    # custom_1Dmesh_from_np(a)

    mesh = dfx.mesh.create_unit_square(
        MPI.COMM_WORLD, 10, 10, dfx.mesh.CellType.triangle
    )
    V = dfx.fem.functionspace(mesh, ("DG", 1))
    func = dfx.fem.Function(V)
    func.x.array[::3] = 4.0

    write_hdf5("test", V, func, 1.0)

    FS, f = read_hdf5("test", t=1.0, reconstruct_FunctionSpace=True)
    print(V)
    print(FS)
    test = dfx.fem.Function(V)
    test.x.array[:] = f

    from Norms import error_norm_ref

    print(error_norm_ref(func, test, "H1"))
