from mpi4py import MPI
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import dolfinx as dfx
import pyvista
from petsc4py import PETSc
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

pyvista.OFF_SCREEN = True

# plt.style.use(["science", "high-vis", "muted"])


def plot_sol_pyvista(
    V: dfx.fem.function.FunctionSpace,
    sol: dfx.fem.function.Function = None,
    filename: Path = Path('figure.png'),
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_cbar: bool = True,
    warp_by_scalar: bool = False,
    warping_factor: float = 0.5,
    show_edges: bool = False,
    jupyter_backend: Optional[str] = None,  # "static"
):
    """
    Simple plotting with pyvista and dolfinx
    Args:
        V (dfx.fem.function.FunctionSpace): dfx.fem.FunctionSpace
        sol (dfx.fem.function.Function): solution function
        cmap (Optional[str], optional): colormap. Defaults to None.
        clim (Optional[Tuple[float,float]], optional): (vmin,vmax). Defaults to None.
        name (str, optional): name of solution. Defaults to "u".
        show_cbar (bool, optional):  Defaults to True.
        warp_by_scalar (bool, optional):  Defaults to False.
        warping_factor (float, optional):  Defaults to 0.5.
        show_edges (bool, optional):  Defaults to False.
        jupyter_backend (Optional[str], optional): Valid: 'trame','static','client','server','none'; Default: 'static'.
    """
    logger.debug(f"Started plotting {filename}")
    name = 'u'
    cells, types, x = dfx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    if sol == None:
        grid.point_data[name] = 1.0
        clim = (0.0, 1.0)
    else:
        grid.point_data[name] = sol.x.array.real
    grid.set_active_scalars(name)

    plotter = pyvista.Plotter()
    plotter.add_mesh(
        grid,
        scalars=name,
        show_edges=show_edges,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_cbar and not warp_by_scalar,
    )
    if warp_by_scalar == True:
        warped = grid.warp_by_scalar(factor=warping_factor)
        plotter.add_mesh(
            warped, cmap=cmap, clim=clim, show_edges=show_edges, show_scalar_bar=False
        )
        if show_cbar:
            plotter.add_scalar_bar(name)
    else:
        plotter.view_xy()

    if pyvista.OFF_SCREEN:
        plotter.screenshot(f"{filename}", window_size=[1800, 1800])
    else:
        if jupyter_backend is not None:
            plotter.show(jupyter_backend=jupyter_backend)
        else:
            plotter.show()
    logger.debug(f"Finished plotting {filename}")


def plot_sol_with_object(
    V: dfx.fem.FunctionSpace,
    sol: dfx.fem.Function,
    object_coordinates: np.ndarray,
    filename: str,
) -> None:
    # u_topology, u_cell_types, u_geometry = dfx.plot.vtk_mesh(V)
    # u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    # u_grid.point_data["u"] = sol.x.array.real
    # u_grid.set_active_scalars("u")
    # # warped = u_grid.warp_by_scalar(factor=5.0)
    # u_plotter = pyvista.Plotter(off_screen=True)
    # u_plotter.add_mesh(warped, show_scalar_bar=True, cmap="seismic")
    # u_plotter.show(screenshot=f"{filename}.png")
    pyvista.OFF_SCREEN = True
    u_topology, u_cell_types, u_geometry = dfx.plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = sol.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=False, cmap="coolwarm", clim=(-1.0, 1.0))

    # Create the marker mesh
    coords = object_coordinates
    marker_points = pyvista.PolyData(coords)

    # Use glyphs to make them visible (spheres)
    sphere = pyvista.Sphere(radius=0.025)  # adjust radius for visibility
    marker_glyphs = marker_points.glyph(geom=sphere)
    u_plotter.add_mesh(marker_glyphs, color="white")

    # --- Lines connecting the markers ---
    # Connect them in the order given
    n_points = coords.shape[0]
    line_indices = np.arange(n_points, dtype=np.int32)

    # PolyData lines: [n_points, p0, p1, p2, ...]
    lines = np.hstack([[n_points + 1], np.append(line_indices, 0)]).astype(np.int32)
    line_mesh = pyvista.PolyData()
    line_mesh.points = coords
    line_mesh.lines = lines
    u_plotter.add_mesh(line_mesh, color="white", line_width=1)

    u_plotter.view_xy()
    # u_plotter.show(screenshot=f"{filename}.png")

    if pyvista.OFF_SCREEN:
        u_plotter.screenshot(f"{filename}", window_size=[2400, 2400])
    else:
        u_plotter.show()
    logger.debug(f"Finished plotting {filename}")

def plot_sol_with_object_parallel(
    V: dfx.fem.FunctionSpace,
    sol: dfx.fem.Function,
    object_coordinates: np.ndarray,
    filename: str,
) -> None:
    # ###
    # #  ToDo doesnt work yet, maybe one shouldnt even do that
    # ###
    domain = V.mesh
    topology, cell_types, geometry = dfx.plot.vtk_mesh(domain)
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    num_dofs_per_cell = topology[0]
    global_geometry = domain.comm.gather(geometry[:, :], root=0)
    global_topology = domain.comm.gather(
        topology[: (num_dofs_per_cell + 1) * num_cells_local], root=0
    )
    global_ct = domain.comm.gather(cell_types[:num_cells_local], root=0)
    global_sol = domain.comm.gather(sol.x.array, root=0)
    if domain.comm.rank == 0:
        plotter = pyvista.Plotter()
        grid = pyvista.UnstructuredGrid(global_topology[0], global_ct[0], global_geometry[0])
        grid.point_data["u"] = sol.x.array
        grid.set_active_scalars("u")
        plotter.add_mesh(grid, line_width=1, show_edges=True)
        plotter.view_xy()
        if pyvista.OFF_SCREEN:
            plotter.screenshot(f"{filename}", window_size=[1800, 1800])
        else:
            plotter.show()
        logger.debug(f"Finished plotting {filename}")


def plot_mesh_with_material(mesh:dfx.mesh.Mesh, meshtags: dfx.mesh.MeshTags, filename: Path):
    V = dfx.fem.functionspace(mesh,('DG',1))
    kappa_outer = dfx.fem.Expression(dfx.fem.Constant(V.mesh, PETSc.ScalarType(1.0)), V.element.interpolation_points)
    kappa_inner = dfx.fem.Expression(dfx.fem.Constant(V.mesh, PETSc.ScalarType(1.5)), V.element.interpolation_points)
    kappa = dfx.fem.Function(V)
    kappa.interpolate(kappa_outer, cells0=meshtags.find(1))
    kappa.interpolate(kappa_inner, cells0=meshtags.find(2))

    u_topology, u_cell_types, u_geometry = dfx.plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = kappa.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.screenshot(filename)

def plot_mesh_with_face_marker(mesh:dfx.mesh.Mesh, facetags: dfx.mesh.MeshTags, tag_value: int, filename: Path):
    V = dfx.fem.functionspace(mesh, ("DG", 1))
    u = dfx.fem.Function(V)
    c_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = c_map.size_local + c_map.num_ghosts
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    c_to_f = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    for cell in range(num_cells):
        for facet in c_to_f.links(cell):
            if facetags.values[facet] == tag_value:
                # print(cell, facet, V.dofmap.cell_dofs(cell))
                u.x.array[V.dofmap.cell_dofs(cell)] = 2

    u_topology, u_cell_types, u_geometry = dfx.plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = u.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.screenshot(filename)
