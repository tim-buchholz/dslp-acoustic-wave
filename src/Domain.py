import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from typing import Any, Tuple, Optional
from scipy.spatial import KDTree
import warnings
from DolfinVersion import DIRICHLET_BC_TYPE, geometric_dimension
from Config import DOF_TOLERANCE


# use submeshing from parallel -> needs some adaptation in subdomain creation and DSreg
def find_boundary_dofs_DG(
    FS: dfx.fem.function.FunctionSpace,
    domain: dfx.mesh.Mesh,
    return_bc_vertices: bool = False,
):
    ## see https://fenicsproject.discourse.group/t/the-problem-of-dirichlet-boundary-conditions-for-dg/12598
    tdim = domain.topology.dim
    fdim = tdim - 1

    vertex_map = domain.topology.index_map(0)
    num_vertices = vertex_map.size_local + vertex_map.num_ghosts
    vertices = np.arange(num_vertices, dtype=np.int32)
    domain.topology.create_connectivity(0, fdim)
    v_to_f = domain.topology.connectivity(0, fdim)
    domain.topology.create_connectivity(0, tdim)
    v_to_c = domain.topology.connectivity(0, tdim)
    domain.topology.create_connectivity(tdim, 0)
    c_to_v = domain.topology.connectivity(tdim, 0)
    boundary_vertices = np.array(
        [
            vertex
            for vertex in vertices
            if v_to_f.links(vertex).shape[0] > v_to_c.links(vertex).shape[0]
        ]
    ).astype(np.int32)
    boundary_dofs = []
    for vertex in boundary_vertices:
        for cell in v_to_c.links(vertex):
            for i, cell_vertex in enumerate(c_to_v.links(cell)):
                if cell_vertex == vertex:
                    boundary_dofs.append(FS.dofmap.cell_dofs(cell)[i])
                    break
    boundary_dofs_arr = np.array(boundary_dofs).astype(np.int32)
    if return_bc_vertices:
        return boundary_dofs_arr, boundary_vertices
    return boundary_dofs_arr


def get_cells_around_interface_CG(
    V_global: dfx.fem.function.FunctionSpace, interface_dofs_global: np.ndarray
) -> np.ndarray:
    c_map = V_global.mesh.topology.index_map(V_global.mesh.topology.dim)
    num_cells_global = c_map.size_local + c_map.num_ghosts
    additional_cells = []
    for cell in np.arange(num_cells_global).astype(np.int32):
        if np.any(
            np.equal.outer(V_global.dofmap.cell_dofs(cell), interface_dofs_global)
        ):
            additional_cells.append(cell)

    cells_around_interface = np.unique(np.array(additional_cells).astype(np.int32))
    return cells_around_interface


def get_cells_around_interface_DG(
    domain: dfx.mesh.Mesh, interface_vertices: np.ndarray
):
    # get interface vertices including doublings and use v_to_c map -> gives us one layer
    # use functionality from ov sd generation to extend
    tdim = domain.topology.dim
    v_to_c = domain.topology.connectivity(0, tdim)
    additional_cells = []
    for vertex in interface_vertices:
        for cell in v_to_c.links(vertex):
            additional_cells.append(cell)
    cells_around_interface = np.unique(np.array(additional_cells).astype(np.int32))
    return cells_around_interface


def extend_cells_by_layers(
    domain: dfx.mesh.Mesh,
    reference_cells: np.ndarray,
    num_layers: int,
):
    tdim = domain.topology.dim
    vdim = 0
    domain.topology.create_entities(vdim)
    vertex_map = domain.topology.index_map(vdim)
    num_vertices = vertex_map.size_local + vertex_map.num_ghosts
    vertices = np.arange(num_vertices, dtype=np.int32)
    vertex_marker_values = np.zeros_like(vertices, dtype=np.int32)

    ## connectivity
    domain.topology.create_connectivity(tdim, vdim)
    c_to_v = domain.topology.connectivity(tdim, vdim)

    # mark values
    for global_cell in reference_cells:
        vertex_marker_values[c_to_v.links(global_cell)] = 1
    vt = dfx.mesh.meshtags(domain, vdim, vertices, vertex_marker_values)
    c_map = domain.topology.index_map(tdim)
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
        vt = dfx.mesh.meshtags(domain, vdim, vertices, vertex_marker_values)
        extension = np.concatenate((extension, np.array(additional_cells))).astype(
            np.int32
        )

    return np.unique(extension)


class Domain:
    def __init__(self, mesh: dfx.mesh.Mesh, fem_type: str, degree: int) -> None:
        self.mesh: dfx.mesh.Mesh = mesh
        self.fem_type: str = fem_type
        self.degree: int = degree
        self.V: dfx.fem.function.FunctionSpace = dfx.fem.functionspace(
            mesh, (fem_type, degree)
        )
        self.geom_dim: int = geometric_dimension(mesh)
        self.dim: int = len(self.V.tabulate_dof_coordinates())
        self.boundary_dofs: np.ndarray[np.int32] = self.get_boundary_dofs().astype(
            np.int32
        )

    def get_boundary_dofs(self) -> np.ndarray:
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        if self.fem_type == "DG":
            if self.geom_dim == 1:
                num_inner_facets = 2
            else:
                cell_name = self.V.mesh.topology.cell_name()
                if cell_name == "triangle":
                    num_inner_facets = 6
                elif cell_name == "quadrilateral":
                    num_inner_facets = 4
                else:
                    raise NotImplementedError(
                        "Cant determine how many inner faces there are"
                    )

            self.mesh.topology.create_connectivity(fdim, tdim)
            c_to_f = self.mesh.topology.connectivity(tdim, fdim)

            c_map = self.mesh.topology.index_map(self.mesh.topology.dim)
            num_cells = c_map.size_local + c_map.num_ghosts
            # get boundary facets
            f_map = self.mesh.topology.index_map(self.mesh.topology.dim - 1)
            num_facets = f_map.size_local + f_map.num_ghosts
            boundary_facets = dfx.mesh.exterior_facet_indices(self.mesh.topology)
            # create meshtag
            facets = np.arange(num_facets, dtype=np.int32)
            values = np.zeros_like(facets, dtype=np.int32)
            values[boundary_facets] = 1
            ft = dfx.mesh.meshtags(
                self.mesh, self.mesh.topology.dim - 1, facets, values
            )
            self.mesh.topology.create_connectivity(0, fdim)
            v_to_f = self.mesh.topology.connectivity(0, fdim)
            self.mesh.topology.create_connectivity(0, tdim)
            c_to_v = self.mesh.topology.connectivity(tdim, 0)
            boundary_doflist = []
            for cell in range(num_cells):
                for facet in c_to_f.links(cell):
                    if ft.values[facet] == 1:
                        cell_vertices = c_to_v.links(cell)
                        for i, vertex in enumerate(cell_vertices):
                            if len(v_to_f.links(vertex)) < num_inner_facets:
                                boundary_doflist.append(
                                    self.V.dofmap.cell_dofs(cell)[i]
                                )

            boundary_dofs = np.unique(boundary_doflist)

            """
            CG_space = dfx.fem.functionspace(self.mesh, ("Lagrange", self.degree))
            CG_space.mesh.topology.create_connectivity(fdim, tdim)
            boundary_facets_CG = dfx.mesh.exterior_facet_indices(CG_space.mesh.topology)
            boundary_dofs_CG = dfx.fem.locate_dofs_topological(
                CG_space, fdim, boundary_facets_CG
            )
            self.boundary_coords_CG = CG_space.tabulate_dof_coordinates()[
                boundary_dofs_CG
            ]

            def on_boundary(x):
                tree = KDTree(self.boundary_coords_CG)
                matching_indices = tree.query_ball_point(x.transpose(), DOF_TOLERANCE)

                return np.array([len(indices) > 0 for indices in matching_indices])

            boundary_dofs = dfx.mesh.locate_entities(self.mesh, fdim, on_boundary)
            """
        else:
            self.mesh.topology.create_connectivity(fdim, tdim)
            boundary_facets = dfx.mesh.exterior_facet_indices(self.mesh.topology)
            boundary_dofs = dfx.fem.locate_dofs_topological(
                self.V, fdim, boundary_facets
            )

        return boundary_dofs

    def DirichletBC(self, uD: dfx.fem.function.Function) -> DIRICHLET_BC_TYPE:
        return dfx.fem.dirichletbc(uD, self.boundary_dofs)
        """ does not work
        if uD is dfx.fem.function.Function:
            return dfx.fem.dirichletbc(uD, self.boundary_dofs)
        else:
            return dfx.fem.dirichletbc(uD, self.boundary_dofs, self.V)
        """

    def get_h_min(self) -> float:
        dof_coordinates = self.V.tabulate_dof_coordinates()
        diams = np.zeros((dof_coordinates.shape[0],))
        for cell in range(
            self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        ):
            cell_dof_indices = self.V.dofmap.list[cell]
            cell_coordinates = dof_coordinates[cell_dof_indices]
            # self.mesh.geometry.x[cell_dof_indices]

            distances = np.linalg.norm(
                cell_coordinates[:, np.newaxis] - cell_coordinates, axis=2
            )
            np.fill_diagonal(distances, np.inf)
            h = np.min(distances)
            diams[cell_dof_indices] = np.maximum(diams[cell_dof_indices], h)
        return min(diams)


class SubDomain(Domain):
    def __init__(
        self, mesh: dfx.mesh.Mesh, fem_type: str, degree: int, parent: Domain
    ) -> None:
        super().__init__(mesh, fem_type, degree)
        self.parent: Domain = parent
        self.parentV: dfx.fem.function.FunctionSpace = parent.V
        if fem_type == "DG" and degree > 0:
            self.g2lmap: np.ndarray = self.global_to_local_dofmap_DG()
        else:
            self.g2lmap: np.ndarray = self.global_to_local_dofmap()
        self.boundary_dofs_global: np.ndarray = self.local_dof_to_global_dof(
            self.boundary_dofs
        )
        self.parent_boundary_dofs: np.ndarray = self.get_parent_boundary_dofs()
        self.interface_dofs: np.ndarray = self.get_interface_dofs()
        self.interface_dofs_global: np.ndarray = self.local_dof_to_global_dof(
            self.interface_dofs
        )

    def global_to_local_dofmap(self, secure_interpolation: bool = False) -> np.ndarray:
        numbering_vec = np.arange(self.parent.dim, dtype=np.int32)
        vec = dfx.fem.Function(self.parentV)
        vec.x.array[:] = numbering_vec[:]
        local = dfx.fem.Function(self.V)
        if secure_interpolation:
            print("Calculating subdomain dofmap by hand, this may take some time...")
            parent_coords = self.parentV.tabulate_dof_coordinates()
            for i, coord in enumerate(self.V.tabulate_dof_coordinates()):
                index_parent = np.argmin(np.sum(np.abs(parent_coords - coord), axis=1))
                local.x.array[i] = index_parent
        else:  
            # New interface, see PR #3177
            num_cells_local = self.V.dofmap.list.shape[0]
            cells = np.arange(num_cells_local,dtype=np.int32)
            interpolation_data = (
                dfx.fem.create_interpolation_data(
                    self.V,
                    self.parentV,
                    cells,
                    padding=DOF_TOLERANCE,
                )
            )
            local.interpolate_nonmatching(
                vec,
                cells,
                interpolation_data,
            )
        dmap = np.round(local.x.array[:]).astype(np.int32)
        return dmap

    def global_to_local_dofmap_DG(self):
        local = dfx.fem.Function(self.V)
        # step 1: construct a cell map from local context to global context
        DG0_space = dfx.fem.functionspace(self.parentV.mesh, ("DG", 0))
        DG0_space_local = dfx.fem.functionspace(self.V.mesh, ("DG", 0))
        # transpose mapping cell -> dofs
        dofs_per_cell = len(self.parentV.dofmap.list[0])
        global_dof_to_cell = np.repeat(
            np.arange(
                self.parentV.mesh.topology.index_map(
                    self.parentV.mesh.topology.dim
                ).size_local
            ),
            dofs_per_cell,
        )

        dg0_cells = dfx.fem.Function(DG0_space)
        cells = dfx.fem.Function(self.parentV)
        cells.x.array[:] = global_dof_to_cell
        dg0_cells.interpolate(cells)
        dg0_cells_local = dfx.fem.Function(DG0_space_local)

        # New interface, see PR #3177
        num_cells_local = DG0_space_local.dofmap.list.shape[0]
        cells = np.arange(num_cells_local, dtype=np.int32)
        interpolation_data = dfx.fem.create_interpolation_data(
            DG0_space_local,
            DG0_space,
            cells,
            padding=DOF_TOLERANCE,
        )
        dg0_cells_local.interpolate_nonmatching(dg0_cells, cells, interpolation_data)

        g2l_cell_map = np.round(dg0_cells_local.x.array[:]).astype(np.int32)
        # step 2: use cell map to construct dofmap
        for cell in range(
            self.V.mesh.topology.index_map(self.V.mesh.topology.dim).size_local
        ):
            local_cell_dof_indices = self.V.dofmap.cell_dofs(cell)
            global_cell_dof_indices = self.parentV.dofmap.cell_dofs(g2l_cell_map[cell])
            # step 3: use coordinates to establish correct order, also sanity check
            local_cell_coordinates = self.V.tabulate_dof_coordinates()[
                local_cell_dof_indices
            ]
            global_cell_coordinates = self.parentV.tabulate_dof_coordinates()[
                global_cell_dof_indices
            ]
            shuffle_indices = [
                np.where(
                    (np.isclose(global_cell_coordinates, target_coordinate)).all(axis=1)
                )[0].item()
                for target_coordinate in local_cell_coordinates
            ]

            local.x.array[local_cell_dof_indices] = global_cell_dof_indices[
                shuffle_indices
            ]
        dmap = np.round(local.x.array[:]).astype(np.int32)
        return dmap

    def local_dof_to_global_dof(self, local_dofs: np.ndarray) -> np.ndarray:
        """
        maps local DoF's to global DoF's by using the g2lmap

        Args:
            local_dofs (np.ndarray): local DoF's

        Returns:
            np.ndarray: global DoF's
        """
        return self.g2lmap[local_dofs]

    def global_dof_to_local_dof(self, global_dofs: np.ndarray) -> np.ndarray:
        """
        finds the indices in the g2lmap array where the elements match the values in the global_dofs array.
        It returns a 1D NumPy array containing the resulting indices.

        Args:
            global_dofs (np.ndarray): global DoF's

        Returns:
            np.ndarray: subset of global DoF's translated to local DoF's
        """
        return np.where(np.isin(self.g2lmap, global_dofs))[0]

    def project_to_global(
        self, func: dfx.fem.function.Function
    ) -> dfx.fem.function.Function:
        global_func = dfx.fem.Function(
            self.parentV
        )  # write a test asserting that vec is zero !
        global_func.x.array[self.g2lmap] = func.x.array
        return global_func

    def restrict_to_local(
        self, func: dfx.fem.function.Function
    ) -> dfx.fem.function.Function:
        local_func = dfx.fem.Function(self.V)
        local_func.x.array[:] = func.x.array[self.g2lmap]
        return local_func

    def get_parent_boundary_dofs(self) -> np.ndarray:
        return self.global_dof_to_local_dof(self.parent.boundary_dofs).astype(np.int32)

    def get_interface_dofs(self) -> np.ndarray:
        return np.setdiff1d(self.boundary_dofs, self.parent_boundary_dofs).astype(
            np.int32
        )

    def get_dofs_around_interface(
        self,
        layers: int = 1,
    ) -> np.ndarray:
        _reference = self.interface_dofs_global
        cell_dofmap = self.parentV.dofmap.list
        for _ in range(layers):
            touches_reference = np.any(np.isin(cell_dofmap, _reference), axis=1)
            dofs_around = np.unique(
                np.concatenate(
                    [cell for i, cell in enumerate(cell_dofmap) if touches_reference[i]]
                )
            )
            _reference = dofs_around
        return _reference

    def get_dofs_around_interface_DG(
        self,
        layers: int = 1,
    ) -> np.ndarray:
        _reference = self.interface_dofs_global
        cell_dofmap = self.parentV.dofmap.list

        for _ in range(layers):
            _reference_coords = self.parentV.tabulate_dof_coordinates()[_reference]
            tree = KDTree(_reference_coords)

            def extends_reference(x):
                matching_indices = tree.query_ball_point(x.transpose(), DOF_TOLERANCE)

                return np.array([len(indices) > 0 for indices in matching_indices])

            self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
            vertices_ref = dfx.mesh.locate_entities(
                self.parentV.mesh, 0, extends_reference
            )
            c_to_v = self.parentV.mesh.topology.connectivity(self.geom_dim, 0).array
            c_map = self.parentV.mesh.topology.index_map(self.parentV.mesh.topology.dim)
            num_cells = c_map.size_local + c_map.num_ghosts

            contains_reference_vertex = np.any(
                np.isin(c_to_v.reshape((num_cells, -1)), vertices_ref), axis=1
            )
            _reference = cell_dofmap[contains_reference_vertex].flatten()
        return _reference

    def generate_prediction_subdomain(self, layers: int = 1) -> None:
        if self.fem_type == "DG":
            dofs_around = self.get_dofs_around_interface_DG(layers=layers)
        else:
            dofs_around = self.get_dofs_around_interface(layers=layers)

        coords = self.parentV.tabulate_dof_coordinates()[dofs_around]
        tree = KDTree(coords)

        def prediction_omega(x):
            matching_indices = tree.query_ball_point(x.transpose(), DOF_TOLERANCE)
            return np.array([len(indices) > 0 for indices in matching_indices])

        prediction_cells = dfx.mesh.locate_entities(
            self.parent.mesh, self.parent.mesh.topology.dim, prediction_omega
        )
        submsh, entity_map, vertex_map, geom_map = dfx.mesh.create_submesh(
            self.parent.mesh,
            geometric_dimension(self.parentV),
            prediction_cells,
        )
        self.prediction_subdomain = PredictionSubDomain(
            submsh, self.fem_type, self.degree, self.parent, self.interface_dofs_global
        )
        self.prediction_subdomain.num_layers = layers
        # later
        # func.x.array[self.interface_dofs] = predicition[self.prediction_subdomain.prediction_dofs]


class OvSubDomain(SubDomain):
    def __init__(
        self,
        ov_mesh: dfx.mesh.Mesh,
        non_ov_mesh: dfx.mesh.Mesh,
        fem_type: str,
        degree: int,
        parent: Domain,
    ) -> None:
        super().__init__(ov_mesh, fem_type, degree, parent)
        self.non_ov: SubDomain = SubDomain(non_ov_mesh, fem_type, degree, parent)
        self.mapping: np.ndarray = np.argsort(self.g2lmap)
        self.indices_non_ov: np.ndarray = np.searchsorted(
            self.g2lmap[self.mapping], self.non_ov.g2lmap
        )

    def restrict_to_non_overlapping_SD_via_global(
        self, func: dfx.fem.function.Function
    ) -> dfx.fem.function.Function:
        warnings.warn("try to use restrict_from_ov_to_non_ov_SD instead")
        global_func = self.project_to_global(func)  # can this be done better?
        local_func = self.non_ov.restrict_to_local(global_func)
        return local_func

    def restrict_from_ov_to_non_ov_SD(
        self, func: dfx.fem.function.Function
    ) -> dfx.fem.function.Function:
        """
        enables the transfer of data between subdomains in a domain decomposition context,
        specifically for restricting data from a larger subdomain to a smaller subdomain while preserving
        the correct ordering based on global-to-local index mappings (g2lmap).

                Args:
                    func (dfx.fem.function.Function): Function on Ov Subdomain

                Returns:
                    dfx.fem.function.Function: Function on the Non-Ov Subdomain
        """
        # Perform the restriction
        local_func = dfx.fem.Function(self.non_ov.V)
        local_func.x.array[:] = func.x.array[:][self.mapping][self.indices_non_ov]
        return local_func

    def non_overlapping_SD_to_global(
        self, func: dfx.fem.function.Function
    ) -> dfx.fem.function.Function:
        global_func = self.non_ov.project_to_global(func)
        return global_func


class PredictionSubDomain(SubDomain):
    def __init__(
        self,
        mesh: dfx.mesh.Mesh,
        fem_type: str,
        degree: int,
        parent: Domain,
        global_prediction_dofs: np.ndarray,
    ) -> None:
        super().__init__(mesh, fem_type, degree, parent)
        self.prediction_dofs_global = global_prediction_dofs
        self.prediction_dofs = self.global_dof_to_local_dof(self.prediction_dofs_global)

    def prediction_mask(self, func: dfx.fem.function.Function) -> None:
        mask = np.isin(np.arange(len(func.x.array)), self.prediction_dofs)
        func.x.array *= mask


class DS_reg(Domain):
    """
    Overlapping and Non-Overlapping Domain Decomposition where fem_type and degree of the subdomain spaces
    coincide with the global ones. Domains then get defined solely by there mesh and the
    global parameters.
    """

    def __init__(
        self,
        mesh: dfx.mesh.Mesh,
        fem_type: str,
        degree: int,
        SD_non_OV_meshlist: list[dfx.mesh.Mesh],
        SD_OV_meshlist: list[dfx.mesh.Mesh],
    ) -> None:
        super().__init__(mesh, fem_type, degree)
        self.SDs: list[OvSubDomain] = []  # list of overlapping subdomains
        assert len(SD_non_OV_meshlist) == len(SD_OV_meshlist)
        for i, meshi in enumerate(SD_OV_meshlist):
            SDi = OvSubDomain(meshi, SD_non_OV_meshlist[i], fem_type, degree, self)
            self.SDs.append(SDi)
        self.avg_weights: np.ndarray = self.get_avg_weights()
        self.assert_avg_ones()

    def get_avg_weights(self) -> np.ndarray:
        if self.fem_type == "DG":
            # in a DG context we need no averaging
            return np.array(None)

        one = dfx.fem.Function(self.V)
        one.x.array[:] = np.ones(one.x.array[:].shape)
        boundary_occurences = dfx.fem.Function(self.V)
        # boundary_occurences.interpolate(ScalarType(0.0)) # not needed but maybe safer
        for subdomain in self.SDs:
            # elementwise plus
            boundary_occurences.x.array[subdomain.non_ov.boundary_dofs_global] += 1.0
        return np.reciprocal(
            np.maximum(one.x.array[:], boundary_occurences.x.array[:]), dtype=float
        )

    def avg(self, *args: Tuple[dfx.fem.function.Function]) -> dfx.fem.function.Function:
        avg_func = dfx.fem.Function(self.V)
        # avg_func.interpolate(ScalarType(0.0)) # not needed
        for i, func in enumerate(args):
            non_ov_func = self.SDs[i].restrict_from_ov_to_non_ov_SD(func)
            avg_func.x.array[self.SDs[i].non_ov.g2lmap] += non_ov_func.x.array[:]

        if self.fem_type != "DG":
            avg_func.x.array[:] *= self.avg_weights

        return avg_func

    def assert_avg_ones(self):
        u = dfx.fem.Function(self.V)
        test = np.ones(self.dim)
        u.x.array[:] = test
        sd_sols = [SD.restrict_to_local(u) for SD in self.SDs]
        avg = self.avg(*sd_sols)
        np.testing.assert_array_equal(test, avg.x.array[:])
