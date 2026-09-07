from mpi4py import MPI
import dolfinx as dfx
import numpy as np
from pathlib import Path
from time import time 
import pandas as pd
import h5py
import logging
from typing import Any, Tuple, Optional, Callable, Dict
from CommunicationPlan import unflatten_schedule
from Meshing import read_mesh_with_meshtags
import logging
import sys

logger = logging.getLogger(__name__)

class Domain:

    def __init__(
        self,
        hdf5_filename: Path,
        communicator: MPI.Intracomm,
        ghost_mode: Optional[dfx.mesh.GhostMode] = None,
    ) -> None:
        self.communicator = communicator
        self.rank = self.communicator.Get_rank()
        self.ghost_mode = ghost_mode
        self.read_information(hdf5_filename)
        self.V: dfx.fem.function.FunctionSpace = dfx.fem.functionspace(
            self.mesh, (self.fem_type, self.degree)
        )
        logger.debug(f"== restored FunctionSpace '\n'  {self.V}")
        self.dim: int = self.mesh.ufl_domain().geometric_dimension()
        # variables
        self.nDoFs: int
        self.fem_type: str
        self.degree: int
        self.mesh: dfx.mesh.Mesh
        self.material_tags: dfx.mesh.MeshTags
        self.boundary_tags: dfx.mesh.MeshTags

    def read_information(self, hdf5_filename: str) -> None:
        if self.rank == 0:
            with h5py.File(hdf5_filename, "r") as f:
                logger.debug(f"== Reading {hdf5_filename}")
                logger.debug("Found keys: %s" % f.keys())
                logger.debug(f"Meshes contains \n %s" % f["Meshes"].attrs.keys())
                # readings
                boundary_dofs = f["Meshes"]["mesh"]["boundary"][()]
                data2bcast = {
                    "mesh_file_name": str(f["Meshes"].attrs["mesh"]),
                    "fem_type": str(f["Meshes"].attrs["fem-type"]),
                    "degree": int(f["Meshes"].attrs["degree"]),
                    "h_gmsh": int(f["Meshes"].attrs["h_gmsh"]),
                    "num_global_dofs": int(f["Meshes"].attrs["num_dofs_global"]),
                    "boundary_shape": boundary_dofs.shape,
                    "boundary_dtype": boundary_dofs.dtype.str,
                }

        else: 
            data2bcast = None 
            boundary_dofs = None

        data2bcast = self.communicator.bcast(data2bcast, root = 0)
        if self.rank != 0:
            boundary_dofs = np.empty(
                data2bcast["boundary_shape"],
                dtype=np.dtype(data2bcast["boundary_dtype"]),
            )
        self.communicator.Bcast(boundary_dofs, root=0)

        # store data
        self.mesh_file_name = Path(data2bcast["mesh_file_name"])
        self.fem_type       = data2bcast["fem_type"]
        self.degree         = data2bcast["degree"]
        self.h_gmsh         = data2bcast["h_gmsh"]
        self.num_global_dofs = data2bcast["num_global_dofs"]
        self.boundary_dofs  = boundary_dofs

        # reconstruct
        logger.debug(str(self.mesh_file_name))
        self.mesh, tag_dict = read_mesh_with_meshtags(
            self.mesh_file_name,
            self.communicator,
            mesh_tag_names=["material_tag", "boundary_tag"],
            ghost_mode = self.ghost_mode
        )
        self.material_tags = tag_dict["material_tag"]
        self.boundary_tags = tag_dict["boundary_tag"]
        logger.debug(
            f"degree={self.degree}, type={self.fem_type}, restored mesh: {self.mesh}"
        )


class SubDomain(Domain):
    def __init__(
        self,
        config_filename: str,
        subdomain_communicator: MPI.Intracomm,
        mesh_communicator: MPI.Intracomm,
    ) -> None:
        self.mesh_communicator = mesh_communicator
        super().__init__(config_filename, subdomain_communicator)
        self.communicator.Barrier()
        self.nonov_V: dfx.fem.function.FunctionSpace = dfx.fem.functionspace(
            self.nonov_mesh, (self.fem_type, self.degree)
        )
        logger.debug(f"== restored non-overlapping FunctionSpace '\n'  {self.V}")
        self.pred_V: dfx.fem.function.FunctionSpace = dfx.fem.functionspace(
            self.pred_mesh, (self.fem_type, self.degree)
        )
        logger.debug(f"== restored prediction FunctionSpace '\n'  {self.V}")
        # mesh variables
        self.mesh: dfx.mesh.Mesh
        self.material_tags: dfx.mesh.MeshTags
        self.boundary_tags: dfx.mesh.MeshTags
        self.nonov_mesh: dfx.mesh.Mesh
        self.non_overlapping_material_tags: dfx.mesh.MeshTags
        self.non_overlapping_boundary_tags: dfx.mesh.MeshTags
        self.pred_mesh: dfx.mesh.Mesh
        self.prediction_material_tags: dfx.mesh.MeshTags
        self.prediction_boundary_tags: dfx.mesh.MeshTags

        # prepare communication arrays
        self.translate_communication_arrays()
        self.mapping: np.ndarray = np.argsort(self.ov_l2gmap)
        self.indices_non_ov: np.ndarray = np.searchsorted(
            self.ov_l2gmap[self.mapping], self.nonov_l2gmap
        )
        (
            self.pred_dofs_in_ov,
            self.pred_dofs_in_ov_id_ov,
            self.pred_dofs_in_ov_id_pred,
        ) = np.intersect1d(self.ov_l2gmap, self.pred_l2gmap, assume_unique= True, return_indices=True)
        self.nonov_interface_dofs_ov_indexing = self.global_dof_to_local_ov_dof(
            self.nonov_interface_dofs_global
        )
        self.avg_weights = self.derive_averaging_weights()
        self._prepare_comm_buffers()
        self._prepare_allocation_free_comm()

    def read_information_new(self, hdf5_filename: str) -> None:
        if self.rank == 0:
            # -------- Read everything on rank 0 --------
            with h5py.File(hdf5_filename, "r") as f:
                logger.debug(f"== Reading {hdf5_filename}")
                logger.debug("Found keys: %s" % f.keys())
                logger.debug(f"Meshes contains \n %s" % f["Meshes"].attrs.keys())

                # small scalars
                small_meta = {
                    "fem_type": str(f["Meshes"].attrs["fem-type"]),
                    "degree": int(f["Meshes"].attrs["degree"]),
                    "ell": int(f["Meshes"].attrs["ell"]),
                    "h_gmsh": int(f["Meshes"].attrs["h_gmsh"]),
                    "prediction_ell": int(f["Meshes"].attrs["prediction_ell"]),
                    "num_global_dofs": int(f["Meshes"].attrs["num_dofs_global"]),
                    "num_nonov_dofs": int(f["Meshes"].attrs["num_dofs_nonov"]),
                    "num_ov_dofs": int(f["Meshes"].attrs["num_dofs_ov"]),
                    "num_pred_dofs": int(f["Meshes"].attrs["num_dofs_pred"]),
                    # filenames as strings (will be used by each rank to load its mesh)
                    "nonov_mesh_file_name": str(f["Meshes"].attrs["non-overlapping"]),
                    "ov_mesh_file_name": str(f["Meshes"].attrs["overlapping"]),
                    "pred_mesh_file_name": str(f["Meshes"].attrs["prediction"]),
                }
                # large arrays (read here, broadcast via Bcast)
                nonov_entity_map = f["Meshes"]["non-overlapping"]["entity_map"][()]
                nonov_l2gmap = f["Meshes"]["non-overlapping"]["l2gmap"][()]
                nonov_boundary_dofs = f["Meshes"]["non-overlapping"]["boundary"][()]
                nonov_interface_dofs_global = f["Meshes"]["non-overlapping"]["interface"][
                    ()
                ]

                ov_entity_map = f["Meshes"]["overlapping"]["entity_map"][()]
                ov_l2gmap = f["Meshes"]["overlapping"]["l2gmap"][()]
                ov_boundary_dofs = f["Meshes"]["overlapping"]["boundary"][()]
                ov_interface_dofs_global = f["Meshes"]["overlapping"]["interface"][()]

                pred_entity_map = f["Meshes"]["prediction"]["entity_map"][()]
                pred_l2gmap = f["Meshes"]["prediction"]["l2gmap"][()]
                pred_boundary_dofs = f["Meshes"]["prediction"]["boundary"][()]

                # Prepare shape/dtype descriptors used by receivers to allocate buffers
                big_array_meta = {
                    "nonov_entity_map": (
                        nonov_entity_map.shape,
                        nonov_entity_map.dtype.str,
                    ),
                    "nonov_l2gmap": (nonov_l2gmap.shape, nonov_l2gmap.dtype.str),
                    "nonov_boundary_dofs": (
                        nonov_boundary_dofs.shape,
                        nonov_boundary_dofs.dtype.str,
                    ),
                    "nonov_interface_dofs_global": (
                        nonov_interface_dofs_global.shape,
                        nonov_interface_dofs_global.dtype.str,
                    ),
                    "ov_entity_map": (ov_entity_map.shape, ov_entity_map.dtype.str),
                    "ov_l2gmap": (ov_l2gmap.shape, ov_l2gmap.dtype.str),
                    "ov_boundary_dofs": (
                        ov_boundary_dofs.shape,
                        ov_boundary_dofs.dtype.str,
                    ),
                    "ov_interface_dofs_global": (
                        ov_interface_dofs_global.shape,
                        ov_interface_dofs_global.dtype.str,
                    ),
                    "pred_entity_map": (pred_entity_map.shape, pred_entity_map.dtype.str),
                    "pred_l2gmap": (pred_l2gmap.shape, pred_l2gmap.dtype.str),
                    "pred_boundary_dofs": (
                        pred_boundary_dofs.shape,
                        pred_boundary_dofs.dtype.str,
                    ),
                }
                # communication dicts: shapes and dtypes
                communication_incoming = {}
                comm_in_meta = {}
                comm_in_dtype = {}
                for key in f["IncomingCommunication"].attrs.keys():
                    interior = f["IncomingCommunication"]["from" + key]["interior"][()]
                    prediction = f["IncomingCommunication"]["from" + key]["prediction"][()]
                    interface = f["IncomingCommunication"]["from" + key]["interface"][()]
                    key_int = int(key)
                    communication_incoming[key_int] = [interior, prediction, interface]
                    comm_in_meta[key_int] = [arr.shape for arr in [interior, prediction, interface]]
                    comm_in_dtype[key_int] = [arr.dtype.str for arr in [interior, prediction, interface]]

                communication_outgoing = {}
                comm_out_meta = {}
                comm_out_dtype = {}
                for key in f["OutgoingCommunication"].attrs.keys():
                    interior = f["OutgoingCommunication"]["to" + key]["interior"][()]
                    prediction = f["OutgoingCommunication"]["to" + key]["prediction"][()]
                    interface = f["OutgoingCommunication"]["to" + key]["interface"][()]
                    key_int = int(key)
                    communication_outgoing[key_int] = [interior, prediction, interface]
                    comm_out_meta[key_int] = [arr.shape for arr in [interior, prediction, interface]]
                    comm_out_dtype[key_int] = [arr.dtype.str for arr in [interior, prediction, interface]]

                comm_schedule = unflatten_schedule(
                    f["OutgoingCommunication"]["schedule"][()]
                )
                logger.debug("Rank 0 finished reading file.")
        else:
            # non-root placeholders
            small_meta = None
            big_array_meta = None
            comm_in_meta = None
            comm_out_meta = None
            comm_in_dtype = None
            comm_out_dtype = None
            comm_schedule = None
        
        small_meta = self.communicator.bcast(small_meta, root=0)
        big_array_meta = self.communicator.bcast(big_array_meta, root=0)
        comm_in_meta = self.communicator.bcast(comm_in_meta, root=0)
        comm_out_meta = self.communicator.bcast(comm_out_meta, root=0)
        comm_in_dtype = self.communicator.bcast(comm_in_dtype, root=0)
        comm_out_dtype = self.communicator.bcast(comm_out_dtype, root=0)
        comm_schedule = self.communicator.bcast(comm_schedule, root=0)

        if self.rank != 0:
            nonov_entity_map = np.empty(big_array_meta["nonov_entity_map"][0], dtype=np.dtype(big_array_meta["nonov_entity_map"][1]))
            nonov_l2gmap = np.empty(big_array_meta["nonov_l2gmap"][0], dtype=np.dtype(big_array_meta["nonov_l2gmap"][1]))
            nonov_boundary_dofs = np.empty(big_array_meta["nonov_boundary_dofs"][0], dtype=np.dtype(big_array_meta["nonov_boundary_dofs"][1]))
            nonov_interface_dofs_global = np.empty(big_array_meta["nonov_interface_dofs_global"][0], dtype=np.dtype(big_array_meta["nonov_interface_dofs_global"][1]))

            ov_entity_map = np.empty(big_array_meta["ov_entity_map"][0], dtype=np.dtype(big_array_meta["ov_entity_map"][1]))
            ov_l2gmap = np.empty(big_array_meta["ov_l2gmap"][0], dtype=np.dtype(big_array_meta["ov_l2gmap"][1]))
            ov_boundary_dofs = np.empty(big_array_meta["ov_boundary_dofs"][0], dtype=np.dtype(big_array_meta["ov_boundary_dofs"][1]))
            ov_interface_dofs_global = np.empty(big_array_meta["ov_interface_dofs_global"][0], dtype=np.dtype(big_array_meta["ov_interface_dofs_global"][1]))

            pred_entity_map = np.empty(big_array_meta["pred_entity_map"][0], dtype=np.dtype(big_array_meta["pred_entity_map"][1]))
            pred_l2gmap = np.empty(big_array_meta["pred_l2gmap"][0], dtype=np.dtype(big_array_meta["pred_l2gmap"][1]))
            pred_boundary_dofs = np.empty(big_array_meta["pred_boundary_dofs"][0], dtype=np.dtype(big_array_meta["pred_boundary_dofs"][1]))

            communication_incoming = {
                key: [np.empty(shape, dtype=np.dtype(dtype)) for shape, dtype in zip(comm_in_meta[key], comm_in_dtype[key])]
                for key in comm_in_meta
            }
            communication_outgoing = {
                key: [np.empty(shape, dtype=np.dtype(dtype)) for shape, dtype in zip(comm_out_meta[key], comm_out_dtype[key])]
                for key in comm_out_meta
            }

        # ----------------- Broadcast arrays -----------------
        root = 0
        for arr in [nonov_entity_map, nonov_l2gmap, nonov_boundary_dofs, nonov_interface_dofs_global,
                    ov_entity_map, ov_l2gmap, ov_boundary_dofs, ov_interface_dofs_global,
                    pred_entity_map, pred_l2gmap, pred_boundary_dofs]:
            self.communicator.Bcast(arr, root=root)

        for comm_dict in [communication_incoming, communication_outgoing]:
            for key, arr_list in comm_dict.items():
                for arr in arr_list:
                    self.communicator.Bcast(arr, root=root)

        # ----------------- Broadcast schedule -----------------
        comm_schedule = self.communicator.bcast(comm_schedule, root=root)
        self.communicator.Barrier()

        self.fem_type = small_meta["fem_type"]
        self.degree = small_meta["degree"]
        self.ell = small_meta["ell"]
        self.h_gmsh = small_meta["h_gmsh"]
        self.prediction_ell = small_meta["prediction_ell"]
        self.num_global_dofs = small_meta["num_global_dofs"]
        self.num_nonov_dofs = small_meta["num_nonov_dofs"]
        self.num_ov_dofs = small_meta["num_ov_dofs"]
        self.num_pred_dofs = small_meta["num_pred_dofs"]
        nonov_mesh_file_name = Path(small_meta["nonov_mesh_file_name"])
        ov_mesh_file_name = Path(small_meta["ov_mesh_file_name"])
        pred_mesh_file_name = Path(small_meta["pred_mesh_file_name"])
        self.nonov_entity_map = nonov_entity_map
        self.nonov_l2gmap = nonov_l2gmap
        self.nonov_boundary_dofs = nonov_boundary_dofs
        self.nonov_interface_dofs_global = nonov_interface_dofs_global

        self.ov_entity_map = ov_entity_map
        self.ov_l2gmap = ov_l2gmap
        self.ov_boundary_dofs = ov_boundary_dofs
        self.ov_interface_dofs_global = ov_interface_dofs_global

        self.pred_entity_map = pred_entity_map
        self.pred_l2gmap = pred_l2gmap
        self.pred_boundary_dofs = pred_boundary_dofs

        # communication dicts and schedule
        self.communication_incoming = communication_incoming
        self.communication_outgoing = communication_outgoing
        self.comm_schedule = comm_schedule

        self.mesh, tag_dict = read_mesh_with_meshtags(
            ov_mesh_file_name,
            self.mesh_communicator,
            mesh_tag_names=["material_tag", "boundary_tag"],
        )
        self.material_tags = tag_dict["material_tag"]
        self.boundary_tags = tag_dict["boundary_tag"]

        self.nonov_mesh, tag_dict = read_mesh_with_meshtags(
            nonov_mesh_file_name,
            self.mesh_communicator,
            mesh_tag_names=["material_tag", "boundary_tag"],
        )
        self.non_overlapping_material_tags = tag_dict["material_tag"]
        self.non_overlapping_boundary_tags = tag_dict["boundary_tag"]

        self.pred_mesh, tag_dict = read_mesh_with_meshtags(
            pred_mesh_file_name,
            self.mesh_communicator,
            mesh_tag_names=["material_tag", "boundary_tag"],
        )
        self.prediction_material_tags = tag_dict["material_tag"]
        self.prediction_boundary_tags = tag_dict["boundary_tag"]

        # ----------------- Post-processing (same as original) -----------------
        logger.debug(
            f"degree={self.degree}, type={self.fem_type}, restored mesh: {self.mesh}, DoF's: {self.num_ov_dofs}"
        )
        logger.debug(
            f"degree={self.degree}, type={self.fem_type}, restored nonov_mesh: {self.nonov_mesh}, DoF's: {self.num_nonov_dofs}"
        )
        logger.debug(
            f"degree={self.degree}, type={self.fem_type}, restored pred_mesh: {self.pred_mesh}, DoF's: {self.num_pred_dofs}"
        )

        self.ov_interface_dofs_local = self.global_dof_to_local_ov_dof(
            self.ov_interface_dofs_global
        )
        self.ov_boundary_without_interface_dofs_local = np.setdiff1d(
            self.ov_boundary_dofs, self.ov_interface_dofs_local
        )
        self.ov_boundary_without_interface_dofs_global = self.local_dof_to_global_dof(
            self.ov_boundary_without_interface_dofs_local
        )

        self.pred_dofs_local = self.global_dof_to_local_pred_dof(
            self.ov_interface_dofs_global
        )
        logging.debug(f" # prediction dofs {len(self.pred_dofs_local)}")

    def read_information(self, hdf5_filename: str) -> None:
        with h5py.File(hdf5_filename, "r") as f:
            logger.debug(f"== Reading {hdf5_filename}")
            logger.debug("Found keys: %s" % f.keys())
            logger.debug(f"Meshes contains \n %s" % f["Meshes"].attrs.keys())
            self.fem_type: str = f["Meshes"].attrs["fem-type"]
            self.degree: int = f["Meshes"].attrs["degree"]
            self.ell: int = f["Meshes"].attrs["ell"]
            self.h_gmsh: int = f["Meshes"].attrs["h_gmsh"]
            self.prediction_ell: int = f["Meshes"].attrs["prediction_ell"]
            nonov_mesh_file_name: Path = Path(f["Meshes"].attrs["non-overlapping"])
            ov_mesh_file_name: Path = Path(f["Meshes"].attrs["overlapping"])
            pred_mesh_file_name: Path = Path(f["Meshes"].attrs["prediction"])
            self.num_global_dofs: int = f["Meshes"].attrs["num_dofs_global"]
            self.num_nonov_dofs: int = f["Meshes"].attrs["num_dofs_nonov"]
            self.num_ov_dofs: int = f["Meshes"].attrs["num_dofs_ov"]
            self.num_pred_dofs: int = f["Meshes"].attrs["num_dofs_pred"]

            # restore nonov metadata
            self.nonov_entity_map: np.ndarray = f["Meshes"]["non-overlapping"][
                "entity_map"
            ][()]
            self.nonov_l2gmap: np.ndarray = f["Meshes"]["non-overlapping"]["l2gmap"][()]
            self.nonov_boundary_dofs: np.ndarray = f["Meshes"]["non-overlapping"][
                "boundary"
            ][()]
            self.nonov_interface_dofs_global: np.ndarray = f["Meshes"][
                "non-overlapping"
            ]["interface"][()]
            # restore ov metadata
            self.ov_entity_map: np.ndarray = f["Meshes"]["overlapping"]["entity_map"][
                ()
            ]
            self.ov_l2gmap: np.ndarray = f["Meshes"]["overlapping"]["l2gmap"][()]
            self.ov_boundary_dofs: np.ndarray = f["Meshes"]["overlapping"]["boundary"][
                ()
            ]
            self.ov_interface_dofs_global: np.ndarray = f["Meshes"]["overlapping"][
                "interface"
            ][()]

            self.pred_entity_map: np.ndarray = f["Meshes"]["prediction"]["entity_map"][
                ()
            ]
            self.pred_l2gmap: np.ndarray = f["Meshes"]["prediction"]["l2gmap"][()]
            self.pred_boundary_dofs: np.ndarray = f["Meshes"]["prediction"]["boundary"][
                ()
            ]

            logger.debug(
                f"Fetching incoming communication, keys: \n %s"
                % f["IncomingCommunication"].attrs.keys()
            )
            self.communication_incoming = {}
            for key in list(f["IncomingCommunication"].attrs.keys()):
                interior_dofs = f["IncomingCommunication"]["from" + key]["interior"][()]
                prediction_dofs = f["IncomingCommunication"]["from" + key][
                    "prediction"
                ][()]
                interface_dofs = f["IncomingCommunication"]["from" + key]["interface"][
                    ()
                ]
                self.communication_incoming[int(key)] = [
                    interior_dofs,
                    prediction_dofs,
                    interface_dofs,
                ]
            logger.info(
                f"Fetching outgoing communication, keys: \n %s"
                % f["OutgoingCommunication"].attrs.keys()
            )
            self.communication_outgoing = {}
            for key in list(f["OutgoingCommunication"].attrs.keys()):
                interior_dofs = f["OutgoingCommunication"]["to" + key]["interior"][()]
                prediction_dofs = f["OutgoingCommunication"]["to" + key]["prediction"][
                    ()
                ]
                interface_dofs = f["OutgoingCommunication"]["to" + key]["interface"][()]
                self.communication_outgoing[int(key)] = [
                    interior_dofs,
                    prediction_dofs,
                    interface_dofs,
                ]
            logger.debug("Fetching communication schedule")
            self.comm_schedule = unflatten_schedule(
                f["OutgoingCommunication"]["schedule"][()]
            )
            logger.debug("Communication schedule:")
            for i, round in enumerate(self.comm_schedule):
                logger.debug(f"{i} # {[(r[0],r[1]) for r in round]}")

            # read overlapping mesh
            self.mesh, tag_dict = read_mesh_with_meshtags(
                ov_mesh_file_name,
                self.mesh_communicator,
                mesh_tag_names=["material_tag", "boundary_tag"],
            )
            self.material_tags = tag_dict["material_tag"]
            self.boundary_tags = tag_dict["boundary_tag"]
            # read non overlapping mesh (do we even need this?)
            self.nonov_mesh, tag_dict = read_mesh_with_meshtags(
                nonov_mesh_file_name,
                self.mesh_communicator,
                mesh_tag_names=["material_tag", "boundary_tag"],
            )
            self.non_overlapping_material_tags = tag_dict["material_tag"]
            self.non_overlapping_boundary_tags = tag_dict["boundary_tag"]
            # read prediction mesh
            self.pred_mesh, tag_dict = read_mesh_with_meshtags(
                pred_mesh_file_name,
                self.mesh_communicator,
                mesh_tag_names=["material_tag", "boundary_tag"],
            )
            self.prediction_material_tags = tag_dict["material_tag"]
            self.prediction_boundary_tags = tag_dict["boundary_tag"]

            logger.debug(
                f"degree={self.degree}, type={self.fem_type}, restored mesh: {self.mesh}, DoF's: {self.num_ov_dofs}"
            )
            logger.debug(
                f"degree={self.degree}, type={self.fem_type}, restored mesh: {self.nonov_mesh}, DoF's: {self.num_nonov_dofs}"
            )
            logger.debug(
                f"degree={self.degree}, type={self.fem_type}, restored mesh: {self.pred_mesh}, DoF's: {self.num_pred_dofs}"
            )

            self.ov_interface_dofs_local = self.global_dof_to_local_ov_dof(
                self.ov_interface_dofs_global
            )
            self.ov_boundary_without_interface_dofs_local = np.setdiff1d(self.ov_boundary_dofs, self.ov_interface_dofs_local)
            self.ov_boundary_without_interface_dofs_global = (
                self.local_dof_to_global_dof(
                    self.ov_boundary_without_interface_dofs_local
                )
            )
            # restore predicition metadata

            self.pred_dofs_local: np.ndarray = self.global_dof_to_local_pred_dof(
                self.ov_interface_dofs_global
            )
            logging.debug(f" # prediction dofs {len(self.pred_dofs_local)}")

    def local_dof_to_global_dof(self, local_dofs: np.ndarray) -> np.ndarray:
        """
        maps local DoF's to global DoF's by using the l2gmap

        Args:
            local_dofs (np.ndarray): local DoF's

        Returns:
            np.ndarray: global DoF's
        """
        return self.ov_l2gmap[local_dofs]

    def global_dof_to_local_ov_dof(self, global_dofs: np.ndarray) -> np.ndarray:
        """
        finds the indices in the ov_l2gmap array where the elements match the values in the global_dofs array.
        It returns a 1D NumPy array containing the resulting indices.

        Args:
            global_dofs (np.ndarray): global DoF's

        Returns:
            np.ndarray: subset of global DoF's translated to local DoF's
        """
        reverse_dofmap_series = pd.Series(
            data=np.arange(self.ov_l2gmap.size), index=self.ov_l2gmap
        )
        return reverse_dofmap_series[global_dofs].values

    def global_dof_to_local_nonov_dof(self, global_dofs: np.ndarray) -> np.ndarray:
        """
        finds the indices in the nonov_l2gmap array where the elements match the values in the global_dofs array.
        It returns a 1D NumPy array containing the resulting indices.

        Args:
            global_dofs (np.ndarray): global DoF's

        Returns:
            np.ndarray: subset of global DoF's translated to local DoF's
        """
        reverse_dofmap_series = pd.Series(
            data=np.arange(self.nonov_l2gmap.size), index=self.nonov_l2gmap
        )
        return reverse_dofmap_series[global_dofs].values

    def global_dof_to_local_pred_dof(self, global_dofs: np.ndarray) -> np.ndarray:
        """
        finds the indices in the pred_l2gmap array where the elements match the values in the global_dofs array.
        It returns a 1D NumPy array containing the resulting indices.

        Args:
            global_dofs (np.ndarray): global DoF's

        Returns:
            np.ndarray: subset of global DoF's translated to local DoF's
        """
        reverse_dofmap_series = pd.Series(
            data=np.arange(self.pred_l2gmap.size), index=self.pred_l2gmap
        )
        return reverse_dofmap_series[global_dofs].values

    def restrict_from_ov_to_non_ov_SD(
        self, func: dfx.fem.function.Function
    ) -> dfx.fem.function.Function:
        """
        enables the transfer of data between subdomains in a domain decomposition context,
        specifically for restricting data from a larger subdomain to a smaller subdomain while preserving
        the correct ordering based on local-to-global index mappings (l2gmap).

                Args:
                    func (dfx.fem.function.Function): Function on Ov Subdomain

                Returns:
                    dfx.fem.function.Function: Function on the Non-Ov Subdomain
        """
        # Perform the restriction
        local_func = dfx.fem.Function(self.nonov_V)
        local_func.x.array[:] = func.x.petsc_vec[:][self.mapping][self.indices_non_ov]
        return local_func

    def restrict_from_pred_to_ov_SD(self, func: dfx.fem.function.Function
    ) -> dfx.fem.function.Function:
        ov_func = dfx.fem.Function(self.V)
        ov_func.x.array[self.pred_dofs_in_ov_id_ov] = func.x.array[self.pred_dofs_in_ov_id_pred]
        return ov_func

    def prediction_dofs_around_interface(self, layers: int) -> np.ndarray:
        """Return prediction-space DoFs within ``layers`` cells of the interface."""
        if layers < 1:
            raise ValueError("layers must be at least 1")
        if layers > self.prediction_ell:
            raise ValueError(
                f"layers={layers} exceeds prediction_ell={self.prediction_ell}"
            )

        cell_index_map = self.pred_mesh.topology.index_map(
            self.pred_mesh.topology.dim
        )
        num_cells = cell_index_map.size_local + cell_index_map.num_ghosts
        cell_dofs = [
            self.pred_V.dofmap.cell_dofs(cell) for cell in range(num_cells)
        ]
        reference = np.asarray(self.pred_dofs_local, dtype=np.int32)
        for _ in range(layers):
            touching_cells = [
                dofs for dofs in cell_dofs if np.intersect1d(dofs, reference).size
            ]
            if not touching_cells:
                return np.array([], dtype=np.int32)
            reference = np.unique(np.concatenate(touching_cells)).astype(np.int32)
        return reference

    def prediction_cutoff(self, inner_layers: int) -> dfx.fem.Function:
        """Construct the binary test-localization cutoff on the prediction mesh."""
        cutoff = dfx.fem.Function(self.pred_V)
        inner_dofs = self.prediction_dofs_around_interface(inner_layers)
        cutoff.x.array[inner_dofs] = 1.0
        return cutoff

    def merge_prediction_state(
        self,
        destination: dfx.fem.Function,
        synchronized_ov: dfx.fem.Function,
        received_prediction: dfx.fem.Function,
    ) -> None:
        """Complete a received prediction view with synchronized overlapping data."""
        destination.x.array[:] = received_prediction.x.array
        destination.x.array[self.pred_dofs_in_ov_id_pred] = synchronized_ov.x.array[
            self.pred_dofs_in_ov_id_ov
        ]

    def translate_communication_arrays(self):
        logger.debug("Translating communication arrays to local indexing")

        tt_0 = time()
        for key, value in self.communication_incoming.items():
            self.communication_incoming[key] = [
                self.global_dof_to_local_ov_dof(value[0]),
                self.global_dof_to_local_pred_dof(value[1]),
                self.global_dof_to_local_ov_dof(value[2]),
            ]
        for key, value in self.communication_outgoing.items():
            self.communication_outgoing[key] = [
                self.global_dof_to_local_nonov_dof(value[0]),
                self.global_dof_to_local_nonov_dof(value[1]),
                self.global_dof_to_local_nonov_dof(value[2]),
            ]
        tt_1 = time()

        logger.debug(f"Translation finished after {tt_1-tt_0} seconds")

    def derive_averaging_weights(self):
        logger.debug("deriving averaging weights")
        nonov_one = dfx.fem.Function(self.nonov_V)
        nonov_one.x.array[:] = 1.0
        counter = dfx.fem.Function(self.V)
        counter.x.array[:] = 0.0
        status = MPI.Status()
        for i, comm_round in enumerate(self.comm_schedule):
            tag = i
            for rank_tuple in comm_round:
                # first smaller rank sends, bigger rank receives
                # then smaller rank receives, bigger rank sends
                # take func and send all
                if self.rank == max(rank_tuple):
                    partner_rank = int(min(rank_tuple))
                    # message 1: partner_rank -> self.rank
                    len_message1 = len(self.communication_incoming[partner_rank][2])
                    recv_buf = np.empty((len_message1), dtype=np.float64)
                    request = self.communicator.Irecv(
                        (recv_buf, len_message1, MPI.DOUBLE), partner_rank, tag
                    )
                    request.Wait(status)
                    # message 2: self.rank -> partner_rank
                    len_message2 = len(self.communication_outgoing[partner_rank][2])
                    message2 = nonov_one.x.array[
                        self.communication_outgoing[partner_rank][2]
                    ]
                    self.communicator.Send(
                        (message2, len_message2, MPI.DOUBLE), partner_rank, tag
                    )
                elif self.rank == min(rank_tuple):
                    partner_rank = int(max(rank_tuple))
                    # message 1: self.rank -> partner_rank
                    len_message1 = len(self.communication_outgoing[partner_rank][2])
                    message1 = nonov_one.x.array[
                        self.communication_outgoing[partner_rank][2]
                    ]
                    self.communicator.Send(
                        (message1, len_message1, MPI.DOUBLE), partner_rank, tag
                    )
                    # message 2:
                    len_message2 = len(self.communication_incoming[partner_rank][2])
                    recv_buf = np.empty((len_message2), dtype=np.float64)
                    request = self.communicator.Irecv(
                        (recv_buf, len_message2, MPI.DOUBLE), partner_rank, tag
                    )
                    request.Wait(status)

                else:
                    pass
                if self.rank in rank_tuple:
                    # add interface values to nonov_interface
                    counter.x.array[
                        self.communication_incoming[partner_rank][2]
                    ] += recv_buf[:]
            self.communicator.Barrier()
        # add one to the counter (own contribution)
        counter.x.array[self.nonov_interface_dofs_ov_indexing] += 1.0
        logger.debug(
            f"derived weights for {len(self.nonov_interface_dofs_ov_indexing)} interface DoFs"
        )
        return np.reciprocal(counter.x.array[self.nonov_interface_dofs_ov_indexing])

    def communicate_and_sync(
        self, overlapping_func: dfx.fem.Function
    ) -> Tuple[dfx.fem.Function]:
        nonov_interface = dfx.fem.Function(self.V)
        nonov_interface.x.array[self.nonov_interface_dofs_ov_indexing] = (
            overlapping_func.x.array[self.nonov_interface_dofs_ov_indexing]
        )
        updated_pred_func = dfx.fem.Function(self.pred_V)
        updated_ov_func = dfx.fem.Function(self.V)
        updated_ov_func.x.array[:] = overlapping_func.x.array[:]
        status = MPI.Status()
        non_overlapping_func = self.restrict_from_ov_to_non_ov_SD(overlapping_func)
        for i, comm_round in enumerate(self.comm_schedule):
            tag = i
            for rank_tuple in comm_round:
                # first smaller rank sends, bigger rank receives
                # then smaller rank receives, bigger rank sends
                # take func and send all
                if self.rank == max(rank_tuple):
                    partner_rank = min(rank_tuple)
                    # message 1: partner_rank -> self.rank
                    lens_in = [
                        len(m) for m in self.communication_incoming[partner_rank]
                    ]
                    len_message1 = sum(lens_in)
                    recv_buf = np.empty((len_message1), dtype=np.float64)
                    request = self.communicator.Irecv(
                        (recv_buf, len_message1, MPI.DOUBLE), partner_rank, tag
                    )
                    request.Wait(status)
                    # message 2: self.rank -> partner_rank
                    lens_out = [
                        len(m) for m in self.communication_outgoing[partner_rank]
                    ]
                    len_message2 = sum(lens_out)
                    message2 = np.concatenate(
                        (
                            non_overlapping_func.x.array[
                                self.communication_outgoing[partner_rank][0]
                            ],
                            non_overlapping_func.x.array[
                                self.communication_outgoing[partner_rank][1]
                            ],
                            non_overlapping_func.x.array[
                                self.communication_outgoing[partner_rank][2]
                            ],
                        ),
                        dtype=np.float64,
                    )
                    self.communicator.Send(
                        (message2, len_message2, MPI.DOUBLE), partner_rank, tag
                    )
                elif self.rank == min(rank_tuple):
                    partner_rank = max(rank_tuple)
                    # message 1: self.rank -> partner_rank
                    lens_out = [
                        len(m) for m in self.communication_outgoing[partner_rank]
                    ]
                    len_message1 = sum(lens_out)
                    message1 = np.concatenate(
                        (
                            non_overlapping_func.x.array[
                                self.communication_outgoing[partner_rank][0]
                            ],
                            non_overlapping_func.x.array[
                                self.communication_outgoing[partner_rank][1]
                            ],
                            non_overlapping_func.x.array[
                                self.communication_outgoing[partner_rank][2]
                            ],
                        ),
                        dtype=np.float64,
                    )
                    self.communicator.Send(
                        (message1, len_message1, MPI.DOUBLE), partner_rank, tag
                    )
                    # message 2:
                    lens_in = [
                        len(m) for m in self.communication_incoming[partner_rank]
                    ]
                    len_message2 = sum(lens_in)
                    recv_buf = np.empty((len_message2), dtype=np.float64)
                    request = self.communicator.Irecv(
                        (recv_buf, len_message2, MPI.DOUBLE), partner_rank, tag
                    )
                    request.Wait(status)

                else:
                    pass
                if self.rank in rank_tuple:
                    # update interior values
                    updated_ov_func.x.array[
                        self.communication_incoming[partner_rank][0]
                    ] = recv_buf[: lens_in[0]]
                    # update prediction values
                    updated_pred_func.x.array[
                        self.communication_incoming[partner_rank][1]
                    ] = recv_buf[lens_in[0] : lens_in[0] + lens_in[1]]
                    # add interface values to nonov_interface
                    nonov_interface.x.array[
                        self.communication_incoming[partner_rank][2]
                    ] += recv_buf[lens_in[0] + lens_in[1] :]
            self.communicator.Barrier()
        # scale interface values by weights
        updated_ov_func.x.array[self.nonov_interface_dofs_ov_indexing] = np.multiply(
            nonov_interface.x.array[self.nonov_interface_dofs_ov_indexing],
            self.avg_weights,
        )
        # return updated funcs
        return (updated_ov_func, updated_pred_func)

    def _prepare_comm_buffers(self) -> None:
        self._send_bufs: Dict[int, np.ndarray] = {}
        self._recv_bufs: Dict[int, np.ndarray] = {}
        for rank, (out_interior, out_pred, out_interface) in self.communication_outgoing.items():
            n = len(out_interior) + len(out_pred) + len(out_interface)
            self._send_bufs[rank] = np.zeros(n, dtype=np.float64)
        for rank, (in_interior, in_pred, in_interface) in self.communication_incoming.items():
            n = len(in_interior) + len(in_pred) + len(in_interface)
            self._recv_bufs[rank] = np.zeros(n, dtype=np.float64)

    def _prepare_allocation_free_comm(self) -> None:
        # a) precompute direct OV-index gather arrays per partner, bypassing restrict_from_ov_to_non_ov_SD
        self._ov_send_indices: Dict[int, np.ndarray] = {}
        for rank, (out_interior, out_pred, out_interface) in self.communication_outgoing.items():
            flat = np.concatenate([out_interior, out_pred, out_interface])
            self._ov_send_indices[rank] = self.mapping[self.indices_non_ov[flat]]

        # b) persistent MPI requests — post receives before sends (rendezvous protocol)
        self._persistent_send_reqs = [
            self.communicator.Send_init(buf, dest=rank, tag=0)
            for rank, buf in self._send_bufs.items()
        ]
        self._persistent_recv_reqs = [
            self.communicator.Recv_init(buf, source=rank, tag=0)
            for rank, buf in self._recv_bufs.items()
        ]

        # c) double-buffered output functions — two slots because the time integrator
        #    calls communicate_and_sync twice per step and reads both results after both calls
        self._alloc_free_ov_func = [dfx.fem.Function(self.V), dfx.fem.Function(self.V)]
        self._alloc_free_pred_func = [dfx.fem.Function(self.pred_V), dfx.fem.Function(self.pred_V)]
        self._alloc_free_nonov_interface_func = [dfx.fem.Function(self.V), dfx.fem.Function(self.V)]
        self._alloc_free_slot: int = 0

        # cache per-partner receive split points (sizes of interior and pred parts)
        self._recv_n: Dict[int, Tuple[int, int]] = {
            rank: (len(in_interior), len(in_pred))
            for rank, (in_interior, in_pred, _) in self.communication_incoming.items()
        }

    def free_persistent_requests(self) -> None:
        for req in self._persistent_send_reqs + self._persistent_recv_reqs:
            req.Free()
        self._persistent_send_reqs = []
        self._persistent_recv_reqs = []

    def communicate_and_sync_new_nonblocking(
        self, overlapping_func: dfx.fem.Function
    ) -> Tuple[dfx.fem.Function]:
        """Intermediate variant: non-persistent Isend/Irecv, no barriers, allocates per call.
        Useful for isolating barrier overhead vs allocation overhead."""
        updated_pred_func = dfx.fem.Function(self.pred_V)
        updated_ov_func = dfx.fem.Function(self.V)
        updated_ov_func.x.array[:] = overlapping_func.x.array[:]
        nonov_interface = dfx.fem.Function(self.V)
        nonov_interface.x.array[self.nonov_interface_dofs_ov_indexing] = (
            overlapping_func.x.array[self.nonov_interface_dofs_ov_indexing]
        )

        non_overlapping_func = self.restrict_from_ov_to_non_ov_SD(overlapping_func)

        requests = []
        for rank, (out_interior, out_pred, out_interface) in self.communication_outgoing.items():
            send_buf = self._send_bufs[rank]
            n0 = len(out_interior)
            n1 = len(out_pred)
            send_buf[:n0] = non_overlapping_func.x.array[out_interior]
            send_buf[n0:n0 + n1] = non_overlapping_func.x.array[out_pred]
            send_buf[n0 + n1:] = non_overlapping_func.x.array[out_interface]
            requests.append(self.communicator.Irecv(self._recv_bufs[rank], source=rank, tag=0))
            requests.append(self.communicator.Isend(send_buf, dest=rank, tag=0))

        MPI.Request.Waitall(requests)

        for rank, (in_interior, in_pred, in_interface) in self.communication_incoming.items():
            recv_buf = self._recv_bufs[rank]
            n0, n1 = self._recv_n[rank]
            updated_ov_func.x.array[in_interior] = recv_buf[:n0]
            updated_pred_func.x.array[in_pred] = recv_buf[n0:n0 + n1]
            nonov_interface.x.array[in_interface] += recv_buf[n0 + n1:]

        updated_ov_func.x.array[self.nonov_interface_dofs_ov_indexing] = np.multiply(
            nonov_interface.x.array[self.nonov_interface_dofs_ov_indexing],
            self.avg_weights,
        )
        return (updated_ov_func, updated_pred_func)

    def communicate_and_sync_allocation_free(
        self, overlapping_func: dfx.fem.Function
    ) -> Tuple[dfx.fem.Function]:
        """Production-optimised variant: persistent MPI requests, no allocations per call,
        double-buffered outputs, direct OV-array gather into send buffers."""
        slot = self._alloc_free_slot
        self._alloc_free_slot = 1 - slot
        ov_func = self._alloc_free_ov_func[slot]
        pred_func = self._alloc_free_pred_func[slot]
        nonov_interface = self._alloc_free_nonov_interface_func[slot]

        ov_func.x.array[:] = overlapping_func.x.array[:]
        pred_func.x.array[:] = 0.0
        nonov_interface.x.array[:] = 0.0
        nonov_interface.x.array[self.nonov_interface_dofs_ov_indexing] = (
            overlapping_func.x.array[self.nonov_interface_dofs_ov_indexing]
        )

        # single-step gather directly from ov array into send buffers
        for rank, buf in self._send_bufs.items():
            buf[:] = overlapping_func.x.array[self._ov_send_indices[rank]]

        # start receives before sends, then wait for all
        for req in self._persistent_recv_reqs + self._persistent_send_reqs:
            req.Start()
        MPI.Request.Waitall(self._persistent_recv_reqs + self._persistent_send_reqs)

        for rank, (in_interior, in_pred, in_interface) in self.communication_incoming.items():
            recv_buf = self._recv_bufs[rank]
            n0, n1 = self._recv_n[rank]
            ov_func.x.array[in_interior] = recv_buf[:n0]
            pred_func.x.array[in_pred] = recv_buf[n0:n0 + n1]
            nonov_interface.x.array[in_interface] += recv_buf[n0 + n1:]

        ov_func.x.array[self.nonov_interface_dofs_ov_indexing] = np.multiply(
            nonov_interface.x.array[self.nonov_interface_dofs_ov_indexing],
            self.avg_weights,
        )
        return (ov_func, pred_func)

    def reconstruct_global_approximation_on_root(
        self, comm_world, global_Domain: Domain, ov_subdomain_func: dfx.fem.Function
    ) -> dfx.fem.Function:
        from mpi4py import MPI

        subdomain_rank = comm_world.Get_rank()

        non_ov_func = self.restrict_from_ov_to_non_ov_SD(ov_subdomain_func)

        # local send buffers
        indices = np.asarray(self.nonov_l2gmap, dtype=np.int32)  # send ints
        elements = np.asarray(
            non_ov_func.x.array, dtype=np.float32
        )  # send floats (float32)

        # root gathers sizes
        local_count = len(indices)
        gathered_counts = comm_world.gather(
            local_count, root=0
        )  # list on root, None on non-root

        if subdomain_rank == 0:
            sendcounts = np.array(gathered_counts, dtype=np.int32)
            total = int(sendcounts.sum())
            # displacements: starting index for each rank in the receive buffer
            displs = np.empty_like(sendcounts)
            displs[0] = 0
            if len(sendcounts) > 1:
                displs[1:] = np.cumsum(sendcounts)[:-1]
            # allocate receive buffers
            all_indices = np.empty((total,), dtype=np.int32)
            all_elements = np.empty((total,), dtype=np.float32)
        else:
            sendcounts = None
            displs = None
            all_indices = None
            all_elements = None

        # perform Gatherv for indices (sendbuf first). Non-root passes None for recvbuf.
        comm_world.Gatherv(
            [indices, MPI.INT32_T],
            [all_indices, sendcounts, displs, MPI.INT32_T] if subdomain_rank == 0 else None,
            root=0,
        )

        # perform Gatherv for elements
        comm_world.Gatherv(
            [elements, MPI.FLOAT32_T],
            (
                [all_elements, sendcounts, displs, MPI.FLOAT32_T]
                if subdomain_rank == 0
                else None
            ),
            root=0,
        )

        global_func = None
        if subdomain_rank == 0:
            # split the flat arrays by sendcounts to recover each rank's contribution
            split_by = np.cumsum(sendcounts)[:-1].tolist()  # boundaries
            splitted_indices = np.split(all_indices, split_by)
            splitted_elements = np.split(all_elements, split_by)

            global_func = dfx.fem.Function(global_Domain.V)
            for i, idx_arr in enumerate(splitted_indices):
                global_func.x.array[idx_arr] = splitted_elements[i]

        return global_func


if __name__ == "__main__":
    from Plotting import plot_sol_pyvista

    comm_world = MPI.COMM_WORLD
    subdomain_rank = comm_world.Get_rank()
    N_subdomains = comm_world.Get_size()
    comm_self = MPI.COMM_SELF

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

    partition_of_unity_test = True

    Omega_i_delta = SubDomain(
        f"meshes/Omega_{subdomain_rank}^delta.hdf5", comm_world, comm_self
    )
    ovfunc = dfx.fem.Function(Omega_i_delta.V)
    if partition_of_unity_test:
        ovfunc.x.array[:] = 1.0 
    else:
        ovfunc.x.array[:] = subdomain_rank * 1.0
    if subdomain_rank == 0:
        logger.info("Start one communication cycle")
        tcomm_0 = time()
    new_ov_func, new_pred_func = Omega_i_delta.communicate_and_sync(ovfunc)
    if subdomain_rank == 0:
        tcomm_1 = time()
        logger.info(f"One communication cycle took {tcomm_1-tcomm_0} seconds")

    filename = Path(f'plot/comm_cycle_ov_{subdomain_rank}.png')
    plot_sol_pyvista(Omega_i_delta.V, new_ov_func, filename, clim=(0,N_subdomains))

    filename = Path(f"plot/comm_cycle_pred_{subdomain_rank}.png")
    plot_sol_pyvista(Omega_i_delta.pred_V, new_pred_func, filename, clim=(0, N_subdomains))

    Omega = None
    if subdomain_rank == 0:
        Omega = Domain("meshes/Omega.hdf5", comm_self)
    comm_world.Barrier()

    global_func = Omega_i_delta.reconstruct_global_approximation_on_root(comm_world, Omega, ovfunc)

    if subdomain_rank == 0: 
        print(global_func.x.array)
        filename = Path(f'plot/globalReconstruction.png')
        plot_sol_pyvista(Omega.V, global_func, filename, clim=(0,N_subdomains))
