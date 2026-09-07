import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
sys.path.append("src")

import os
import SplittedGrids1D
import SplittedGrids2D
import TimeIntegration
import TimeIntegrationDG
import DomainSplitting
import SpaceDiscretization
import ProblemDataUFL
import Domain
from Norms import (
    error_norm,
    error_norm_ref,
    error_norm_ref_QuadratureSpaceProjection,
    exact_norm,
    function_norm,
)
import pandas as pd
from mpi4py import MPI
import dolfinx as dfx
from concurrent import futures
from Config import *
from typing import Optional, Tuple
import numpy as np
from MeshIO import (
    read_h5,
    read_hdf5,
    append_hdf5,
    read_adios_function,
    append_adios_function,
)
import time
import shutil

ERROR_FILE_PATH = "results"
XDMF_FILE_PATH = "results/xdmf/"
VTX_FILE_PATH = "results/vtx/"
HDF5_FILE_PATH = "results/hdf5/"
ADIOS_FILE_PATH = "results/adios/"
DATABASE = "results/database.csv"
COMBINEDTC_PATH = "results/combinedTimeConvergence/"
LOCALTC_PATH = "results/localTimeConvergence/"
SC_PATH = "results/SpaceConvergence/"
TC_PATH = "results/TimeConvergence/"
NUM_MAX_CORES = 16


def check_folder(path: str):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Created new folder results in {os.getcwd()}")


def check_folders(
    paths=[
        ERROR_FILE_PATH,
        XDMF_FILE_PATH,
        HDF5_FILE_PATH,
        ADIOS_FILE_PATH,
        COMBINEDTC_PATH,
        LOCALTC_PATH,
        SC_PATH,
        TC_PATH,
    ]
):
    for path in paths:
        check_folder(path)


def rreplace(s: str, old: str, new: str, occurrence: int = -1):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def move_files_to_directory(files: list[str], directory: str):
    for file in files:
        nodes = file.split("/")
        newfile = "/".join(nodes[:-1]) + "/" + directory + "/" + nodes[-1]
        try:
            shutil.move(file, newfile)
        except FileNotFoundError as e:
            print(e)


def copy_files_to_directory(files: list[str], directory: str):
    for file in files:
        nodes = file.split("/")
        newfile = "/".join(nodes[:-1]) + "/" + directory + "/" + nodes[-1]
        try:
            shutil.copy(file, newfile)
        except FileNotFoundError as e:
            print(e)


def generate_identifier(
    time_integrator: TimeIntegration.TimeIntegratorWave, flags: dict = {}
) -> str:
    identifier = time_integrator.__repr__()
    for key in flags:
        identifier += "::" + str(key) + "=" + str(flags[key])
    return identifier


def parse_identifier(identifier: str) -> dict:
    info = identifier.split("::")
    if len(info[0].split("=")) > 1:
        start_at = 0
    else:
        start_at = 1
    output = {}
    for entry in info[start_at:]:
        key, value = entry.split("=")
        output[key] = value
    return output


def save_error_single_file(identifier: str, error_data: dict) -> str:
    csvfile = ERROR_FILE_PATH + identifier + ".csv"
    info_dict = parse_identifier(identifier)
    for key, value in info_dict.items():
        error_data[key] = value
    out = pd.DataFrame(data=error_data)
    out.to_csv(csvfile)
    return csvfile


def fit_params_to_entity(param_list, entity=1.0):
    return [entity / round(entity / param) for param in param_list]


def save_error_to_database():
    csvfile = DATABASE
    raise NotImplementedError


def save_series(list_of_identifiers: list[str], list_of_error_data: list[dict]) -> str:
    assert len(list_of_identifiers) > 1
    common_attributes = parse_identifier(list_of_identifiers[0])
    for i, identifier in enumerate(list_of_identifiers):
        for key, value in parse_identifier(identifier).items():
            list_of_error_data[i][key] = value
            if key in common_attributes:
                if value != common_attributes[key]:
                    del common_attributes[key]
    csvfile = "results/series"
    for key, value in common_attributes.items():
        csvfile += "::" + f"{key}={value}"
    csvfile += ".csv"
    out = pd.DataFrame(data=list_of_error_data)
    out.to_csv(csvfile)
    return csvfile


def save_xdmf(
    time_integrator: TimeIntegration.TimeIntegratorWave,
    uh: dfx.fem.function.Function,
    short_name: str,
    flags: dict = {},
) -> str:
    identifier = generate_identifier(time_integrator, flags=flags)
    filename = XDMF_FILE_PATH + short_name + "::" + identifier
    time_integrator.save_xdmf(uh, filename)
    return filename


def save_vtx(
    time_integrator: TimeIntegration.TimeIntegratorWave,
    uh: dfx.fem.function.Function,
    short_name: str,
    flags: dict = {},
    t: float = 0.0,
) -> str:
    identifier = generate_identifier(time_integrator, flags=flags)
    filename = VTX_FILE_PATH + short_name + "::" + identifier
    time_integrator.save_vtk(uh, filename, t)
    return filename


def save_hdf5(
    time_integrator: TimeIntegration.TimeIntegratorWave,
    uh: dfx.fem.function.Function,
    short_name: str,
    flags: dict = {},
    t: float = 0.0,
):
    identifier = generate_identifier(time_integrator, flags=flags)
    filename = HDF5_FILE_PATH + short_name + "::" + identifier
    time_integrator.save_hdf5(uh, time_integrator.V, filename, t)
    return filename
    # can be read by
    # FS, f = read_hdf5("test", t=1.0, reconstruct_FunctionSpace=True)


def save_adios(
    time_integrator: TimeIntegration.TimeIntegratorWave,
    uh: dfx.fem.function.Function,
    short_name: str,
    flags: dict = {},
    t: float = 0.0,
):
    identifier = generate_identifier(time_integrator, flags=flags)
    filename = ADIOS_FILE_PATH + short_name + "::" + identifier + ".bp"
    time_integrator.save_adios(uh, time_integrator.V, filename, t)
    return filename


def _reference_extension(ref_mode: str) -> str:
    if ref_mode == "adios":
        return ".bp"
    if ref_mode in ["xdmf", "hdf5"]:
        return ".h5"
    raise ValueError(f"Unknown reference mode {ref_mode}")


def _replace_time_integrator_in_identifier(identifier: str, replacement: str) -> str:
    current = parse_identifier(identifier)["TimeIntegrator"]
    return identifier.replace(
        f"TimeIntegrator={current}", f"TimeIntegrator={replacement}", 1
    )


def _safe_relative_error(error: float | None, norm: float | None) -> float | None:
    if error is None or norm is None:
        return None
    if np.isclose(norm, 0.0):
        return None
    return error / norm


def _square_or_zero(value: float | None) -> float:
    return 0.0 if value is None else value**2


def _add_relative_exact_errors(
    errors: dict,
    V: dfx.fem.function.FunctionSpace,
    problem: ProblemDataUFL.ProblemData,
    T: float,
    prefix: str = "",
):
    q_H10_norm = exact_norm(V, problem.u_exakt(T), "H10")
    q_L2_norm = exact_norm(V, problem.u_exakt(T), "L2")
    p_L2_norm = (
        None if problem.v_exakt is None else exact_norm(V, problem.v_exakt(T), "L2")
    )
    p_Hm1_norm = (
        None
        if problem.v_exakt is None
        else exact_norm(V, problem.v_exakt(T), "Hminus1")
    )
    errors[prefix + REL_ERROR_PREFIX + H1ERROR_Q_STR] = _safe_relative_error(
        errors[prefix + H1ERROR_Q_STR], q_H10_norm
    )
    errors[prefix + REL_ERROR_PREFIX + L2ERROR_Q_STR] = _safe_relative_error(
        errors[prefix + L2ERROR_Q_STR], q_L2_norm
    )
    errors[prefix + REL_ERROR_PREFIX + L2ERROR_P_STR] = _safe_relative_error(
        errors[prefix + L2ERROR_P_STR], p_L2_norm
    )
    errors[prefix + REL_ERROR_PREFIX + HMERROR_P_STR] = _safe_relative_error(
        errors[prefix + HMERROR_P_STR], p_Hm1_norm
    )
    errors[prefix + REL_H1L2_ERROR_STR] = _safe_relative_error(
        np.sqrt(
            _square_or_zero(errors[prefix + H1ERROR_Q_STR])
            + _square_or_zero(errors[prefix + L2ERROR_P_STR])
        ),
        np.sqrt(_square_or_zero(q_H10_norm) + _square_or_zero(p_L2_norm)),
    )
    errors[prefix + REL_L2HM1_ERROR_STR] = _safe_relative_error(
        np.sqrt(
            _square_or_zero(errors[prefix + L2ERROR_Q_STR])
            + _square_or_zero(errors[prefix + HMERROR_P_STR])
        ),
        np.sqrt(_square_or_zero(q_L2_norm) + _square_or_zero(p_Hm1_norm)),
    )


def _add_relative_reference_errors(
    errors: dict,
    qref: dfx.fem.function.Function,
    pref: dfx.fem.function.Function,
):
    q_H10_norm = function_norm(qref, "H10")
    q_L2_norm = function_norm(qref, "L2")
    p_L2_norm = function_norm(pref, "L2")
    p_Hm1_norm = function_norm(pref, "Hminus1")
    errors[REL_ERROR_PREFIX + H1ERROR_Q_STR] = _safe_relative_error(
        errors[H1ERROR_Q_STR], q_H10_norm
    )
    errors[REL_ERROR_PREFIX + L2ERROR_Q_STR] = _safe_relative_error(
        errors[L2ERROR_Q_STR], q_L2_norm
    )
    errors[REL_ERROR_PREFIX + L2ERROR_P_STR] = _safe_relative_error(
        errors[L2ERROR_P_STR], p_L2_norm
    )
    errors[REL_ERROR_PREFIX + HMERROR_P_STR] = _safe_relative_error(
        errors[HMERROR_P_STR], p_Hm1_norm
    )
    errors[REL_H1L2_ERROR_STR] = _safe_relative_error(
        np.sqrt(errors[H1ERROR_Q_STR] ** 2 + errors[L2ERROR_P_STR] ** 2),
        np.sqrt(q_H10_norm**2 + p_L2_norm**2),
    )
    errors[REL_L2HM1_ERROR_STR] = _safe_relative_error(
        np.sqrt(errors[L2ERROR_Q_STR] ** 2 + errors[HMERROR_P_STR] ** 2),
        np.sqrt(q_L2_norm**2 + p_Hm1_norm**2),
    )


def run(
    time_integrator: TimeIntegration.TimeIntegratorWave,
    save_approximation: bool = False,
    flags: dict = {},
    save_mode: str = "adios",
    initialization: bool = True,
) -> dict:
    if initialization:
        time_integrator.__init__(
            time_integrator.T,
            time_integrator.tau,
            time_integrator.Omega,
            time_integrator.SpaceDiscretizationClass,
            time_integrator.problem,
        )
    t0 = time.time()
    qn, pn = time_integrator.integrate(**flags)
    t1 = time.time()
    problem = time_integrator.problem
    T = time_integrator.T
    errors = {}
    errors[H1ERROR_Q_STR] = error_norm(qn, problem.u_exakt(T), "H10")
    errors[L2ERROR_Q_STR] = error_norm(qn, problem.u_exakt(T), "L2")
    errors[L2ERROR_P_STR] = (
        None if problem.v_exakt is None else error_norm(pn, problem.v_exakt(T), "L2")
    )
    errors[HMERROR_P_STR] = (
        None
        if problem.v_exakt is None
        else error_norm(pn, problem.v_exakt(T), "Hminus1")
    )
    _add_relative_exact_errors(errors, time_integrator.V, problem, T)
    errors[TIME_STR] = t1 - t0

    if save_approximation:
        if save_mode == "xdmf":
            save_xdmf(time_integrator, qn, "qn", flags)
            save_xdmf(time_integrator, pn, "pn", flags)
        elif save_mode == "hdf5":
            save_hdf5(time_integrator, qn, "qn", flags, t=T)
            save_hdf5(time_integrator, pn, "pn", flags, t=T)
        elif save_mode == "adios":
            save_adios(time_integrator, qn, "qn", flags, t=T)
            save_adios(time_integrator, pn, "pn", flags, t=T)
        else:
            raise ValueError(f"Unknown save mode {save_mode}")
    return errors


def run_against_reference(
    time_integrator: TimeIntegration.TimeIntegratorWave,
    reference: str,
    save_approximation: bool = False,
    flags: dict = {},
    ref_mode: str = "adios",
    save_mode: str = "adios",
    compare_via_quad_space: bool = False,
    initialization: bool = True,
):
    if initialization:
        time_integrator.__init__(
            time_integrator.T,
            time_integrator.tau,
            time_integrator.Omega,
            time_integrator.SpaceDiscretizationClass,
            time_integrator.problem,
        )
    t0 = time.time()
    qn, pn = time_integrator.integrate(**flags)
    t1 = time.time()
    problem = time_integrator.problem
    T = time_integrator.T
    V = time_integrator.V
    if ref_mode == "xdmf":
        qvalues, _, _ = read_h5(XDMF_FILE_PATH + "qn::" + reference)
        pvalues, _, _ = read_h5(XDMF_FILE_PATH + "pn::" + reference)
        qref = dfx.fem.Function(V)
        qref.x.petsc_vec[:] = qvalues
        pref = dfx.fem.Function(V)
        pref.x.petsc_vec[:] = pvalues
        FSq = V
        FSp = V
    elif ref_mode == "hdf5":
        FSq, qvalues = read_hdf5(
            filename=HDF5_FILE_PATH + "qn::" + reference,
            t=T,
            reconstruct_FunctionSpace=True,
        )
        qref = dfx.fem.Function(FSq)
        qref.x.array[:] = qvalues
        FSp, pvalues = read_hdf5(
            filename=HDF5_FILE_PATH + "pn::" + reference,
            t=T,
            reconstruct_FunctionSpace=True,
        )
        pref = dfx.fem.Function(FSp)
        pref.x.array[:] = pvalues
    elif ref_mode == "adios":
        FSq, qvalues = read_adios_function(
            filename=ADIOS_FILE_PATH + "qn::" + reference,
            t=T,
            reconstruct_FunctionSpace=True,
        )
        qref = dfx.fem.Function(FSq)
        qref.x.array[:] = qvalues
        FSp, pvalues = read_adios_function(
            filename=ADIOS_FILE_PATH + "pn::" + reference,
            t=T,
            reconstruct_FunctionSpace=True,
        )
        pref = dfx.fem.Function(FSp)
        pref.x.array[:] = pvalues
    else:
        raise ValueError(
            "Unknown reference mode. Cannot read file. Note that there is no Reader for VTX files yet."
        )

    errors = {}
    if compare_via_quad_space:
        errors[H1ERROR_Q_STR] = error_norm_ref_QuadratureSpaceProjection(
            qn, qref, FSq, "H10"
        )
        errors[L2ERROR_Q_STR] = error_norm_ref_QuadratureSpaceProjection(
            qn, qref, FSq, "L2"
        )
        errors[L2ERROR_P_STR] = error_norm_ref_QuadratureSpaceProjection(
            pn, pref, FSp, "L2"
        )
        errors[HMERROR_P_STR] = error_norm_ref_QuadratureSpaceProjection(
            pn, pref, FSp, "Hminus1"
        )

    else:
        errors[H1ERROR_Q_STR] = error_norm_ref(qn, qref, "H10")
        errors[L2ERROR_Q_STR] = error_norm_ref(qn, qref, "L2")
        errors[L2ERROR_P_STR] = error_norm_ref(pn, pref, "L2")
        errors[HMERROR_P_STR] = error_norm_ref(pn, pref, "Hminus1")
    if problem.u_exakt is not None:
        _add_relative_exact_errors(errors, time_integrator.V, problem, T)
    else:
        _add_relative_reference_errors(errors, qref, pref)
    errors[TIME_STR] = t1 - t0

    if save_approximation:
        if save_mode == "xdmf":
            save_xdmf(time_integrator, qn, "qn", flags)
            save_xdmf(time_integrator, pn, "pn", flags)
        elif save_mode == "hdf5":
            save_hdf5(time_integrator, qn, "qn", flags, t=T)
            save_hdf5(time_integrator, pn, "pn", flags, t=T)
        elif save_mode == "adios":
            save_adios(time_integrator, qn, "qn", flags, t=T)
            save_adios(time_integrator, pn, "pn", flags, t=T)
        else:
            raise ValueError(f"Unknown save mode {save_mode}")

    return errors


def run_side_by_side(
    time_integrator: (
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ),
    reference_time_integrator: (
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ),
    flags: dict = {},
    flags_ref: dict = {},
    save_approximation: bool = False,
    save_mode: str = "adios",
):
    time_integrator.__init__(
        time_integrator.T,
        time_integrator.tau,
        time_integrator.Omega,
        time_integrator.SpaceDiscretizationClass,
        time_integrator.problem,
    )
    reference_time_integrator.__init__(
        reference_time_integrator.T,
        reference_time_integrator.tau,
        reference_time_integrator.Omega,
        reference_time_integrator.SpaceDiscretizationClass,
        reference_time_integrator.problem,
    )
    t0 = time.time()
    qn, pn = time_integrator.integrate(**flags)
    t1 = time.time()

    t0_ref = time.time()
    qref, pref = reference_time_integrator.integrate(**flags_ref)
    t1_ref = time.time()
    errors = {}

    problem = time_integrator.problem
    T = time_integrator.T
    ## erro data

    errors[H1ERROR_Q_STR] = error_norm(qn, problem.u_exakt(T), "H10")
    errors[L2ERROR_Q_STR] = error_norm(qn, problem.u_exakt(T), "L2")
    errors[L2ERROR_P_STR] = (
        None if problem.v_exakt is None else error_norm(pn, problem.v_exakt(T), "L2")
    )
    errors[HMERROR_P_STR] = (
        None
        if problem.v_exakt is None
        else error_norm(pn, problem.v_exakt(T), "Hminus1")
    )
    _add_relative_exact_errors(errors, time_integrator.V, problem, T)
    errors[TIME_STR] = t1 - t0

    ## error data reference
    errors["Competitor"] = type(reference_time_integrator).__name__
    errors["Reference_" + H1ERROR_Q_STR] = error_norm(qref, problem.u_exakt(T), "H10")
    errors["Reference_" + L2ERROR_Q_STR] = error_norm(qref, problem.u_exakt(T), "L2")
    errors["Reference_" + L2ERROR_P_STR] = (
        None if problem.v_exakt is None else error_norm(pref, problem.v_exakt(T), "L2")
    )
    errors["Reference_" + HMERROR_P_STR] = (
        None
        if problem.v_exakt is None
        else error_norm(pref, problem.v_exakt(T), "Hminus1")
    )
    _add_relative_exact_errors(errors, reference_time_integrator.V, problem, T, "Reference_")
    errors["Reference_" + TIME_STR] = t1_ref - t0_ref

    ## error_data difference
    errors["Difference_" + H1ERROR_Q_STR] = error_norm_ref(qn, qref, "H10")
    errors["Difference_" + L2ERROR_Q_STR] = error_norm_ref(qn, qref, "L2")
    errors["Difference_" + L2ERROR_P_STR] = error_norm_ref(pn, pref, "L2")
    errors["Difference_" + HMERROR_P_STR] = error_norm_ref(pn, pref, "Hminus1")
    _add_relative_exact_errors(errors, time_integrator.V, problem, T, "Difference_")

    if save_approximation:
        if save_mode == "xdmf":
            save_xdmf(time_integrator, qn, "qn", flags)
            save_xdmf(time_integrator, pn, "pn", flags)
        elif save_mode == "hdf5":
            save_hdf5(time_integrator, qn, "qn", flags, t=T)
            save_hdf5(time_integrator, pn, "pn", flags, t=T)
        elif save_mode == "adios":
            save_adios(time_integrator, qn, "qn", flags, t=T)
            save_adios(time_integrator, pn, "pn", flags, t=T)
        else:
            raise ValueError(f"Unknown save mode {save_mode}")

    return errors


def run_series(
    time_integrators: list[TimeIntegration.TimeIntegratorWave],
    save_approximations: bool = False,
    flags: dict = {},
    parallel: bool = False,
    competitor: Optional[str] = None,
    save_mode: str = "adios",
    ref_mode: str = "adios",
):
    print(f"Running series of {len(time_integrators)} simulations...")
    if parallel:
        num_time_integrators = len(time_integrators)
        num_cores = min(NUM_MAX_CORES, os.cpu_count())  # else out of memory error
        num_jobs = 1 if (num_cores is None) else num_cores

        global task

        def task(i):
            error_data = {}
            time_integrator = time_integrators[i]
            identifier = generate_identifier(time_integrator, flags)
            if competitor is None:
                error_data = run(
                    time_integrator,
                    save_approximations,
                    flags,
                    save_mode=save_mode,
                )
            else:
                reference_str = (
                    _replace_time_integrator_in_identifier(identifier, competitor)
                    + _reference_extension(ref_mode)
                )
                error_data = run_against_reference(
                    time_integrator,
                    reference_str,
                    save_approximations,
                    flags,
                    ref_mode=ref_mode,
                    save_mode=save_mode,
                )
                current = parse_identifier(identifier)["TimeIntegrator"]
                identifier = _replace_time_integrator_in_identifier(
                    identifier, current + "VS" + competitor
                )
            print(f"Completed task {i}")
            return (identifier, error_data)

        out_par = [0] * num_time_integrators
        with futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
            future_to_out = {
                executor.submit(task, k): k for k in range(num_time_integrators)
            }
            for future in futures.as_completed(future_to_out):
                k = future_to_out[future]
                out_par[k] = future.result()
        results = list(zip(*out_par))

        return results[0][:], results[1][:]
    else:
        list_of_identifiers = []
        list_of_error_data = []
        for k, time_integrator in enumerate(time_integrators):
            error_data = {}
            identifier = generate_identifier(time_integrator, flags)
            if competitor is None:
                error_data = run(
                    time_integrator,
                    save_approximations,
                    flags,
                    save_mode=save_mode,
                )
            else:
                reference_str = (
                    _replace_time_integrator_in_identifier(identifier, competitor)
                    + _reference_extension(ref_mode)
                )
                error_data = run_against_reference(
                    time_integrator,
                    reference_str,
                    save_approximations,
                    flags,
                    ref_mode=ref_mode,
                    save_mode=save_mode,
                )
                current = parse_identifier(identifier)["TimeIntegrator"]
                identifier = _replace_time_integrator_in_identifier(
                    identifier, current + "VS" + competitor
                )
            list_of_error_data.append(error_data)
            list_of_identifiers.append(identifier)
            print(f"Completed task {k}")

        return list_of_identifiers, list_of_error_data


def run_series_against_reference_solution(
    time_integrators: list[TimeIntegration.TimeIntegratorWave],
    reference_str: str,
    save_approximations: bool = False,
    flags: dict = {},
    parallel: bool = False,
    ref_mode: str = "adios",
    save_mode: str = "adios",
):
    print(f"Running series of {len(time_integrators)} simulations...")
    if parallel:
        num_time_integrators = len(time_integrators)
        num_cores = min(NUM_MAX_CORES, os.cpu_count())  # else out of memory error
        num_jobs = 1 if (num_cores is None) else num_cores

        global task

        def task(i):
            error_data = {}
            time_integrator = time_integrators[i]
            identifier = generate_identifier(time_integrator, flags)
            error_data = run_against_reference(
                time_integrator,
                reference_str,
                save_approximations,
                flags,
                ref_mode=ref_mode,
                save_mode=save_mode,
                compare_via_quad_space=False,
            )
            print(f"Completed task {i}")
            return (identifier, error_data)

        out_par = [0] * num_time_integrators
        with futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
            future_to_out = {
                executor.submit(task, k): k for k in range(num_time_integrators)
            }
            for future in futures.as_completed(future_to_out):
                k = future_to_out[future]
                out_par[k] = future.result()
        results = list(zip(*out_par))

        return results[0][:], results[1][:]
    else:
        list_of_identifiers = []
        list_of_error_data = []
        for k, time_integrator in enumerate(time_integrators):
            error_data = {}
            identifier = generate_identifier(time_integrator, flags)
            error_data = run_against_reference(
                time_integrator,
                reference_str,
                save_approximations,
                flags,
                ref_mode=ref_mode,
                save_mode=save_mode,
                compare_via_quad_space=False,
            )
            list_of_error_data.append(error_data)
            list_of_identifiers.append(identifier)
            print(f"Completed task {k}")

        return list_of_identifiers, list_of_error_data


def run_multiseries(
    list_of_TIseries: list[list[TimeIntegration.TimeIntegratorWave]],
    save_approximations: Optional[list[bool]] = None,
    flags: Optional[list[dict]] = None,
    parallel: bool = False,
    competitors: Optional[list[Optional[str]]] = None,
    save_individual_series: bool = True,
    save_mode: str = "adios",
    ref_mode: str = "adios",
):
    num_series = len(list_of_TIseries)
    print(f"Running a multiseries of {num_series} series")
    if save_approximations is None:
        save_approximations = [False] * num_series
    if flags is None:
        flags = [{}] * num_series
    if competitors is None:
        competitors = [None] * num_series
    if num_series == len(save_approximations) == len(flags) == len(competitors):
        list_of_identifiers = []
        list_of_error_data = []
        for i, TIseries in enumerate(list_of_TIseries):
            idents, errors = run_series(
                TIseries,
                save_approximations[i],
                flags[i],
                parallel,
                competitors[i],
                save_mode=save_mode,
                ref_mode=ref_mode,
            )
            if save_individual_series:
                save_series(idents, errors)
            list_of_identifiers += idents
            list_of_error_data += errors
            if save_individual_series:
                save_series(idents, errors)
            print(f"Completed series {i}")
        return list_of_identifiers, list_of_error_data
    else:
        raise AssertionError("Lengths of ")


def run_series_side_by_side(
    time_integrators: list[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    reference_time_integrators: list[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    save_approximations: bool = False,
    flags: dict = {},
    flags_ref: dict = {},
    parallel: bool = False,
):
    print(f"Running series of {len(time_integrators)} simulations...")
    if parallel:
        num_time_integrators = len(time_integrators)
        num_cores = min(NUM_MAX_CORES, os.cpu_count())  # else out of memory error
        num_jobs = 1 if (num_cores is None) else num_cores

        global task

        def task(i):
            error_data = {}
            time_integrator = time_integrators[i]
            reference_time_integrator = reference_time_integrators[i]
            identifier = generate_identifier(time_integrator, flags)
            error_data = run_side_by_side(
                time_integrator,
                reference_time_integrator,
                flags,
                flags_ref,
                save_approximations,
            )
            print(f"Completed task {i}")
            return (identifier, error_data)

        out_par = [0] * num_time_integrators
        with futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
            future_to_out = {
                executor.submit(task, k): k for k in range(num_time_integrators)
            }
            for future in futures.as_completed(future_to_out):
                k = future_to_out[future]
                out_par[k] = future.result()
        results = list(zip(*out_par))

        return results[0][:], results[1][:]
    else:
        list_of_identifiers = []
        list_of_error_data = []
        for k, time_integrator in enumerate(time_integrators):
            error_data = {}
            identifier = generate_identifier(time_integrator, flags)
            reference_time_integrator = reference_time_integrators[k]
            error_data = run_side_by_side(
                time_integrator,
                reference_time_integrator,
                flags,
                flags_ref,
                save_approximations,
            )
            list_of_error_data.append(error_data)
            list_of_identifiers.append(identifier)
            print(f"Completed task {k}")

        return list_of_identifiers, list_of_error_data


def run_side_by_side_with_separate_initialization(
    problem_no: int,
    a: float | int,
    b: float | int,
    h: float,
    ell: int,
    tau: float,
    T: float | int,
    kappa: float | int,
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    CIclass: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    SGclass: type[Domain.DS_reg],
    flags: dict = {},
    save_approximation: bool = False,
    save_mode: str = "adios",
):
    Omega = SGclass(a, b, h, ell)
    problem = ProblemDataUFL.ProblemData(
        problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
    )
    time_integrator = TIclass(T, tau, Omega, SDclass, problem)
    competitor = CIclass(T, tau, Omega, SDclass, problem)
    identifier = generate_identifier(time_integrator, flags) + "::vsRefsw=CN"
    errors = run_side_by_side(
        time_integrator=time_integrator,
        reference_time_integrator=competitor,
        flags=flags,
        save_approximation=save_approximation,
        save_mode=save_mode,
    )

    return identifier, errors


def run_with_separate_initialization(
    problem_no: int,
    a: float | int,
    b: float | int,
    h: float,
    ell: int,
    tau: float,
    T: float | int,
    kappa: float | int,
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    SGclass: type[Domain.DS_reg],
    flags: dict = {},
    save_approximation: bool = False,
    save_mode: str = "adios",
    reference: Optional[str] = None,
    ref_mode="adios",
):
    Omega = SGclass(a, b, h, ell)
    problem = ProblemDataUFL.ProblemData(
        problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
    )
    time_integrator = TIclass(T, tau, Omega, SDclass, problem)
    if not reference:
        identifier = generate_identifier(time_integrator, flags)
        errors = run(
            time_integrator,
            save_approximation=save_approximation,
            save_mode=save_mode,
            initialization=False,
            flags=flags,
        )
    else:
        identifier = generate_identifier(time_integrator, flags) + "::vsRef=CN"
        errors = run_against_reference(
            time_integrator,
            reference=reference,
            save_approximation=save_approximation,
            flags=flags,
            ref_mode=ref_mode,
            save_mode=save_mode,
            compare_via_quad_space=False,
            initialization=False,
        )

    return identifier, errors


def create_reference_solution(
    problem_no: int,
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    SGclass: type[Domain.DS_reg],
    a: float = 0.0,
    b: float = 1.0,
    h: float = 0.001,
    ell: int = 8,
    kappa: float = 1.0,
    T: float = 1.0,
    tau: float = 1e-3,
    flags: dict = {},
    save_each: int = 0,
    save_mode: str = "adios",
):
    Omega = SGclass(a, b, h, ell)
    problem = ProblemDataUFL.ProblemData(
        problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
    )
    time_integrator = TIclass(T, tau, Omega, SDclass, problem)

    identifier = generate_identifier(time_integrator, flags)
    time_integrator.__init__(
        time_integrator.T,
        time_integrator.tau,
        time_integrator.Omega,
        time_integrator.SpaceDiscretizationClass,
        time_integrator.problem,
    )
    # if time_integrator.V.mesh.comm.rank == 1:

    print("identifier", identifier)
    t0 = time.time()
    tn = 0.0
    qn, pn = time_integrator.startup()
    # if time_integrator.V.mesh.comm.rank == 1:
    if save_mode == "hdf5":
        filename_q = save_hdf5(time_integrator, qn, "qn", flags, t=tn)
        filename_p = save_hdf5(time_integrator, pn, "pn", flags, t=tn)
        append_function = append_hdf5
    elif save_mode == "adios":
        filename_q = save_adios(time_integrator, qn, "qn", flags, t=tn)
        filename_p = save_adios(time_integrator, pn, "pn", flags, t=tn)
        append_function = append_adios_function
    else:
        raise ValueError("Reference solutions can be saved with 'adios' or 'hdf5'")
    print("Assembling space discretization")
    time_integrator.assemble_space_discretization()
    print(f"Calculating reference solution")
    u_D = dfx.fem.Function(time_integrator.V)
    for n in range(time_integrator.start_index, time_integrator.N):
        tn = n * time_integrator.tau

        if time_integrator.problem.u_bc is not None:
            u_D.interpolate(
                time_integrator.problem.dfExpression(
                    time_integrator.problem.u_bc(tn + time_integrator.tau),
                    time_integrator.V,
                )
            )
            time_integrator.bc = time_integrator.Omega.DirichletBC(u_D)

        qn_plus_1, pn_plus_1 = time_integrator.step(
            tn, time_integrator.tau, qn, pn, time_integrator.bc
        )

        qn.x.array[:] = qn_plus_1.x.array
        pn.x.array[:] = pn_plus_1.x.array

        # if time_integrator.V.mesh.comm.rank == 1:
        if save_each > 0 and (n + 1) % save_each == 0 and (n + 1) != time_integrator.N:
            print("saving sample at t=", tn + tau, " ...", end="")
            append_function(filename_q, qn, t=tn + tau)
            append_function(filename_p, pn, t=tn + tau)
            print("done")
    t1 = time.time()

    problem = time_integrator.problem
    T = time_integrator.T
    # if time_integrator.V.mesh.comm.rank == 1:
    print("Finished reference solution")
    print("Total time needed ", t1 - t0, "secs")
    print("saving reference solution to file")
    append_function(filename_q, qn, t=T)
    append_function(filename_p, pn, t=T)
    print("Finished saving")

    return (identifier, qn, pn, time_integrator.V)


def local_time_convergence(
    problem_no: type[ProblemDataUFL.ProblemData],
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[TimeIntegration.TimeIntegratorWave],
    SGclass: type[Domain.DS_reg],
    a: float = 0.0,
    b: float = 1.0,
    h: float = 0.001,
    ell: int = 8,
    kappa: float = 1.0,
    tau_min: float = 1e-3,
    tau_max: float = 1e-1,
    num_taus: int = 20,
    save_approximations: bool = False,
    flags: dict = {},
    parallelize_series: bool = True,
) -> str:
    """Local time convergence experiment

    Args:
        problem_no (type[ProblemDataUFL.ProblemData]): Number of problem in ProblemDataUFL
        SDclass (type[SpaceDiscretization.SpaceDiscretization]): SpaceDiscretization class
        TIclass (type[TimeIntegration.TimeIntegratorWave]): TimeIntegration class
        SGclass (type[Domain.DS_reg]): DS_reg class
        a (float, optional): first domain parameter. Defaults to 0.0.
        b (float, optional): second domain parameter. Defaults to 1.0.
        h (float, optional): space discretization width. Defaults to 0.001.
        ell (int, optional): overlap parameter for domain decomposition. Defaults to 8.
        kappa (float, optional): wave propagation speed. Defaults to 1.0.
        tau_min (float, optional): minmal time step size. Defaults to 1e-3.
        tau_max (float, optional): maximal time step size. Defaults to 1e-1.
        num_taus (int, optional): Number of different timesteps. Defaults to 20.
        save_approximations (bool, optional): Declares whether approximations at T are saved. Defaults to False.
        flags (dict, optional): additional flags. Defaults to {}.
        parallelize_series (bool, optional): Declares whether series is parallelized. Defaults to True.

    Returns:
        str: csvfile where the results are saved to
    """
    check_folders()
    Omega = SGclass(a, b, h, ell)
    problem = ProblemDataUFL.ProblemData(
        problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
    )
    tau_lst = np.round(np.geomspace(tau_max, tau_min, num_taus), decimals=8)
    my_time_integrators = [
        TIclass(tau, tau, Omega, SDclass, problem) for tau in tau_lst
    ]
    identifier, errors = run_series(
        my_time_integrators,
        save_approximations=save_approximations,
        flags=flags,
        parallel=parallelize_series,
    )
    csvfile = save_series(identifier, errors)
    return csvfile


def space_convergence(
    problem_no: type[ProblemDataUFL.ProblemData],
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[TimeIntegration.TimeIntegratorWave],
    SGclass: type[Domain.DS_reg],
    a: float = 0.0,
    b: float = 1.0,
    ell: int = 8,
    kappa: float = 1.0,
    T: float = 1.0,
    tau: float = 0.001,
    h_min: float = 1e-3,
    h_max: float = 1e-1,
    num_hs: int = 20,
    save_approximations: bool = False,
    flags: dict = {},
    parallelize_series: bool = True,
) -> str:
    """Space Convergence experiment

    Args:
        problem_no (type[ProblemDataUFL.ProblemData]): Number of problem in ProblemDataUFL
        SDclass (type[SpaceDiscretization.SpaceDiscretization]): SpaceDiscretization class
        TIclass (type[TimeIntegration.TimeIntegratorWave]): TimeIntegration class
        SGclass (type[Domain.DS_reg]): DS_reg class
        a (float, optional): first domain parameter. Defaults to 0.0.
        b (float, optional): second domain parameter. Defaults to 1.0.
        ell (int, optional): overlap parameter for domain decomposition. Defaults to 8.
        kappa (float, optional): wave propagation speed. Defaults to 1.0.
        T (float,optional): end time. Defaults to 1.0.
        tau (float, optional): time step size. Defaults to 0.001.
        h_min (float, optional): minimal space discretization width passed to SGclass. Defaults to 1e-3.
        h_max (float, optional): maximal space discretization width passed to SGclass. Defaults to 1e-1.
        num_hs (int, optional): number of different space discretization widths. Defaults to 20.
        save_approximations (bool, optional): Declares whether approximations at T are saved. Defaults to False.
        flags (dict, optional): additional flags. Defaults to {}.
        parallelize_series (bool, optional): Declares whether series is parallelized. Defaults to True.

    Returns:
        str: csvfile where the results are saved to
    """
    check_folders()
    h_lst = fit_params_to_entity(np.geomspace(h_max, h_min, num_hs), b - a)
    Omegas = [SGclass(a, b, h, ell) for h in h_lst]
    problems = [
        ProblemDataUFL.ProblemData(
            problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
        )
        for Omega in Omegas
    ]
    my_time_integrators = [
        TIclass(T, tau, Omegas[i], SDclass, problems[i]) for i, h in enumerate(h_lst)
    ]
    identifier, errors = run_series(
        my_time_integrators,
        save_approximations=save_approximations,
        flags=flags,
        parallel=parallelize_series,
    )
    csvfile = save_series(identifier, errors)
    return csvfile


def time_convergence(
    problem_no: type[ProblemDataUFL.ProblemData],
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    SGclass: type[Domain.DS_reg],
    a: float = 0.0,
    b: float = 1.0,
    h: float = 0.001,
    ell: int = 8,
    kappa: float = 1.0,
    T: float = 1.0,
    tau_min: float = 1e-3,
    tau_max: float = 1e-1,
    num_taus: int = 20,
    save_approximations: bool = False,
    flags: dict = {},
    parallelize_series: bool = True,
    reference_str: Optional[str] = None,
    return_identifiers: bool = False,
) -> str:
    """Time convergence experiment

    Args:
        problem_no (type[ProblemDataUFL.ProblemData]): Number of problem in ProblemDataUFL
        SDclass (type[SpaceDiscretization.SpaceDiscretization]): SpaceDiscretization class
        TIclass (type[TimeIntegration.TimeIntegratorWave]): TimeIntegration class
        SGclass (type[Domain.DS_reg]): DS_reg class
        a (float, optional): first domain parameter. Defaults to 0.0.
        b (float, optional): second domain parameter. Defaults to 1.0.
        h (float, optional): space discretization width. Defaults to 0.001.
        ell (int, optional): overlap parameter for domain decomposition. Defaults to 8.
        kappa (float, optional): wave propagation speed. Defaults to 1.0.
        T (float,optional): end time. Defaults to 1.0.
        tau_min (float, optional): minmal time step size. Defaults to 1e-3.
        tau_max (float, optional): maximal time step size. Defaults to 1e-1.
        num_taus (int, optional): Number of different timesteps. Defaults to 20.
        save_approximations (bool, optional): Declares whether approximations at T are saved. Defaults to False.
        flags (dict, optional): additional flags. Defaults to {}.
        parallelize_series (bool, optional): Declares whether series is parallelized. Defaults to True.
        reference_str (str, optional): indicates where to find a reference solution if no exact solution exists

    Returns:
        str: csvfile where the results are saved to
    """
    check_folders()
    Omega = SGclass(a, b, h, ell)
    problem = ProblemDataUFL.ProblemData(
        problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
    )
    tau_lst = fit_params_to_entity(np.geomspace(tau_max, tau_min, num_taus), T)
    my_time_integrators = [TIclass(T, tau, Omega, SDclass, problem) for tau in tau_lst]
    if problem.u_exakt is not None:
        identifier, errors = run_series(
            my_time_integrators,
            save_approximations=save_approximations,
            flags=flags,
            parallel=parallelize_series,
        )
    else:
        identifier, errors = run_series_against_reference_solution(
            my_time_integrators,
            reference_str,
            save_approximations=save_approximations,
            flags=flags,
            parallel=parallelize_series,
        )
    csvfile = save_series(identifier, errors)
    if return_identifiers:
        return csvfile, identifier
    else:
        return csvfile


def time_convergence_batchwise(
    problem_no: type[ProblemDataUFL.ProblemData],
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ],
    SGclass: type[Domain.DS_reg],
    a: float = 0.0,
    b: float = 1.0,
    h: float = 0.001,
    ell: int = 8,
    kappa: float = 1.0,
    T: float = 1.0,
    tau_min: float = 1e-3,
    tau_max: float = 1e-1,
    num_taus: int = 20,
    save_approximations: bool = False,
    flags: dict = {},
    reference_identifiers: Optional[list[str]] = None,
    batchsize: int = 8,
    sidewise_againstCN: bool = False,
) -> str:
    """Time convergence experiment

    Args:
        problem_no (type[ProblemDataUFL.ProblemData]): Number of problem in ProblemDataUFL
        SDclass (type[SpaceDiscretization.SpaceDiscretization]): SpaceDiscretization class
        TIclass (type[TimeIntegration.TimeIntegratorWave]): TimeIntegration class
        SGclass (type[Domain.DS_reg]): DS_reg class
        a (float, optional): first domain parameter. Defaults to 0.0.
        b (float, optional): second domain parameter. Defaults to 1.0.
        h (float, optional): space discretization width. Defaults to 0.001.
        ell (int, optional): overlap parameter for domain decomposition. Defaults to 8.
        kappa (float, optional): wave propagation speed. Defaults to 1.0.
        T (float,optional): end time. Defaults to 1.0.
        tau_min (float, optional): minmal time step size. Defaults to 1e-3.
        tau_max (float, optional): maximal time step size. Defaults to 1e-1.
        num_taus (int, optional): Number of different timesteps. Defaults to 20.
        save_approximations (bool, optional): Declares whether approximations at T are saved. Defaults to False.
        flags (dict, optional): additional flags. Defaults to {}.
        reference_str (str, optional): indicates where to find a reference solution if no exact solution exists
        batchsize (int, defualts=8): indicates in what batches the experiments will be started. 1 means serial, maximum batchsize is min(NUM_MAX_CORES, os.cpu_count())

    Returns:
        str: csvfile where the results are saved to
    """
    if reference_identifiers is not None and sidewise_againstCN:
        raise NotImplementedError(
            "Please decide whether you want to test side by side against CN or use a reference solution!"
        )

    check_folders()
    tau_lst = fit_params_to_entity(np.geomspace(tau_max, tau_min, num_taus), T)

    print(f"Running series of {len(tau_lst)} simulations (started batch-wise)...")
    num_experiments = len(tau_lst)
    num_cores = min(
        batchsize, NUM_MAX_CORES, os.cpu_count()
    )  # else out of memory error
    num_jobs = 1 if (num_cores is None) else num_cores

    global task

    def task(i):
        error_data = {}
        if reference_identifiers is not None:
            reference = reference_identifiers[i].strip("results/xdmf/") + ".h5"
        else:
            reference = reference_identifiers
        if not sidewise_againstCN:
            identifier, error_data = run_with_separate_initialization(
                problem_no=problem_no,
                a=a,
                b=b,
                h=h,
                ell=ell,
                tau=tau_lst[i],
                T=T,
                kappa=kappa,
                SDclass=SDclass,
                TIclass=TIclass,
                SGclass=SGclass,
                flags=flags,
                save_approximation=save_approximations,
                reference=reference,
            )
        else:
            identifier, error_data = run_side_by_side_with_separate_initialization(
                problem_no=problem_no,
                a=a,
                b=b,
                h=h,
                ell=ell,
                tau=tau_lst[i],
                T=T,
                kappa=kappa,
                SDclass=SDclass,
                TIclass=TIclass,
                CIclass=TimeIntegration.CrankNicolson,
                SGclass=SGclass,
                flags=flags,
                save_approximation=save_approximations,
            )
        print(f"Completed task {i}")
        return (identifier, error_data)

    out_par = [0] * num_experiments
    with futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        future_to_out = {executor.submit(task, k): k for k in range(num_experiments)}
        for future in futures.as_completed(future_to_out):
            k = future_to_out[future]
            out_par[k] = future.result()
    results = list(zip(*out_par))

    identifier, errors = results[0][:], results[1][:]
    csvfile = save_series(identifier, errors)
    return csvfile


def time_convergence_combined(
    problem_no: type[ProblemDataUFL.ProblemData],
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    SGclass: type[Domain.DS_reg],
    DSclass: type[DomainSplitting.DomainSplitting] = DomainSplitting.DomainSplitting,
    a: float = 0.0,
    b: float = 1.0,
    h: float = 0.001,
    ell: int = 8,
    kappa: float = 1.0,
    T: float = 1.0,
    tau_min: float = 1e-3,
    tau_max: float = 1e-1,
    num_taus: int = 20,
    implicit_competitor: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ] = TimeIntegration.CrankNicolson,
    explicit_competitor: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ] = TimeIntegration.leapfrog,
    save_approximations: bool = False,
    flags: dict = {},
    DSflags: dict = {},
    parallelize_series: bool = True,
) -> str:
    """Combined Time Convergence Experiment
       Tests for time convergence of CrankNicolson, leapfrog, Domainsplitting against exact solution
             and of DomainSplitting against CrankNicolson

    Args:
        problem_no (type[ProblemDataUFL.ProblemData]): Number of problem in ProblemDataUFL
        SDclass (type[SpaceDiscretization.SpaceDiscretization]): SpaceDiscretization class
        SGclass (type[Domain.DS_reg]): DS_reg class
        a (float, optional): first domain parameter. Defaults to 0.0.
        b (float, optional): second domain parameter. Defaults to 1.0.
        h (float, optional): space discretization width. Defaults to 0.001.
        ell (int, optional): overlap parameter for domain decomposition. Defaults to 8.
        kappa (float, optional): wave propagation speed. Defaults to 1.0.
        T (float,optional): end time. Defaults to 1.0.
        tau_min (float, optional): minmal time step size. Defaults to 1e-3.
        tau_max (float, optional): maximal time step size. Defaults to 1e-1.
        num_taus (int, optional): Number of different timesteps. Defaults to 20.
        save_approximations (bool, optional): Declares whether approximations at T are saved. Defaults to False.
        flags (dict, optional): additional flags. Defaults to {}.
        parallelize_series (bool, optional): Declares whether series is parallelized. Defaults to True.

    Returns:
        str: csvfile where the results are saved to
    """
    check_folders()
    Omega = SGclass(a, b, h, ell)
    problem = ProblemDataUFL.ProblemData(
        problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
    )
    tau_lst = fit_params_to_entity(np.geomspace(tau_max, tau_min, num_taus), T)
    TIclasses = [
        implicit_competitor,
        explicit_competitor,
        DSclass,
        DSclass,
    ]
    if FEM_TYPE == 'DG':
        save_approximations = [False, False, False, False]
    else:
        save_approximations = [True, False, False, False]
    competitors = [None, None, None, implicit_competitor.__name__]  # "CrankNicolson"
    for key, value in flags.items():
        if key in DSflags.keys():
            raise ValueError("Please do not pass flags twice")
        else:
            DSflags[key] = value
    list_flags = [flags, flags, DSflags, DSflags]
    my_time_integrators = [
        [TIclass(T, tau, Omega, SDclass, problem) for tau in tau_lst]
        for TIclass in TIclasses
    ]
    identifier, errors = run_multiseries(
        my_time_integrators,
        save_approximations,
        list_flags,
        parallelize_series,
        competitors,
    )
    csvfile = save_series(identifier, errors)
    return csvfile


def time_convergence_compare_no_save(
    problem_no: type[ProblemDataUFL.ProblemData],
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    SGclass: type[Domain.DS_reg],
    DSclass: type[DomainSplitting.DomainSplitting] = DomainSplitting.DomainSplitting,
    Compet_TIclass: type[
        TimeIntegration.TimeIntegratorWave | TimeIntegrationDG.TimeIntegratorWaveDG
    ] = TimeIntegration.CrankNicolson,
    a: float = 0.0,
    b: float = 1.0,
    h: float = 0.001,
    ell: int = 8,
    kappa: float = 1.0,
    T: float = 1.0,
    tau_min: float = 1e-3,
    tau_max: float = 1e-1,
    num_taus: int = 20,
    DSflags: dict = {},
    REFflags: dict = {},
    parallelize_series: bool = True,
):
    check_folders()
    Omega = SGclass(a, b, h, ell)
    problem = ProblemDataUFL.ProblemData(
        problem_no, Omega.V, wave_propagation_speed=kappa, provide_grad=False
    )
    tau_lst = fit_params_to_entity(np.geomspace(tau_max, tau_min, num_taus), T)
    my_time_integrators = [DSclass(T, tau, Omega, SDclass, problem) for tau in tau_lst]
    ref_time_integrator = [
        Compet_TIclass(T, tau, Omega, SDclass, problem) for tau in tau_lst
    ]
    identifier, errors = run_series_side_by_side(
        my_time_integrators,
        ref_time_integrator,
        False,
        REFflags,
        DSflags,
        parallel=parallelize_series,
    )
    csvfile = save_series(identifier, errors)
    return csvfile


def ell_param_study(
    problem_no: type[ProblemDataUFL.ProblemData],
    SDclass: type[SpaceDiscretization.SpaceDiscretization],
    TIclass: type[TimeIntegration.TimeIntegratorWave],
    SGclass: type[Domain.DS_reg],
    a: float = 0.0,
    b: float = 1.0,
    h: float = 0.001,
    ell_min: int = 1,
    ell_max: int = 20,
    kappa: float = 1.0,
    T: float = 1.0,
    tau_min: float = 1e-3,
    tau_max: float = 1e-1,
    num_taus: int = 20,
    save_approximations: bool = False,
    flags: dict = {},
    batchsize: int = 16,
) -> list[str]:
    """Time convergence experiment for different ell's

    Args:
        problem_no (type[ProblemDataUFL.ProblemData]): Number of problem in ProblemDataUFL
        SDclass (type[SpaceDiscretization.SpaceDiscretization]): SpaceDiscretization class
        TIclass (type[TimeIntegration.TimeIntegratorWave]): TimeIntegration class
        SGclass (type[Domain.DS_reg]): DS_reg class
        a (float, optional): first domain parameter. Defaults to 0.0.
        b (float, optional): second domain parameter. Defaults to 1.0.
        h (float, optional): space discretization width. Defaults to 0.001.
        ell (int, optional): overlap parameter for domain decomposition. Defaults to 8.
        kappa (float, optional): wave propagation speed. Defaults to 1.0.
        T (float,optional): end time. Defaults to 1.0.
        tau_min (float, optional): minmal time step size. Defaults to 1e-3.
        tau_max (float, optional): maximal time step size. Defaults to 1e-1.
        num_taus (int, optional): Number of different timesteps. Defaults to 20.
        save_approximations (bool, optional): Declares whether approximations at T are saved. Defaults to False.
        flags (dict, optional): additional flags. Defaults to {}.
        parallelize_series (bool, optional): Declares whether series is parallelized. Defaults to True.

    Returns:
        str: csvfile where the results are saved to
    """
    check_folders()
    csvfiles = []
    for ell in range(ell_min, ell_max + 1):
        csvfile = time_convergence_batchwise(
            problem_no=problem_no,
            SDclass=SDclass,
            TIclass=TIclass,
            SGclass=SGclass,
            a=a,
            b=b,
            h=h,
            ell=ell,
            kappa=kappa,
            T=T,
            tau_min=tau_min,
            tau_max=tau_max,
            num_taus=num_taus,
            save_approximations=save_approximations,
            flags=flags,
            reference_identifiers=None,
            batchsize=batchsize,
            sidewise_againstCN=False,
        )
        csvfiles.append(csvfile)
    return csvfiles


if __name__ == "__main__":
    csvfile = time_convergence_combined(
        problem_no=9,
        SDclass=SpaceDiscretization.FEM_ml,
        SGclass=SplittedGrids1D.DS_1D_2SD,
        h=0.001,
        tau_max=0.01,
        tau_min=0.001,
        num_taus=10,
    )
