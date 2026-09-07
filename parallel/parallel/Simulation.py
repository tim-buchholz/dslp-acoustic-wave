from mpi4py import MPI
from pathlib import Path
from Norms import error_norm_ref
import logging
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

def simulation_summary(
    simulation_time: float,
    global_num_dofs: int,
    num_time_steps: int,
    N_procs: int,
    num_components: int = 2,
):
    ### calculation of metrics
    num_dofs_total = global_num_dofs * num_components
    throughput = num_dofs_total * num_time_steps / simulation_time
    throughput_per_proc = throughput / N_procs
    wall_time_per_step = simulation_time / num_time_steps
    wall_time = simulation_time
    ### output of metrics and summary
    logger.info("\n")
    logger.info("===>>> Simulation summary <<<===")
    logger.info(f">>> wall time: {wall_time} seconds <<<")
    logger.info(f">>> N procs: {N_procs} <<<")
    logger.info(f">>> number of components: {num_components} <<<")
    logger.info(f">>> number of DoF's per component (global): {global_num_dofs} <<<")
    logger.info(f">>> total number of DoF's (global): {num_dofs_total} <<<")
    logger.info(f">>> time steps: {num_time_steps} <<<")
    logger.info(f"=== Performance metrics ===")
    logger.info(f">>> throughput: {throughput} DoF's / second <<<")
    logger.info(f">>> throughput per proc: {throughput_per_proc} DoF's / second <<<")
    logger.info(f">>> wall time per time step: {wall_time_per_step} seconds <<<")

