# src/quantum_hpc/distributed/coordinator.py

import logging
import time
import threading
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

class HPCJobState(Enum):
    """
    Represents the lifecycle state of an HPC quantum job.
    """
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

@dataclass
class HPCJob:
    """
    Specification and status for an HPC quantum job.

    Attributes:
        job_id: Unique identifier for this job.
        qubit_count: Number of qubits for the simulation/task.
        code_distance: Code distance if relevant (e.g., surface code).
        num_cycles: Number of QEC cycles or HPC steps to run.
        parameters: Arbitrary dictionary for additional HPC or simulation details.
        callback: Optional function to call with job results upon completion.
        state: Current lifecycle state (queued, running, completed, etc.).
        start_time: UNIX timestamp when job starts running; None if not started.
        end_time: UNIX timestamp when job ended (completed, failed, or canceled).
        progress: A percentage or iteration count indicating job completion.
        result: Arbitrary dictionary storing final or partial results.
        error: Any error message if the job failed or got canceled.
    """
    job_id: str
    qubit_count: int
    code_distance: int = 3
    num_cycles: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # Lifecycle state
    state: HPCJobState = HPCJobState.QUEUED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: Union[int, float] = 0
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class HPCJobCoordinator:
    """
    Schedules and manages distributed quantum tasks (e.g., large-scale QEC simulations).
    Integrates HPC job lifecycle states (QUEUED, RUNNING, COMPLETED, FAILED, CANCELED).
    In a real system, you'd interface with an HPC scheduler or parallel framework.
    """

    def __init__(self):
        """
        Initialize an HPC job coordinator with in-memory job tracking.
        """
        self.jobs: Dict[str, HPCJob] = {}
        logger.debug("HPCJobCoordinator initialized.")

    def submit_job(self, job: HPCJob) -> str:
        """
        Submit a new HPC job with initial state=QUEUED.

        Args:
            job: HPCJob containing specification and HPC params.

        Returns:
            The job_id as a string.
        """
        if job.job_id in self.jobs:
            raise ValueError(f"Job ID {job.job_id} already exists.")

        job.state = HPCJobState.QUEUED
        job.start_time = None
        job.end_time = None
        job.progress = 0
        job.error = None
        logger.info(f"Submitting HPC job {job.job_id} with {job.qubit_count} qubits, "
                    f"d={job.code_distance}, cycles={job.num_cycles}")

        self.jobs[job.job_id] = job
        # In a real HPC system, you'd dispatch tasks to nodes or queue here.
        # For demonstration, run it locally on a background thread:
        self._run_job_locally(job)

        return job.job_id

    def get_job_status(self, job_id: str) -> Optional[HPCJob]:
        """
        Retrieve status for a specific job, or None if not found.
        """
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Attempt to cancel a running or queued job.

        Returns:
            True if canceled, False otherwise (e.g., already completed/failed/not found).
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Cannot cancel job {job_id}: not found.")
            return False
        if job.state in [HPCJobState.COMPLETED, HPCJobState.FAILED, HPCJobState.CANCELED]:
            logger.warning(f"Cannot cancel job {job_id}: already in {job.state.value} state.")
            return False

        # Mark it canceled
        job.state = HPCJobState.CANCELED
        job.end_time = time.time()
        job.error = "Job canceled by user."
        job.progress = 100
        logger.info(f"Job {job_id} canceled successfully.")
        return True

    def list_jobs(self) -> List[HPCJob]:
        """
        List statuses of all known jobs.
        """
        return list(self.jobs.values())

    def start_job(self, job_id: str) -> bool:
        """
        Manually transition a job from QUEUED to RUNNING if not already running.
        In a real HPC, you might have an automatic or scheduled approach.
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Cannot start job {job_id}: not found.")
            return False
        if job.state != HPCJobState.QUEUED:
            logger.warning(f"Cannot start job {job_id}: state={job.state.value}.")
            return False

        job.state = HPCJobState.RUNNING
        job.start_time = time.time()
        logger.info(f"Job {job_id} is now RUNNING.")
        return True

    def mark_completed(self, job_id: str) -> bool:
        """
        Mark a job as COMPLETED if it's currently RUNNING or QUEUED (e.g., finishing early).
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.state in [HPCJobState.RUNNING, HPCJobState.QUEUED]:
            job.state = HPCJobState.COMPLETED
            job.end_time = time.time()
            job.progress = 100
            logger.info(f"Job {job_id} marked COMPLETED.")
            return True
        return False

    def mark_failed(self, job_id: str, error_msg: str) -> bool:
        """
        Mark a job as FAILED if it's currently RUNNING or QUEUED.
        Provide an error message describing the failure.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.state in [HPCJobState.RUNNING, HPCJobState.QUEUED]:
            job.state = HPCJobState.FAILED
            job.end_time = time.time()
            job.error = error_msg
            logger.error(f"Job {job_id} failed: {error_msg}")
            return True
        return False

    def _run_job_locally(self, job: HPCJob) -> None:
        """
        Demonstration: run the job in a local thread or process.
        For HPC, you'd distribute tasks across nodes or a parallel framework.
        If job is queued, we auto-start it for demonstration; real HPC might differ.
        """
        def job_thread():
            # Move from QUEUED to RUNNING automatically
            if job.state == HPCJobState.QUEUED:
                job.state = HPCJobState.RUNNING
                job.start_time = time.time()
                logger.info(f"Job {job.job_id} auto-started locally.")

            try:
                # Fake progress by stepping through cycles
                for cycle_idx in range(job.num_cycles):
                    # Check for cancellation or error
                    if job.state == HPCJobState.CANCELED:
                        return
                    # Simulate HPC cycle
                    self._simulate_cycle(job, cycle_idx)
                    job.progress = ((cycle_idx + 1) / job.num_cycles) * 100

                # Mark completion if not canceled/failed
                if job.state not in [HPCJobState.CANCELED, HPCJobState.FAILED]:
                    job.state = HPCJobState.COMPLETED
                    job.end_time = time.time()
                    job.progress = 100
                    job.result = {"status": "complete", "job_id": job.job_id}
                    logger.info(f"Job {job.job_id} completed successfully.")

                # Callback if specified
                if job.callback and job.state == HPCJobState.COMPLETED:
                    job.callback(job.result)

            except Exception as e:
                job.state = HPCJobState.FAILED
                job.end_time = time.time()
                job.error = str(e)
                logger.error(f"Error in local run of job {job.job_id}: {e}")

        t = threading.Thread(target=job_thread, daemon=True)
        t.start()

    def _simulate_cycle(self, job: HPCJob, cycle_idx: int) -> None:
        """
        Demonstration method: simulates a single cycle of HPC activity.
        Could represent QEC steps, parallel ops, etc.
        """
        logger.debug(f"Simulating HPC cycle {cycle_idx+1}/{job.num_cycles} for job {job.job_id}.")
        # Emulate compute time
        time.sleep(0.1)
        # In real HPC: distribute tasks, gather partial results, etc.
        pass
