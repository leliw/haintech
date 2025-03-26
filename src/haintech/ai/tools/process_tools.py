import shlex
from typing import List, Optional, Dict

from haintech.ai.tools.background_process import BackgroundProcess

_processes: Dict[int, BackgroundProcess] = {}


def create_background_process(
    command: List[str] | str, cwd: Optional[str] = None
) -> BackgroundProcess:
    """Creates a BackgroundProcess instance and stores it by PID.

    Args:
        command: The command to execute, either a string or a list of strings.
        cwd: The working directory.

    Returns:
        A BackgroundProcess instance.
    """
    if isinstance(command, str):
        command = shlex.split(command)  # Split the command string
    process = BackgroundProcess(command, cwd)
    return process


def start_background_process(command: List[str] | str, cwd: Optional[str] = None) -> int:
    """Creates and starts a BackgroundProcess, storing it by PID.

    Args:
        command: The command to execute, either a string or a list of strings.
        cwd: The working directory.

    Returns:
        The PID of the started BackgroundProcess instance.
    """
    process = create_background_process(command, cwd)
    process.start()
    _processes[process.process.pid] = process  # Store process by PID
    return process.process.pid


def stop_background_process(pid: int) -> None:
    """Stops a BackgroundProcess by PID.

    Args:
        pid: The PID of the BackgroundProcess to stop.
    """
    pid = int(pid)
    if pid in _processes.keys():
        process = _processes[pid]
        process.stop()
        del _processes[pid]  # Remove the process from the dictionary
    else:
        raise ValueError(f"Process with PID {pid} not found.")


def get_background_process_output(pid: int) -> Dict[str, str]:
    """Gets the output of a BackgroundProcess by PID.

    Args:
        pid: The PID of the BackgroundProcess.

    Returns:
        A dictionary containing "stdout" and "stderr".
    """
    if pid in _processes:
        process = _processes[pid]
        stdout, stderr = process.get_output()
        return {"stdout": stdout, "stderr": stderr}
    else:
        raise ValueError(f"Process with PID {pid} not found.")


def is_background_process_running(pid: int) -> bool:
    """Checks if a BackgroundProcess is running by PID.

    Args:
        pid: The PID of the BackgroundProcess.

    Returns:
        True if the process is running, False otherwise.
    """
    if pid in _processes:
        process = _processes[pid]
        return process.is_running()
    else:
        return False
