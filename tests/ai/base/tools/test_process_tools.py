import time

from haintech.ai.tools.background_process import BackgroundProcess
from haintech.ai.tools.process_tools import (
    create_background_process,
    start_background_process,
    stop_background_process,
    get_background_process_output,
    is_background_process_running,
)


def test_create_background_process():
    # Given: A command to run
    command = ["echo", "test"]
    # When: Creating a background process
    process = create_background_process(command)
    # Then: The process is created correctly
    assert isinstance(process, BackgroundProcess)
    assert process.command == command


def test_start_stop_background_process():
    # Given: A command
    command = ["sleep", "0.1"]
    # When: Starting the process
    pid = start_background_process(command)
    time.sleep(0.05)  # Ensure it starts
    assert is_background_process_running(pid) is True  # Use PID
    # And: Stop it
    stop_background_process(pid)  # Use PID
    # Then: The process is stopped
    assert is_background_process_running(pid) is False  # Use PID


def test_get_background_process_output():
    # Given: A command that echoes "test"
    command = "echo \"test\""
    # When: Starting the process
    pid = start_background_process(command)
    time.sleep(0.1)  # enough time to finish
    output = get_background_process_output(pid) # Use PID
    # Then: The output is correct
    assert output["stdout"] == "test\n"
    assert output["stderr"] == ""
    stop_background_process(pid)




def test_is_background_process_running():
    # Given: A command that sleeps
    command = ["sleep", "0.5"]
    # When/Then: Checking running state before starting
    process = create_background_process(command) # This part doesn't start, so no PID yet
    assert is_background_process_running(process.process) is False # Check via process object as not started
    # Start and check
    pid = start_background_process(command) # Now we have a PID
    time.sleep(0.1)
    assert is_background_process_running(pid) is True # Use PID
    #Stop and check again
    stop_background_process(pid)  # Use PID
    assert is_background_process_running(pid) is False  # Use PID

