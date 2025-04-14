import pytest
import time
import os
from haintech.ai.tools.background_process import BackgroundProcess


def test_background_process_start_stop():
    process = BackgroundProcess(["echo", "Hello"])
    process.start()
    time.sleep(0.5)  # Allow some time for the process to run
    process.stop()
    assert not process.is_running()


def test_background_process_get_output():
    process = BackgroundProcess(["echo", "Hello"])
    process.start()
    time.sleep(0.5)
    process.stop()
    stdout, stderr = process.get_output()
    assert "Hello\n" in stdout
    assert stderr == ""


def test_background_process_cwd():
    # Create a temporary directory
    os.makedirs("temp_dir", exist_ok=True)
    with open("temp_dir/test.txt", "w") as f:
        f.write("Test content")

    process = BackgroundProcess(["ls"], cwd="temp_dir")
    process.start()
    time.sleep(0.5)
    process.stop()
    stdout, _ = process.get_output()
    assert "test.txt" in stdout

    # clean-up temporary directory and its contents.
    os.remove("temp_dir/test.txt")
    os.rmdir("temp_dir")


def test_background_process_empty_cwd():
    process = BackgroundProcess(["echo", "Hello"], cwd="")
    process.start()
    time.sleep(0.5)  # Allow some time for the process to run
    process.stop()
    assert not process.is_running()


def test_background_process_already_running():
    process = BackgroundProcess(["sleep", "1"])
    process.start()
    with pytest.raises(RuntimeError):
        process.start()
    process.stop()


def test_background_process_long_running_command():
    process = BackgroundProcess(["sleep", "0.5"])
    process.start()
    time.sleep(0.2)  # Check before it finishes.
    assert process.is_running()
    time.sleep(0.5)  # Now let it finish.
    assert not process.is_running()
    process.stop()  # Ensure a clean stop even if finished.


def test_background_process_stderr():
    process = BackgroundProcess(["ls", "non_existent_file"])
    process.start()
    time.sleep(0.5)
    process.stop()
    _, stderr = process.get_output()
    assert any(
        [
            c in stderr
            for c in ["No such file or directory", "cannot access", "nie ma dostępu"]
        ]
    )


# def test_background_process_keyboard_interrupt():  # More complex, might require signals
#    pass


def test_background_process_no_command():
    with pytest.raises(
        IndexError
    ):  # subprocess.Popen will raise ValueError if no command given.
        process = BackgroundProcess([])
        process.start()


def test_background_process_invalid_command():
    with pytest.raises(
        FileNotFoundError
    ):  # Or a different exception depending on the OS/command.
        process = BackgroundProcess(["invalid_command"])
        process.start()


def test_background_process_is_not_running():
    process = BackgroundProcess(["echo", "Hello"])
    assert not process.is_running()  # Check before starting

    process.start()
    time.sleep(0.5)
    process.stop()
    assert not process.is_running()  # Check after stopping
