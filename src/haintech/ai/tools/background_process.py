import subprocess
import threading
import time
import io
from typing import List, Tuple, Optional


class BackgroundProcess:
    """Runs a command in the background and captures its output.

    This class allows you to execute a command as a subprocess, capture its
    stdout and stderr streams, and manage its lifecycle.  It uses threads to
    read the output streams asynchronously, preventing deadlocks.

    Args:
        command: The command to execute as a list of strings.
        cwd: The working directory for the command. Defaults to None.
    """

    def __init__(self, command: List[str], cwd: Optional[str] = None):
        self.command = command
        self.cwd = cwd or None
        self.process: Optional[subprocess.Popen] = None
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._running: bool = False

    def start(self) -> None:
        """Starts the background process."""
        if self._running:
            raise RuntimeError("Process already running")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
        )
        self._running = True
        self._stdout_thread = threading.Thread(
            target=self._read_stream, args=(self.process.stdout, self.stdout_buffer)
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stream, args=(self.process.stderr, self.stderr_buffer)
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stream(self, stream: io.TextIOBase, buffer: io.StringIO) -> None:
        """Reads from a stream and writes to a buffer.

        Args:
            stream: The stream to read from (stdout or stderr).
            buffer: The buffer to write to.
        """
        while self._running:
            line = stream.readline()
            if line:
                buffer.write(line)
            else:
                time.sleep(0.1)
        stream.close()

    def get_output(self) -> Tuple[str, str]:
        """Returns the captured stdout and stderr.

        Returns:
            A tuple containing the stdout and stderr strings.
        """
        return self.stdout_buffer.getvalue(), self.stderr_buffer.getvalue()

    def stop(self) -> None:
        """Stops the background process."""
        if self.process and self._running:
            self._running = False
            self.process.terminate()
            self.process.wait()
            self._stdout_thread.join()
            self._stderr_thread.join()

    def is_running(self) -> bool:
        """Checks if the process is still running.

        Returns:
            True if the process is running, False otherwise.
        """
        if self.process:
            return self.process.poll() is None
        return False

    def __del__(self) -> None:
        """Stops the process when the object is garbage collected."""
        self.stop()

