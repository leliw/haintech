import logging
import threading
from typing import Optional


class ProgressTracker:
    """Tracks progress of a task with a defined number of steps.
    Can be nested, where completing a child tracker increments its parent.
    """

    _log = logging.getLogger(__name__)

    def __init__(self, total_steps: int = 0, parent: Optional["ProgressTracker"] = None, name: str = None):
        """Initializes the ProgressTracker.

        Args:
            parent: An optional parent ProgressTracker. If provided,
                    this tracker will increment the parent when it completes.
        """
        self.total_steps = total_steps  # Total steps required for completion
        self.completed_steps = 0  # Number of steps completed so far
        self._lock = threading.Lock()  # Lock for thread safety
        self.parent = parent  # Reference to the parent tracker, if any
        self.name = name

    def set_total_steps(self, total_steps: int):
        """Sets the total number of steps required for this tracker.

        Args:
            total_steps: Total number of steps. Must be greater than zero.
        Raises:
            ValueError: If `total_steps` is less than or equal to zero,
                        or if some steps have already been completed.
        """
        if total_steps <= 0:
            raise ValueError("Total steps must be greater than zero.")
        with self._lock:
            if self.completed_steps > 0:
                raise ValueError(
                    f"{self.name} Cannot set total steps after steps have been completed."
                )
            self.completed_steps = 0
            self.total_steps = total_steps

    def reset(self, total_steps: int = None):
        with self._lock:
            self.completed_steps = 0
            if total_steps:
                self.total_steps = total_steps

    def increment(self):
        """Increases the number of completed steps by one.
        If this tracker reaches completion (completed_steps == total_steps)
        and has a parent tracker, it increments the parent tracker.
        """
        if self.total_steps == 0:
            raise ValueError(f"{self.name} Total steps must be set before incrementing.")

        is_complete = False
        with self._lock:
            self.completed_steps += 1
            if self.completed_steps > self.total_steps:
                raise ValueError(f"{self.name} Completed steps cannot exceed total steps.")
            if self.completed_steps == self.total_steps:
                is_complete = True
            current_completed = self.completed_steps
            current_total = self.total_steps

        # --- Lock is released here ---

        self.notify(current_completed, current_total)
        if is_complete and self.parent:
            self._log.debug(f"Tracker complete. Incrementing parent: {self.parent}")
            self.parent.increment()

    def reset(self):
        """Resets the completed steps count to zero.
        Does not change the total steps or the parent tracker.
        """
        with self._lock:
            self.completed_steps = 0
        self._log.info("Progress tracker reset.")
        # Optionally notify about the reset state
        self.notify(0, self.total_steps)


    def get_progress(self) -> float:
        """Returns the current progress as a float value between 0.0 and 1.0.

        Returns:
            Progress value (0.0 to 1.0). Returns 0.0 if total_steps is not set.
        """
        with self._lock:
            total = self.total_steps
            completed = self.completed_steps

        if total == 0:
            return 0.0
        return min(completed, total) / total

    def is_complete(self) -> bool:
        """Checks if the tracker has completed all its steps."""
        with self._lock:
            return self.total_steps > 0 and self.completed_steps == self.total_steps

    def notify(self, completed: int, total: int):
        """Logs the current progress."""
        name = f" {self.name}" if self.name else ""
        if total > 0:
            self._log.info("Progress%s: %d/%d (%.1f%%)", name, completed, total, self.get_progress() * 100)
        else:
             self._log.info("Progress%s: %d/???", name, completed)


    def __repr__(self) -> str:
        """Provides a string representation of the tracker's state."""
        with self._lock:
            return f"<ProgressTracker({self.name}: {self.completed_steps}/{self.total_steps})>"
