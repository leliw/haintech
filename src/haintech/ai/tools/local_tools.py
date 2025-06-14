import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, TypeAlias

StrPath: TypeAlias = str | Path

_log = logging.getLogger(__name__)

run_bash_command_definition = {
    "name": "run_bash_command",
    "description": "Run a bash command and return the output",
    "parameters": {
        "type": "object",
        "required": ["command"],
        "properties": {"command": {"type": "string", "description": "The bash command to run"}},
        "additionalProperties": False,
    },
    "strict": True,
}


def run_bash_command(command: str, cwd: Optional[str] = None) -> Dict[str, str]:
    """
    Run a bash command and return the output

    Args:
        command: The bash command to run
        cwd: The working directory to run the command in
    Returns:
        out: The output of the command
        err: The error of the command
    """
    if cwd and cwd.startswith("/"):
        cwd = cwd[1:] or None
    _log.debug(f"Running bash command: {command}")
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    _log.debug(f"Bash command output: {out.decode()}")
    if err:
        _log.error(f"Bash command error: {err.decode()}")
    return {"out": out.decode(), "err": err.decode()}


save_text_to_file_definition = {
    "name": "save_text_to_file",
    "description": "Save text content to a file",
    "parameters": {
        "type": "object",
        "required": ["file_path", "content", "parent_dir"],
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file",
            },
            "content": {
                "type": "string",
                "description": "The text content to save",
            },
            "parent_dir": {
                "type": "string",
                "description": "The parent directory of the file",
            },
        },
        "additionalProperties": False,
    },
    "strict": True,
}


def save_text_to_file(file_path: StrPath, text_content: str, parent_dir: Optional[StrPath] = None):
    """Saves the provided text content to a specified file path. This function writes the complete text content to the file, creating the file if it doesn't exist or overwriting it if it does exist.

    Args:
        file_path: The full path where the file should be saved (e.g., '/path/to/file.txt', 'documents/notes.txt')
        text_content: The complete text content that will be written to the file. This parameter is mandatory and contains all the text data to be saved.
        parent_dir: Optional: The parent directory path where the file should be created. If not provided, the directory will be inferred from file_path.
    """
    _log.debug(f"Saving text content to file: {file_path}")
    if parent_dir and isinstance(parent_dir, str):
        if parent_dir.startswith("/"):
            parent_dir = parent_dir[1:]
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        file_path_abs = os.path.abspath(file_path)
        parent_dir_abs = os.path.abspath(parent_dir)
        if file_path_abs.startswith(parent_dir_abs):
            file_path = Path(file_path)
        else:
            file_path = os.path.join(parent_dir, file_path)
    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    with open(file_path, "wt") as file:
        file.write(text_content)


load_text_from_file_definition = {
    "name": "load_text_from_file",
    "description": "Load text content from a file",
    "parameters": {
        "type": "object",
        "required": ["file_path", "parent_dir"],
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file",
            },
            "parent_dir": {
                "type": "string",
                "description": "The parent directory of the file",
            },
        },
        "additionalProperties": False,
    },
    "strict": True,
}


def load_text_from_file(file_path: StrPath, parent_dir: Optional[StrPath] = None) -> str:
    """
    Load text content from a file

    Args:
        file_path: The path to the file
        parent_dir: The parent directory of the file
    Returns:
        content: The text content of the file
    """
    _log.debug(f"Loading text content from file: {file_path}")
    if parent_dir and isinstance(parent_dir, str):
        if parent_dir.startswith("/"):
            parent_dir = parent_dir[1:]
    if parent_dir:
        file_path_abs = os.path.abspath(file_path)
        parent_dir_abs = os.path.abspath(parent_dir)
        if file_path_abs.startswith(parent_dir_abs):
            file_path = Path(file_path)
        else:
            file_path = os.path.join(parent_dir, file_path)
    try:
        with open(file_path, "rt") as file:
            content = file.read()
    except FileNotFoundError:
        _log.error(f"File not found: {Path(os.getcwd()) / file_path}")
        content = ""
    return content


def get_directory_structure(dir: StrPath) -> Dict[str, List[str]]:
    """
    Get the directory structure of a directory

    Args:
        rootdir: The root directory
    Returns:
        dir_structure: The directory structure where the keys are the directories and the values are the files in the directory
    """
    _log.debug(f"Getting directory structure of: {dir}")
    dir_structure = {}
    dir = Path(dir)
    for dirpath, _, filenames in os.walk(dir):
        relative_path = os.path.relpath(dirpath, dir)
        if relative_path == ".":
            relative_path = ""
        dir_structure[relative_path] = filenames
    return dir_structure


def get_git_tracked_files(rootdir: Path) -> set[Path]:
    """
    Get a set of all files tracked by git in the repository.

    Args:
        rootdir: The root directory of the git repository.

    Returns:
        A set of paths relative to the rootdir that are tracked by git.
    """
    try:
        git_files = subprocess.check_output(["git", "ls-files"], cwd=str(rootdir), text=True)
        tracked_files = set(Path(line) for line in git_files.splitlines())
        return tracked_files
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to list git tracked files: {e}")


get_git_directory_structure_definition = {
    "name": "get_git_directory_structure",
    "description": "Get the directory structure of a git repository",
    "parameters": {
        "type": "object",
        "required": ["rootdir"],
        "properties": {
            "rootdir": {
                "type": "string",
                "description": "The root directory from which to obtain the directory structure",
            }
        },
        "additionalProperties": False,
    },
    "strict": True,
}


def get_git_directory_structure(rootdir: StrPath) -> Dict[str, List[str]]:
    """
    Get the directory structure of a git repository filtered by git tracked files.

    Args:
        rootdir: The root directory of the git repository.

    Returns:
        dir_structure: The directory structure where the keys are directories and the values are lists of git-tracked files.
    """
    _log.debug(f"Getting directory structure of: {rootdir}")
    dir_structure = {"": []}
    rootdir = Path(rootdir).resolve()
    # Get the set of git-tracked files
    tracked_files = get_git_tracked_files(rootdir)
    # Iterate over tracked files and construct the directory structure
    for file in tracked_files:
        file_directory = "" if str(file.parent) == "." else str(file.parent)
        dir_structure.setdefault(file_directory, []).append(file.name)
    return dir_structure


def get_directory_structure_formated(
    path: str = ".", exclude_dirs: Optional[List[str]] = None, max_depth: int = 1
) -> str:
    """
    Displays the structure of the current directory and its subdirectories up to a specified depth,
    excluding specified directories.

    Args:
        path (str): The path to the directory to explore. Defaults to the current directory (".").
        exclude_dirs (list of str, optional): A list of subdirectory names to exclude. Defaults to None.
        max_depth (int, optional): The maximum depth to explore subdirectories. Defaults to 1.
    """
    path = path or "."
    max_depth = int(max_depth)
    if exclude_dirs is None:
        exclude_dirs = []

    # Helper function to determine if an item is a directory or a file
    def is_dir(item):
        return os.path.isdir(os.path.join(path, item))

    # Recursive helper function to format the directory structure
    def ident(lines: str) -> str:
        ret = ""
        for line in lines.split("\n"):
            if line:
                ret += "|   " + line + "\n"
        return ret

    try:
        items = os.listdir(path)
    except FileNotFoundError:
        return f"Error: Directory '{path}' not found."

    # Sort items, putting directories first, then files
    items.sort(key=lambda item: (not is_dir(item), item.lower()))

    ret = ""
    for item in items:
        if item.startswith("."):
            continue

        full_item_path = os.path.join(path, item)

        if is_dir(item):
            ret += f"|-- {item}/\n"  # Print directory name with '/' to indicate it is a directory

            if item not in exclude_dirs and max_depth >= 1:
                try:
                    ret += ident(
                        get_directory_structure_formated(
                            full_item_path,
                            exclude_dirs=exclude_dirs,
                            max_depth=max_depth - 1,
                        )
                    )
                except PermissionError:
                    ret += "|   |-- (Permission Denied)\n"
        else:
            ret += f"|-- {item}\n"
    return ret
