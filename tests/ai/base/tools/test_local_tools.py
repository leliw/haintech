import os
from pathlib import Path
import subprocess

import pytest
from haintech.ai.tools import (
    get_directory_structure,
    get_git_directory_structure,
    run_bash_command,
    save_text_to_file,
    load_text_from_file,
)


def test_run_bash_command_ok():
    # Given: A bash command
    cmd = "echo 'Hello, World!'"
    # When: Running the bash command
    ret = run_bash_command(cmd)
    # Then: The output is correct
    assert ret["out"] == "Hello, World!\n"
    # And: There is no error
    assert ret["err"] == ""


def test_run_bash_command_error():
    # Given: A bash command that produces an error
    cmd = "ls /not_existing_dir"
    # When: Running the bash command
    ret = run_bash_command(cmd)
    # Then: There is no output
    assert ret["out"] == ""
    # And: The error is not empty
    assert ret["err"] != ""


def test_run_bash_command_empty():
    # Given: An empty bash command
    cmd = ""
    # When: Running the bash command
    ret = run_bash_command(cmd)
    # Then: There is no output
    assert ret["out"] == ""
    # And: There is no error
    assert ret["err"] == ""


def test_save_text_to_file_ok(tmp_path):
    # Given: A file path and content
    file_path = tmp_path / "test.txt"
    content = "Hello, World!"
    # When: Saving the content to the file
    save_text_to_file(file_path, content)
    # Then: The file exists
    assert file_path.exists()
    # And: The file content is correct
    with open(file_path, "r") as file:
        assert file.read() == content


def test_save_text_to_file_parent_dir(tmp_path):
    # Given: A file path with a parent directory and content
    parent_dir = tmp_path / "parent"
    file_path = "test.txt"
    content = "Hello, World!"
    # When: Saving the content to the file
    save_text_to_file(file_path, content, parent_dir)
    # Then: The parent directory exists
    assert parent_dir.exists()
    # And: The file exists
    assert (parent_dir / file_path).exists()
    # And: The file content is correct
    with open(parent_dir / file_path, "r") as file:
        assert file.read() == content


@pytest.fixture
def tmp_path_cwd(tmp_path):
    cur_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cur_dir)


def test_save_text_to_file_parent_dir_redundant(tmp_path_cwd: Path):
    # Given: A file path with a parent directory and content
    parent_dir = "parent"
    file_path = "parent/test.txt"
    content = "Hello, World!"
    # When: Saving the content to the file
    save_text_to_file(file_path, content, parent_dir)
    # Then: The parent directory exists
    assert tmp_path_cwd.joinpath(parent_dir).exists()
    # And: The file exists
    assert tmp_path_cwd.joinpath(file_path).exists()
    # And: The file content is correct
    with open(tmp_path_cwd / file_path, "r") as file:
        assert file.read() == content


def test_load_text_from_file_ok(tmp_path):
    # Given: A file path and content
    file_path = tmp_path / "test.txt"
    content = "Hello, World!"
    with open(file_path, "w") as file:
        file.write(content)
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path)
    # Then: The loaded content is correct
    assert loaded_content == content


def test_load_text_from_file_parent_dir(tmp_path):
    # Given: A file path with a parent directory and content
    parent_dir = tmp_path / "parent"
    os.makedirs(parent_dir, exist_ok=True)
    file_path = "test.txt"
    content = "Hello, World!"
    with open(parent_dir / file_path, "w") as file:
        file.write(content)
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path, parent_dir)
    # Then: The loaded content is correct
    assert loaded_content == content


def test_load_text_from_file_parent_dir_redundant(tmp_path_cwd: Path):
    # Given: A file path with a parent directory and content
    parent_dir = "parent"
    os.makedirs(tmp_path_cwd / parent_dir, exist_ok=True)
    file_path = "parent/test.txt"
    content = "Hello, World!"
    with open(tmp_path_cwd / file_path, "w") as file:
        file.write(content)
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path, parent_dir)
    # Then: The loaded content is correct
    assert loaded_content == content

def test_load_text_from_file_not_existing(tmp_path):
    # Given: A file path that does not exist
    file_path = tmp_path / "not_existing.txt"
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path)
    # Then: The loaded content is empty
    assert loaded_content == ""


def test_load_text_from_file_not_existing_parent_dir(tmp_path):
    # Given: A file path with a parent directory that does not exist
    parent_dir = tmp_path / "not_existing"
    file_path = "test.txt"
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path, parent_dir)
    # Then: The loaded content is empty
    assert loaded_content == ""


def test_load_text_from_file_empty(tmp_path):
    # Given: An empty file
    file_path = tmp_path / "empty.txt"
    file_path.touch()
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path)
    # Then: The loaded content is empty
    assert loaded_content == ""


def test_load_text_from_file_empty_parent_dir(tmp_path):
    # Given: A file path with a parent directory that is empty
    parent_dir = tmp_path / "empty"
    os.makedirs(parent_dir, exist_ok=True)
    file_path = "test.txt"
    (parent_dir / file_path).touch()
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path, parent_dir)
    # Then: The loaded content is empty
    assert loaded_content == ""


def test_load_text_from_file_error(tmp_path):
    # Given: A file path that produces an error
    file_path = tmp_path / "not_existing.txt"
    # When: Loading the content from the file
    loaded_content = load_text_from_file(file_path)
    # Then: The loaded content is empty
    assert loaded_content == ""


def test_get_directory_structure(tmp_path):
    # Given: A directory structure
    rootdir = tmp_path / "root"
    os.makedirs(rootdir / "dir1/dir2", exist_ok=True)
    (rootdir / "dir1/file1.txt").touch()
    (rootdir / "dir1/dir2/file2.txt").touch()
    # When: Getting the directory structure
    dir_structure = get_directory_structure(rootdir)
    # Then: The directory structure is correct
    assert dir_structure == {
        "": [],
        "dir1": ["file1.txt"],
        "dir1/dir2": ["file2.txt"],
    }


def test_get_directory_structure_empty(tmp_path):
    # Given: An empty directory
    rootdir = tmp_path / "empty"
    os.makedirs(rootdir, exist_ok=True)
    # When: Getting the directory structure
    dir_structure = get_directory_structure(rootdir)
    # Then: The directory structure is correct
    assert dir_structure == {"": []}


def test_get_git_directory_structure(tmp_path):
    # Given: A directory structure
    rootdir = tmp_path / "root"
    os.makedirs(rootdir / "dir1/dir2", exist_ok=True)
    (rootdir / "dir1/file1.txt").touch()
    (rootdir / "dir1/dir2/file2.txt").touch()
    # And: A git repository
    subprocess.run(["git", "init"], cwd=rootdir)
    subprocess.run(["git", "add", "."], cwd=rootdir)
    # When: Getting the directory structure
    dir_structure = get_git_directory_structure(rootdir)
    # Then: The directory structure is correct
    assert dir_structure == {
        "": [],
        "dir1": ["file1.txt"],
        "dir1/dir2": ["file2.txt"],
    }


def test_get_git_directory_structure_with_git_ignore(tmp_path):
    # Given: A directory structure
    rootdir = tmp_path / "root"
    os.makedirs(rootdir / "dir1/dir2", exist_ok=True)
    (rootdir / "dir1/file1.txt").touch()
    (rootdir / "dir1/dir2/file2.txt").touch()
    # And: A git repository
    subprocess.run(["git", "init"], cwd=rootdir)
    # And: A git ignore file
    with open(rootdir / ".gitignore", "w") as file:
        file.write("dir1/dir2/")
    subprocess.run(["git", "add", "."], cwd=rootdir)
    # When: Getting the directory structure
    dir_structure = get_git_directory_structure(rootdir)
    # Then: The directory structure contains .gitignore
    # And: Does not contain dir1/dir2/file2.txt
    assert dir_structure == {
        "": [".gitignore"],
        "dir1": ["file1.txt"],
    }
