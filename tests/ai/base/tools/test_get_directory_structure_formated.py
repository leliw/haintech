import os
import pytest
from haintech.ai.tools.local_tools import get_directory_structure_formated

# Przygotuj tymczasowy katalog do testów
@pytest.fixture(scope="module")
def temp_dir(tmpdir_factory):
    temp_directory = tmpdir_factory.mktemp("test_dir")
    os.chdir(str(temp_directory))

    # Utwórz strukturę katalogów i plików
    os.makedirs("subdir1")
    os.makedirs("subdir2")
    with open("file1.txt", "w") as f:
        f.write("Zawartość pliku 1")
    with open("subdir1/file2.txt", "w") as f:
        f.write("Zawartość pliku 2")
    with open("subdir2/file3.txt", "w") as f:
        f.write("Zawartość pliku 3")
    return temp_directory


def test_get_directory_structure_formated_basic(temp_dir):
    expected_output = """|-- subdir1/
|   |-- file2.txt
|-- subdir2/
|   |-- file3.txt
|-- file1.txt
"""
    assert get_directory_structure_formated() == expected_output


def test_get_directory_structure_formated_exclude(temp_dir):
    expected_output = """|-- subdir1/
|-- subdir2/
|   |-- file3.txt
|-- file1.txt
"""
    assert get_directory_structure_formated(exclude_dirs=["subdir1"]) == expected_output



def test_get_directory_structure_formated_nonexistent_path(temp_dir):

    expected_output = "Error: Directory 'nonexistent_path' not found."
    assert get_directory_structure_formated("nonexistent_path") == expected_output

# Dodatkowe testy można dodać dla przypadków takich jak:
# - Brak dostępu do podkatalogu (PermissionError)
# - Katalog z ukrytymi plikami/katalogami
# - Różne poziomy zagłębień katalogów

