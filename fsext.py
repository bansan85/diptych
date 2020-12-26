import os
from pathlib import Path
import shutil


def del_pattern(folder: str, pattern: str) -> None:
    for path in Path(folder).glob(pattern):
        path.unlink()


def is_file_exists(filename: str) -> bool:
    return os.path.exists(filename)


def copy_file(source: str, destination: str) -> None:
    shutil.copyfile(source, destination)


def get_absolute_from_current_path(module_filename: str, filename: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(module_filename)), filename
    )


def extract_path(full_path: str) -> str:
    return os.path.dirname(os.path.abspath(full_path))


def delete_file(filename: str) -> None:
    os.remove(filename)
