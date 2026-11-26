from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def create_directory(path: PathLike) -> str:
    """Create `path` (and parents) if needed and return its absolute path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def check_existing_file(path: PathLike) -> bool:
    return Path(path).is_file()


def create_path(directory: PathLike, file_name: str) -> str:
    """Create `directory` if needed and return the full path of `file_name` inside it."""
    return str(Path(create_directory(directory)) / file_name)
