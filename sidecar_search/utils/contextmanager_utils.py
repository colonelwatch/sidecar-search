from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from typing import Generator, Iterable


@contextmanager
def del_on_exc(path: Path | Iterable[Path]) -> Generator[None, None, None]:
    paths = [path] if isinstance(path, Path) else path
    try:
        yield
    except (KeyboardInterrupt, Exception):
        for p in paths:
            if not p.exists():
                continue
            if p.is_dir():
                rmtree(p)
            else:
                p.unlink()
        raise
