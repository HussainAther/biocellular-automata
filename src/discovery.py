# src/discovery.py

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import List


def auto_import_package(package_name: str) -> List[str]:
    """
    Import all submodules in a package so that side-effect registration happens
    (e.g., modules calling @register()).

    Returns a list of imported module names.
    """
    imported = []

    pkg: ModuleType = importlib.import_module(package_name)

    if not hasattr(pkg, "__path__"):
        # Not a package (single module), nothing to walk.
        return imported

    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__, package_name + "."):
        importlib.import_module(modname)
        imported.append(modname)

        # Optional: recurse into subpackages
        if ispkg:
            imported.extend(auto_import_package(modname))

    return imported

