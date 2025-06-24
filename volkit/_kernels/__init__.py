# volkit/_kernels/__init__.py
from __future__ import annotations
from importlib import import_module
from .routine import Routine

_ROUTINES: dict[str, Routine] = {}

def _register(mod: str) -> None:
    full_name = f"{__name__}.{mod}"          # volkit._kernels.garch11_normal
    routine_mod = import_module(full_name)
    _ROUTINES[routine_mod.ROUTINE.uid] = routine_mod.ROUTINE

_register("garch11_normal")
# _register("garch_pq_normal")
# ...

def get_routine(uid: str) -> Routine:
    try:
        return _ROUTINES[uid]
    except KeyError as err:
        raise RuntimeError(f"No native routine for spec '{uid}'") from err
    
# if __name__ == "__main__":
#     # For debugging purposes, print all registered routines
#     print("Available routines:")
#     for uid, routine in _ROUTINES.items():
#         print(f" - {uid}: {routine}")
