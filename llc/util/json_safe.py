import json
import math
from dataclasses import asdict, is_dataclass

def _is_primitive(x):
    return isinstance(x, (str, int, float, bool)) or x is None

def json_safe(obj, *, _depth=0, _max_depth=5):
    """Best-effort converter so results are JSON-serializable.
    - Dataclasses -> dict
    - Numpy/JAX arrays -> {"shape": ..., "dtype": ..., "repr": "..."}
    - Exceptions -> {"type": "...", "message": "..."}
    - Fallback -> repr(obj)
    """
    if _depth > _max_depth:
        return f"<truncated depth={_max_depth}>"
    if _is_primitive(obj):
        # Normalize NaN/inf so json.dumps won't choke in strict contexts
        if isinstance(obj, float):
            if math.isnan(obj): return "NaN"
            if math.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if is_dataclass(obj):
        return json_safe(asdict(obj), _depth=_depth+1)
    if isinstance(obj, dict):
        return {str(k): json_safe(v, _depth=_depth+1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, _depth=_depth+1) for v in obj]
    # Numpy / JAX arrays without importing heavy deps
    mod = type(obj).__module__
    if any(m in mod for m in ("numpy", "jax", "jaxlib")) and hasattr(obj, "shape"):
        dtype = getattr(obj, "dtype", None)
        try:
            shape = tuple(getattr(obj, "shape", ()))
        except Exception:
            shape = "?"
        return {"__array__": True, "shape": shape, "dtype": str(dtype), "repr": repr(obj)}
    if isinstance(obj, BaseException):
        return {"__exc__": True, "type": type(obj).__name__, "message": str(obj)}
    # Fallback
    try:
        json.dumps(obj)  # will succeed?
        return obj
    except Exception:
        return repr(obj)