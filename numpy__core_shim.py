# numpy__core_shim.py
import sys, types
import numpy as _np
import numpy.core.multiarray as _real_multi

# 1) Create a fake 'numpy._core' package
pkg = types.ModuleType("numpy._core")
# trick Python into thinking it's a package
pkg.__path__ = []  

# 2) Attach the real multiarray module into it
pkg.multiarray = _real_multi

# 3) Register both modules in sys.modules
sys.modules["numpy._core"] = pkg
sys.modules["numpy._core.multiarray"] = _real_multi
