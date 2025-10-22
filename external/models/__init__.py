def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


from .gfcf import GFCF

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .lightgcn import LightGCN
        from .bism import BISM
        from .fpsr import FPSR
        from .fpsr_plus import FPSRplus
        from .fpsr_plus_f import FPSRplusF
