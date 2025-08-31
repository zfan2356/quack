import os
import pathlib
from functools import partial

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


def load_cubin_module_data_patched(cubin_data, filepath):
    path = pathlib.Path(filepath)
    path.write_bytes(cubin_data)
    return load_cubin_module_data_og(cubin_data)


def cute_compile_patched(*args, **kwargs):
    """A patched version of cute.compile that dump the SASS to a file if CUTE_CUBIN_PATH is set."""
    if os.getenv("CUTE_CUBIN_PATH") is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = partial(
            load_cubin_module_data_patched, filepath=os.getenv("CUTE_CUBIN_PATH")
        )
    output = cute_compile_og(*args, **kwargs)
    if os.getenv("CUTE_CUBIN_PATH") is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = load_cubin_module_data_og
        if extract is not None:
            cubin_path = pathlib.Path(os.getenv("CUTE_CUBIN_PATH"))
            sass = extract(cubin_path, None)
            cubin_path.with_suffix(".annotated.sass").write_text(sass)
    return output


cute.compile = cute_compile_patched
