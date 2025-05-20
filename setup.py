import glob
import os
from os.path import abspath

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    cpu_only = os.getenv("CPU_ONLY", "0") == "1"

    use_cuda = (not cpu_only and torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1"

    if cpu_only:
        print(">>> CPU_ONLY=1 → Building without CUDA")
    elif not use_cuda:
        print(">>> CUDA not available → Falling back to CPU-only build")
    else:
        print(">>> Building with CUDA support")

    extra_compile_args = {
        "cxx": ["-fvisibility=hidden", "-fdiagnostics-color=always", "-Wno-deprecated-declarations"],
        "nvcc": ["-Wno-deprecated-declarations"],
    }

    if debug_mode:
        extra_compile_args["cxx"] += ["-O0", "-g", "-lineinfo"]
        extra_compile_args["nvcc"] += ["-O0", "-g", "-lineinfo"]
    else:
        extra_compile_args["cxx"] += ["-O3"]
        extra_compile_args["nvcc"] += ["-O3"]

    include_dirs = [abspath("src/common")]
    ext_modules = []
    if use_cuda:
        ext_modules += [
            CUDAExtension(
                name="difflogic_cuda",
                sources=glob.glob("src/difflogic/cuda/*.cpp") + glob.glob("src/difflogic/cuda/*.cu"),
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
            ),
            CUDAExtension(
                name="convlogic_cuda",
                sources=glob.glob("src/convlogic/cuda/*.cpp") + glob.glob("src/convlogic/cuda/*.cu"),
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
            ),
        ]
    else:
        print(">>> No CUDA extensions will be built.")

    return ext_modules


setup(
    name="convlogic",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
