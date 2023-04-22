import os
from setuptools import setup
import torch
from pkg_resources import parse_version
import subprocess
import re
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def min_supported_compute_capability(cuda_version):
	if cuda_version >= parse_version("12.0"):
		return 50
	else:
		return 20

def max_supported_compute_capability(cuda_version):
	if cuda_version < parse_version("11.0"):
		return 75
	elif cuda_version < parse_version("11.1"):
		return 80
	elif cuda_version < parse_version("11.8"):
		return 86
	else:
		return 90

if "PENC_CUDA_ARCHITECTURES" in os.environ and os.environ["PENC_CUDA_ARCHITECTURES"]:
	compute_capabilities = [int(x) for x in os.environ["PENC_CUDA_ARCHITECTURES"].replace(";", ",").split(",")]
	print(f"Obtained compute capabilities {compute_capabilities} from environment variable PENC_CUDA_ARCHITECTURES")
elif torch.cuda.is_available():
	major, minor = torch.cuda.get_device_capability()
	compute_capabilities = [major * 10 + minor]
	print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
else:
	raise EnvironmentError("Unknown compute capability. Specify the target compute capabilities in the PENC_CUDA_ARCHITECTURES environment variable or install PyTorch with the CUDA backend to detect it automatically.")

nvcc_flags = [
    '-O3', '-std=c++14', 
    "--generate-line-info", "--extended-lambda", "--expt-relaxed-constexpr",
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

# Get CUDA version and make sure the targeted compute capability is compatible
if os.system("nvcc --version") == 0:
	nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
	cuda_version = re.search(r"release (\S+),", nvcc_out)

	if cuda_version:
		cuda_version = parse_version(cuda_version.group(1))
		print(f"Detected CUDA version {cuda_version}")
		supported_compute_capabilities = [
			cc for cc in compute_capabilities if cc >= min_supported_compute_capability(cuda_version) and cc <= max_supported_compute_capability(cuda_version)
		]

		if not supported_compute_capabilities:
			supported_compute_capabilities = [max_supported_compute_capability(cuda_version)]

		if supported_compute_capabilities != compute_capabilities:
			print(f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead.")
			compute_capabilities = supported_compute_capabilities

min_compute_capability = min(compute_capabilities)

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
    nvcc_flags += [
		"-Xcompiler=-Wno-float-conversion",
		"-Xcompiler=-fno-strict-aliasing",
	]
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']
    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

base_source_files = [
    "./src/PyBridge.cxx",
    "./src/Sphere.cu",
    "./src/OccupancyGrid.cu",
    "./src/RaySampler.cu",
    "./src/RaySamplesPacked.cu",
    "./src/VolumeRendering.cu"
]


VERSION = "0.2.0"
print(f"Building PyTorch extension for permuto_sdf version {VERSION}")

_src_path = os.path.dirname(os.path.abspath(__file__))

def make_extension(compute_capability, base_nvcc_flags, base_definitions, base_cflags, root_dir):
	nvcc_flags = base_nvcc_flags + [f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}" for code in ["compute", "sm"]]
	definitions = base_definitions + [f"-DPENC_MIN_GPU_ARCH={compute_capability}"]

	source_files = base_source_files 

	nvcc_flags = nvcc_flags + definitions
	cflags = base_cflags + definitions

	ext = CUDAExtension(
		name=f"py_permuto_sdf",
		sources=source_files,
		include_dirs=[
			"%s/include" % root_dir,
			"%s/kernels" % root_dir,
			"%s/deps" % root_dir,
		],
		extra_compile_args={"cxx": cflags, "nvcc": nvcc_flags},
		libraries=["cuda", "cudadevrt", "cudart_static"],
	)
	return ext

'''
Usage:

python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)

python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)

python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)

'''
print("Building for compute capability: ", min_compute_capability)
ext_modules = [make_extension(min_compute_capability, nvcc_flags, [], c_flags, _src_path)]

setup(
    name='py_permuto_sdf', # package name, import this to use python API
	version=VERSION,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension,
    }
)