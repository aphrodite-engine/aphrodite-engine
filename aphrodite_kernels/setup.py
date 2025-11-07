import ctypes
import importlib.util
import logging
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

# Use environment variables only - no dependency on main project
envs = None
APHRODITE_TARGET_DEVICE = os.getenv("APHRODITE_TARGET_DEVICE", "cuda")

if sys.platform.startswith("darwin") and APHRODITE_TARGET_DEVICE != "cpu":
    logger.warning("APHRODITE_TARGET_DEVICE automatically set to `cpu` due to macOS")
    APHRODITE_TARGET_DEVICE = "cpu"
elif not (sys.platform.startswith("linux") or sys.platform.startswith("darwin")):
    logger.warning(
        "Aphrodite kernels only supports Linux platform (including WSL) and MacOS."
        "Building on {}, "
        "so Aphrodite kernels may not be able to run correctly",
        sys.platform,
    )
    APHRODITE_TARGET_DEVICE = "empty"
elif (
    sys.platform.startswith("linux")
    and torch.version.cuda is None
    and os.getenv("APHRODITE_TARGET_DEVICE") is None
    and torch.version.hip is None
):
    # if cuda or hip is not available and APHRODITE_TARGET_DEVICE is not set,
    # fallback to cpu
    APHRODITE_TARGET_DEVICE = "cpu"

MAIN_CUDA_VERSION = "12.8"


def _get_available_memory_bytes() -> int | None:
    """Return available system memory in bytes, or None if unknown.

    Tries multiple strategies in order:
    - psutil (if available)
    - POSIX sysconf with SC_AVPHYS_PAGES
    - /proc/meminfo on Linux (MemAvailable)
    - vm_stat on macOS (free + inactive pages)
    """
    # Try psutil if available
    with suppress(Exception):
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)

    # Try POSIX sysconf for available pages
    with suppress(Exception):
        page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[arg-type]
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")  # type: ignore[arg-type]
        return int(page_size) * int(avail_pages)

    # Linux fallback: /proc/meminfo
    with suppress(Exception):
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        # Value is in kB
                        return int(parts[1]) * 1024

    # macOS fallback: vm_stat
    with suppress(Exception):
        if sys.platform.startswith("darwin"):
            out = subprocess.check_output(["vm_stat"], encoding="utf-8")
            page_size_bytes = 4096
            for line in out.splitlines():
                if "page size of" in line and "bytes" in line:
                    # e.g., "Mach VM Stats: (page size of 16384 bytes)"
                    with suppress(Exception):
                        page_size_bytes = int(line.split("page size of")[1].split("bytes")[0].strip())
                    break
            pages_free = 0
            pages_inactive = 0
            for line in out.splitlines():
                if line.strip().startswith("Pages free"):
                    pages_free = int(line.split(":")[1].strip().strip(". "))
                elif line.strip().startswith("Pages inactive"):
                    pages_inactive = int(line.split(":")[1].strip().strip(". "))
            return (pages_free + pages_inactive) * page_size_bytes

    return None


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(int(os.getenv("APHRODITE_DISABLE_SCCACHE", "0")))


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_url_available(url: str) -> bool:
    from urllib.request import urlopen

    status = None
    try:
        with urlopen(url) as f:
            status = f.status
    except Exception:
        return False
    return status == 200


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or computed from available RAM (8 GiB per job),
        # falling back to the number of CPUs available.
        num_jobs = getattr(envs, "MAX_JOBS", None) if envs else None
        if num_jobs is None:
            num_jobs = os.getenv("MAX_JOBS")
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            available_bytes = _get_available_memory_bytes()
            if available_bytes is not None and available_bytes > 0:
                available_gib = max(0, available_bytes // (1024**3))
                # Heuristic: 8 GiB per job
                num_jobs = max(1, int(available_gib // 8))
                logger.info(
                    "RAM heuristic: ~%d GiB avail -> num_jobs=%d (8 GiB/job). If you think this is too low or too high,"
                    " set MAX_JOBS to a higher value.",
                    available_gib,
                    num_jobs,
                )
            else:
                try:
                    # os.sched_getaffinity() not always available; fallback to
                    # os.cpu_count() when needed.
                    num_jobs = len(os.sched_getaffinity(0))
                    logger.info("CPU heuristic: %d jobs (RAM unknown).", num_jobs)
                except AttributeError:
                    num_jobs = os.cpu_count()
                    logger.info("CPU heuristic: os.cpu_count()=%d (RAM unk).", num_jobs)

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = getattr(envs, "NVCC_THREADS", None) if envs else None
            if nvcc_threads is None:
                nvcc_threads = os.getenv("NVCC_THREADS")
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info("Using NVCC_THREADS=%d as the number of nvcc threads.", nvcc_threads)
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = getattr(envs, "CMAKE_BUILD_TYPE", None) if envs else None
        if cfg is None:
            cfg = os.getenv("CMAKE_BUILD_TYPE") or default_cfg

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DAPHRODITE_TARGET_DEVICE={}".format(APHRODITE_TARGET_DEVICE),
        ]

        verbose = getattr(envs, "VERBOSE", False) if envs else False
        if not verbose:
            verbose = bool(os.getenv("VERBOSE", False))
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ["-DAPHRODITE_PYTHON_EXECUTABLE={}".format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ["-DAPHRODITE_PYTHON_PATH={}".format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ["-DNVCC_THREADS={}".format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        # Make sure we use the nvcc from CUDA_HOME
        if _is_cuda():
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]

        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args += other_cmake_args.split()

        subprocess.check_call(["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args], cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("aphrodite_kernels.").removeprefix("aphrodite_flash_attn.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            # We assume only the final component of extension prefix is added by
            # CMake, this is currently true for current extensions but may not
            # always be the case.
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = ["cmake", "--install", ".", "--prefix", prefix, "--component", target_name(ext.name)]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # Run the standard build_ext command to compile the extensions
        super().run()


def _is_hpu() -> bool:
    # if APHRODITE_TARGET_DEVICE env var was set explicitly, skip autodetection
    if os.getenv("APHRODITE_TARGET_DEVICE", None) == APHRODITE_TARGET_DEVICE:
        return APHRODITE_TARGET_DEVICE == "hpu"

    # if APHRODITE_TARGET_DEVICE was not set explicitly, check if hl-smi works;
    # otherwise, check if habanalabs driver is loaded
    is_hpu_available = False
    try:
        out = subprocess.run(["hl-smi"], capture_output=True, check=True)
        is_hpu_available = out.returncode == 0
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        if sys.platform.startswith("linux"):
            try:
                output = subprocess.check_output("lsmod | grep habanalabs | wc -l", shell=True)
                is_hpu_available = int(output) > 0
            except (ValueError, FileNotFoundError, PermissionError, subprocess.CalledProcessError):
                pass
    return is_hpu_available


def _no_device() -> bool:
    return APHRODITE_TARGET_DEVICE == "empty"


def _is_windows() -> bool:
    return APHRODITE_TARGET_DEVICE == "windows"


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return APHRODITE_TARGET_DEVICE == "cuda" and has_cuda and not (_is_tpu() or _is_hpu())


def _is_hip() -> bool:
    return (APHRODITE_TARGET_DEVICE == "cuda" or APHRODITE_TARGET_DEVICE == "rocm") and torch.version.hip is not None


def _is_tpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "tpu"


def _is_cpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "cpu"


def _is_xpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "xpu"


def _build_custom_ops() -> bool:
    # Skip building custom ops if using precompiled binaries
    if os.environ.get("APHRODITE_USE_PRECOMPILED", "").strip().lower() in ("1", "true"):
        return False
    return _is_cuda() or _is_hip() or _is_cpu()


def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    # see https://github.com/ROCm/rocm-core/blob/d11f5c20d500f729c393680a01fa902ebf92094b/rocm_version.cpp#L21
    try:
        librocm_core_file = Path(ROCM_HOME) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(librocm_core_file)
        VerErrors = ctypes.c_uint32
        get_rocm_core_version = librocm_core.getROCmVersion
        get_rocm_core_version.restype = VerErrors
        get_rocm_core_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()

        if get_rocm_core_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)) == 0:
            return f"{major.value}.{minor.value}.{patch.value}"
        return None
    except Exception:
        return None


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_gaudi_sw_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run("hl-smi", shell=True, text=True, capture_output=True, env={"ENABLE_CONSOLE": "true"})
    if output.returncode == 0 and output.stdout:
        return output.stdout.split("\n")[2].replace(" ", "").split(":")[1][:-1].split("-")[0]
    return "0.0.0"  # when hl-smi is not available


def get_kernels_version() -> str:
    """Get the version string for aphrodite-kernels with platform-specific suffix."""
    if env_version := os.getenv("APHRODITE_KERNELS_VERSION_OVERRIDE"):
        return env_version

    # Use setuptools-scm to get version - works from current directory or parent git repo
    try:
        # get_version will search up the directory tree for .git by default
        version = get_version(write_to="aphrodite_kernels/_version.py")
    except LookupError:
        # Fallback if setuptools-scm can't find version (e.g., not in git repo)
        # Try to read from existing _version.py if it exists
        version_file = ROOT_DIR / "aphrodite_kernels" / "_version.py"
        if version_file.exists():
            try:
                with open(version_file, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("__version__"):
                            version = line.split("=")[1].strip().strip("'\"")
                            break
            except Exception:
                version = "0.0.1"
        else:
            version = "0.0.1"

    sep = "+" if "+" not in version else "."  # dev versions might contain +

    # Get MAIN_CUDA_VERSION from environment variable, otherwise use default
    main_cuda_version = os.getenv("APHRODITE_MAIN_CUDA_VERSION", MAIN_CUDA_VERSION)

    if _no_device():
        if APHRODITE_TARGET_DEVICE == "empty":
            version += f"{sep}empty"
    elif _is_cuda():
        use_precompiled = os.environ.get("APHRODITE_USE_PRECOMPILED", "").strip().lower() in ("1", "true")
        if use_precompiled:
            version += f"{sep}precompiled"
        else:
            # Check CUDA version only when not using precompiled
            cuda_version = str(get_nvcc_cuda_version())
            if cuda_version != main_cuda_version:
                cuda_version_str = cuda_version.replace(".", "")[:3]
                # skip this for source tarball, required for pypi
                if "sdist" not in sys.argv:
                    version += f"{sep}cu{cuda_version_str}"
    elif _is_hip():
        # Get the ROCm Version
        rocm_version = get_rocm_version() or torch.version.hip
        if rocm_version and rocm_version != main_cuda_version:
            version += f"{sep}rocm{rocm_version.replace('.', '')[:3]}"
    elif _is_hpu():
        # Get the Intel Gaudi Software Suite version
        gaudi_sw_version = str(get_gaudi_sw_version())
        if gaudi_sw_version != main_cuda_version:
            gaudi_sw_version = gaudi_sw_version.replace(".", "")[:3]
            version += f"{sep}gaudi{gaudi_sw_version}"
    elif _is_tpu():
        version += f"{sep}tpu"
    elif _is_cpu():
        if APHRODITE_TARGET_DEVICE == "cpu":
            version += f"{sep}cpu"
    elif _is_xpu():
        version += f"{sep}xpu"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


def _has_sm90_or_higher() -> bool:
    """Check if any CUDA architecture in TORCH_CUDA_ARCH_LIST is sm_90 (9.0) or higher."""
    arch_list = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    if not arch_list:
        # If not set, try to detect from available GPU
        try:
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability(0)
                return capability[0] >= 9
        except Exception:
            pass
        # Default to True if we can't determine (conservative)
        return True

    # Parse TORCH_CUDA_ARCH_LIST (format: "7.0 7.5 8.0 8.9 9.0 10.0")
    archs = arch_list.split()
    for arch_str in archs:
        # Remove PTX suffix and other modifiers (e.g., "9.0+PTX" -> "9.0")
        arch_str = arch_str.split("+")[0].split("-")[0]
        try:
            arch_float = float(arch_str)
            if arch_float >= 9.0:
                return True
        except ValueError:
            continue
    return False


ext_modules = []

# Skip building extensions if using precompiled binaries
# Match the pattern from aphrodite/envs.py: accepts "1" or "true" (case-insensitive)
use_precompiled = os.environ.get("APHRODITE_USE_PRECOMPILED", "").strip().lower() in ("1", "true")
if not use_precompiled:
    if _is_cuda() or _is_hip():
        ext_modules.append(CMakeExtension(name="aphrodite_kernels._moe_C"))

    if _is_hip():
        ext_modules.append(CMakeExtension(name="aphrodite_kernels._rocm_C"))

    if _is_cuda():
        disable_flash_attn = os.environ.get("APHRODITE_DISABLE_FLASH_ATTN_COMPILE", "0").strip().lower() in (
            "1",
            "true",
        )
        if not disable_flash_attn:
            ext_modules.append(CMakeExtension(name="aphrodite_kernels.aphrodite_flash_attn._vllm_fa2_C"))
            # Build FA3 when using precompiled artifacts, nvcc >= 12.3, and architecture >= sm_90
            if use_precompiled or (get_nvcc_cuda_version() >= Version("12.3") and _has_sm90_or_higher()):
                ext_modules.append(CMakeExtension(name="aphrodite_kernels.aphrodite_flash_attn._vllm_fa3_C"))

        # Build flashmla when using precompiled artifacts or nvcc >= 12.3.
        # Optional since this doesn't get built (produce an .so file) when
        # not targeting a hopper system
        if use_precompiled or get_nvcc_cuda_version() >= Version("12.3"):
            ext_modules.append(CMakeExtension(name="aphrodite_kernels._flashmla_C", optional=True))
            ext_modules.append(CMakeExtension(name="aphrodite_kernels._flashmla_extension_C", optional=True))
        ext_modules.append(CMakeExtension(name="aphrodite_kernels.cumem_allocator"))

    if _build_custom_ops():
        ext_modules.append(CMakeExtension(name="aphrodite_kernels._C"))

if _no_device():
    ext_modules = []

if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {"build_ext": cmake_build_ext}

_kernels_version = get_kernels_version()

# Print version if --version flag is used
if len(sys.argv) > 1 and "--version" in sys.argv:
    print(_kernels_version)
    sys.exit(0)

setup(
    name="aphrodite-kernels",
    version=_kernels_version,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=["aphrodite_kernels"],
    package_dir={"aphrodite_kernels": "aphrodite_kernels"},
    python_requires=">=3.10",
)
