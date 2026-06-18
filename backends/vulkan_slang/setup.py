import os
import subprocess
import sys
from pathlib import Path

# Ensure the installed PyTorch (in the venv) is used instead of any source-tree
# copy that may be on sys.path (e.g. from the parent pytorch repo).
# Without this, setup.py picks up the source tree's torch/ which may require
# C-extension symbols not present in the installed version.
_venv_site = os.path.join(
    os.path.dirname(__file__),
    ".venv",
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)
if os.path.isdir(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

# Remove the root pytorch source tree from sys.path so `import torch` always
# resolves to the venv-installed copy.
_root_pytorch = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_pytorch in sys.path:
    sys.path.remove(_root_pytorch)

# Ensure venv-local tools (ninja, slangc, etc.) are on PATH before any
# build backend probes the environment.
_venv_bin = str(Path(__file__).parent / ".venv" / "bin")
if os.path.isdir(_venv_bin):
    os.environ.setdefault("PATH", _venv_bin + os.pathsep + os.environ.get("PATH", ""))
    # Also prepend to the current PATH for this process
    if _venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _venv_bin + os.pathsep + os.environ["PATH"]

# Enable parallel compilation
os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 4))

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


# Collect all C++ sources
def get_sources():
    src_dirs = ["csrc/vulkan", "csrc/backend", "csrc/ops", "csrc/autocast"]
    sources = []
    root = Path(__file__).parent
    for d in src_dirs:
        dir_path = root / d
        if dir_path.exists():
            sources.extend(
                str(p.relative_to(root))
                for p in dir_path.glob("*.cpp")
                if p.name != "aoti_shims.cpp"  # linked via extra_objects below
            )
    sources.append("csrc/init.cpp")
    # M16.4: build-time validation — model_ops.cpp must not exist.
    _validate_no_model_ops(root)
    return sources


def _validate_no_model_ops(root):
    """M16.4: Fail the build if model_ops.cpp exists."""
    forbidden = root / "csrc" / "ops" / "model_ops.cpp"
    if forbidden.exists():
        raise SystemExit(
            "M16 BLOCKER: csrc/ops/model_ops.cpp exists. "
            "This file was deleted as part of Track 4 (anti-goal #2). "
            "New eager ops belong in csrc/ops/legacy_eager.cpp."
        )
    required = root / "csrc" / "ops" / "legacy_eager.cpp"
    if not required.exists():
        raise SystemExit(
            "M16 BLOCKER: csrc/ops/legacy_eager.cpp is missing. "
            "This file is the required replacement for the deleted model_ops.cpp."
        )


root_dir = Path(__file__).parent.resolve()
vma_dir = str(root_dir / "third_party/VulkanMemoryAllocator/include")


# Custom build extension that compiles shaders before C++
class ShaderBuildExtension(BuildExtension):
    def build_extensions(self):
        root = Path(__file__).parent
        gen_dir = root / "csrc" / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        header = gen_dir / "shaders.h"

        # Try to compile shaders before building C++
        # Skip if SKIP_SHADER_COMPILE=1 or if shaders.h already exists (avoid slow recompilation)
        if not os.environ.get("SKIP_SHADER_COMPILE") and not header.exists():
            try:
                print("Compiling Slang shaders...")
                subprocess.run(
                    [sys.executable, "tools/compile_shaders.py"],
                    check=True,
                    cwd=str(root),
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Warning: Shader compilation failed ({e}).")
                if not header.exists():
                    print("Generating stub shaders.h for compilation...")
                    subprocess.run(
                        [sys.executable, "tools/generate_stub_shaders.py"],
                        check=True,
                        cwd=str(root),
                    )
        super().build_extensions()


setup(
    name="torch_vulkan",
    version="0.1.0",
    description="PyTorch Vulkan backend with Slang shaders for full training support",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CppExtension(
            name="torch_vulkan._C",
            sources=get_sources(),
            include_dirs=[
                str(root_dir / "csrc"),
                str(root_dir / "csrc/generated"),
                vma_dir,
            ],
            libraries=["vulkan"],
            define_macros=[
                ("VMA_STATIC_VULKAN_FUNCTIONS", "0"),
                ("VMA_DYNAMIC_VULKAN_FUNCTIONS", "1"),
            ],
            extra_compile_args=["-std=c++17"],
            extra_objects=[str(root_dir / "csrc" / "backend" / "aoti_shims.o")],
        ),
    ],
    cmdclass={"build_ext": ShaderBuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    entry_points={
        "torch.backends": ["vulkan = torch_vulkan:_register"],
    },
)
