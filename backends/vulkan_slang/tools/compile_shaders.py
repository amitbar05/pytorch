#!/usr/bin/env python3
"""Compile Slang shaders to SPIR-V and embed as C++ byte arrays."""

import os
import re
import subprocess
import sys
from pathlib import Path


SHADER_DIR = Path(__file__).parent.parent / "shaders"
OUTPUT_DIR = Path(__file__).parent.parent / "csrc" / "generated"
_project_root = Path(__file__).resolve().parent.parent
_default_slangc = str(_project_root / "third_party" / "slang" / "build" / "bin" / "slangc")
SLANGC = os.environ.get("SLANGC", _default_slangc)


def find_shaders():
    """Find all .slang files recursively."""
    return sorted(SHADER_DIR.rglob("*.slang"))


def has_entry_point(slang_path: Path, entry_point: str) -> bool:
    """Check if a Slang file defines a specific entry point function."""
    try:
        content = slang_path.read_text()
        # Match function definitions like: void computeMain(...), [shader("compute")] void bwd_computeMain(...)
        # Also handle [Differentiable] and [BackwardDerivative] attributes.
        pattern = rf"^(?:\[.*?\]\s+)*\b(?:void|int|float|uint|bool)\s+{re.escape(entry_point)}\s*\("
        return bool(re.search(pattern, content, re.MULTILINE))
    except Exception:
        return False


def compile_to_spirv(slang_path: Path, entry_point: str, output_path: Path):
    """Compile a Slang file to SPIR-V."""
    cmd = [
        SLANGC,
        str(slang_path),
        "-target", "spirv",
        "-entry", entry_point,
        "-o", str(output_path),
        "-matrix-layout-row-major",
    ]
    # Add include paths for imports: shaders/ (top-level) + shaders/lib/ (module files)
    cmd.extend(["-I", str(SHADER_DIR)])
    cmd.extend(["-I", str(SHADER_DIR / "lib")])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error compiling {slang_path}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    return True


def spirv_to_c_array(spv_path: Path, var_name: str) -> str:
    """Convert SPIR-V binary to C++ byte array."""
    with open(spv_path, "rb") as f:
        data = f.read()

    words = []
    for i in range(0, len(data), 4):
        word = int.from_bytes(data[i:i+4], byteorder="little")
        words.append(f"0x{word:08x}")

    lines = []
    lines.append(f"// Auto-generated from {spv_path.name}")
    lines.append(f"static const uint32_t {var_name}[] = {{")

    for i in range(0, len(words), 8):
        chunk = ", ".join(words[i:i+8])
        lines.append(f"    {chunk},")

    lines.append("};")
    lines.append(f"static const size_t {var_name}_size = sizeof({var_name});")
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shaders = find_shaders()
    # Skip common/ modules (they're imported, not compiled directly)
    # Skip shaders/lib/ (pure Slang module files with no computeMain entry point)
    # Skip shaders/reduction.slang (top-level re-export alias, not a kernel)
    shaders = [s for s in shaders if "common/" not in str(s)
               and "shaders/lib/" not in str(s)
               and str(s) != str(SHADER_DIR / "reduction.slang")]

    if not shaders:
        print("No shaders found to compile.")
        return

    header_parts = [
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        "",
        "// Auto-generated shader SPIR-V byte arrays",
        "namespace torch_vulkan { namespace shaders {",
        "",
    ]

    success_count = 0
    fail_count = 0

    for shader in shaders:
        rel_path = shader.relative_to(SHADER_DIR)
        # Create variable name from path: binary/add.slang -> binary_add
        var_base = str(rel_path.with_suffix("")).replace("/", "_").replace("\\", "_")

        # Compile forward entry point
        spv_path = OUTPUT_DIR / f"{var_base}_fwd.spv"
        print(f"Compiling {rel_path} -> {var_base}_fwd.spv")

        if compile_to_spirv(shader, "computeMain", spv_path):
            c_code = spirv_to_c_array(spv_path, f"{var_base}_fwd")
            header_parts.append(c_code)
            header_parts.append("")
            success_count += 1
        else:
            fail_count += 1

        # Only try backward entry point if the shader defines it
        if has_entry_point(shader, "bwd_computeMain"):
            spv_bwd_path = OUTPUT_DIR / f"{var_base}_bwd.spv"
            if compile_to_spirv(shader, "bwd_computeMain", spv_bwd_path):
                c_code = spirv_to_c_array(spv_bwd_path, f"{var_base}_bwd")
                header_parts.append(c_code)
                header_parts.append("")
        # else: no bwd entry point; silently skip (expected for many shaders)

    header_parts.append("}} // namespace torch_vulkan::shaders")

    # Write combined header
    header_path = OUTPUT_DIR / "shaders.h"
    with open(header_path, "w") as f:
        f.write("\n".join(header_parts))

    print(f"\nCompiled {success_count} shaders ({fail_count} failed)")
    print(f"Generated: {header_path}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
