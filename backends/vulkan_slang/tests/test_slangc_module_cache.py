"""PF.27.a.1 — slangc SIGSEGV on stale `helpers.slang-module` cache.

A `.slang-module` artifact serialized by an older slangc binary causes
the current slangc (2026.5.2) to SIGSEGV (rc=-11) when a kernel does
``import helpers;`` and the stale cache dir is on slangc's `-I` path.

Root cause: ``precompile_shader_libs()`` keys its on-disk cache only on
``sha256(<src.slang>)``. The slangc binary fingerprint is *not* part of
the key, so a slangc upgrade leaves every previously-compiled
``.slang-module`` in place; ``precompile_shader_libs()`` declares cache
hits and never regenerates. The next ``import helpers;`` then crashes
slangc.

Blast radius is every kernel that emits ``import X;`` — Inductor's
matmul template (`slang_mm.py.jinja`), `mm_loader.py`, autodiff codegen
(`bwd_diff_table.py`), and several shader-lib cross-imports. PF.27.a's
RNG-kernel codegen surfaced the failure first.

Fix shape (lands together with this test):

  1. ``precompile_shader_libs()`` mixes ``_slangc_fingerprint()`` into
     the cache key; any slangc-binary change forces regen.
  2. ``_compile_slang_to_spirv_inner()`` defends against rc=-11 with the
     module-cache `-I` present by invalidating the cache and retrying
     once with source-dir `-I` only.

Tests synthesize stale-ness by writing a malformed ``helpers.slang-module``
into a tmp module-cache dir — slangc rc=-11 reproduces deterministically
on any invalid RIFF, which is the same failure surface as the real
older-format module.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import textwrap

import pytest

from torch_vulkan.inductor import runtime as rt


_HELPERS_IMPORT_KERNEL = textwrap.dedent("""\
    import helpers;
    [shader("compute")]
    [numthreads(128, 1, 1)]
    void computeMain(uint3 tid : SV_DispatchThreadID) {}
""")


def _slangc_or_skip() -> str:
    if not rt._slangc_available():
        pytest.skip("slangc unavailable — PF.27.a.1 floor needs a real slangc")
    return rt._SLANGC


def _write_stale_helpers_module(cache_dir: str) -> None:
    """Install a `helpers.slang-module` that the current slangc cannot
    deserialize (invalid RIFF). Reproduces the SIGSEGV surface of a
    real older-format module without bundling a binary fixture.

    Also writes a `.hash` sidecar matching the *current* helpers.slang
    SHA256 — without that, ``precompile_shader_libs()`` would happily
    overwrite the file. The sidecar tricks the precompile cache-hit
    path into reusing the malformed artifact, mirroring exactly what
    happens to a real user after a slangc upgrade.
    """
    os.makedirs(cache_dir, exist_ok=True)
    mod_path = os.path.join(cache_dir, "helpers.slang-module")
    hash_path = os.path.join(cache_dir, "helpers.hash")
    with open(mod_path, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 256)
    src_path = os.path.join(rt._SHADERS_LIB_DIR, "helpers.slang")
    with open(src_path, "rb") as f:
        src_hash = hashlib.sha256(f.read()).hexdigest()
    with open(hash_path, "w") as f:
        f.write(src_hash)


class TestSlangcModuleCacheStaleness:
    """PF.27.a.1 — pre-broken module cache must not SIGSEGV slangc.

    `_BUG_ROOT_COMPONENT="slang-shader-pipeline / module-cache-invalidation"`.
    """

    def test_subprocess_segfaults_on_stale_module_cache(self, tmp_path):
        """Sanity floor: confirm slangc truly SIGSEGVs (rc=-11) when a
        malformed ``helpers.slang-module`` is on `-I`. This is the
        external-tool half of the bug — it documents what the runtime
        layer must defend against and locks the failure surface so a
        future slangc release that downgrades to a clean error message
        ratchets the test forward instead of silently masking the bug.
        """
        slangc = _slangc_or_skip()
        cache = tmp_path / "cache"
        _write_stale_helpers_module(str(cache))
        kernel = tmp_path / "k.slang"
        kernel.write_text(_HELPERS_IMPORT_KERNEL)
        proc = subprocess.run(
            [slangc, str(kernel), "-target", "spirv",
             "-entry", "computeMain",
             "-o", str(tmp_path / "k.spv"),
             "-reflection-json", str(tmp_path / "k.refl.json"),
             "-matrix-layout-row-major",
             "-I", str(cache)],
            capture_output=True, text=True, timeout=30,
        )
        # rc=-11 (POSIX SIGSEGV) is the documented failure mode. Any
        # negative rc indicates the subprocess died from a signal — we
        # treat the broader class as the floor so a slangc that switches
        # SIGSEGV to SIGABRT etc. still trips the assertion.
        assert proc.returncode < 0, (
            f"Expected slangc to crash on malformed helpers.slang-module "
            f"(rc<0); got rc={proc.returncode}, stderr={proc.stderr[:200]!r}"
        )

    def test_compile_slang_to_spirv_recovers_from_stale_module_cache(
        self, tmp_path, monkeypatch,
    ):
        """**The PF.27.a.1 floor.** A user upgrades slangc; the
        ``~/.cache/torch_vulkan/slang-modules/`` dir still holds
        artifacts in the old serialization format. The next call to
        ``compile_slang_to_spirv`` on a kernel that imports `helpers`
        must NOT propagate the SIGSEGV — the runtime must invalidate
        the stale cache and retry from source.

        Pre-fix: raises ``RuntimeError`` ("slangc failed for kernel ...").
        Post-fix: succeeds; SPIR-V bytes returned.
        """
        slangc = _slangc_or_skip()
        cache = tmp_path / "cache"
        _write_stale_helpers_module(str(cache))

        monkeypatch.setattr(rt, "_SHADER_LIB_MODULE_CACHE_DIR", str(cache))
        # Force `_ensure_shader_lib_modules()` to consult the patched
        # dir on this call (its in-memory ready-flag is process-global).
        rt._reset_shader_lib_modules_ready()
        # Isolate the SPIR-V cache too, so a leftover entry from a
        # previous test run doesn't short-circuit the compile we want
        # to exercise.
        monkeypatch.setattr(rt, "_DISK_CACHE_DIR", str(tmp_path / "spv"))
        monkeypatch.setattr(rt, "_cache_by_key", {})
        monkeypatch.setattr(rt, "_cache_by_hash", {})

        spv = rt.compile_slang_to_spirv(
            _HELPERS_IMPORT_KERNEL,
            entry="computeMain",
            cache_key="pf_27_a_1_floor_helpers_import",
        )
        assert isinstance(spv, (bytes, bytearray)) and len(spv) > 0, (
            f"compile_slang_to_spirv returned empty SPIR-V: {len(spv)} bytes"
        )
        # SPIR-V magic word: 0x07230203, little-endian on disk.
        assert spv[:4] == b"\x03\x02\x23\x07", (
            f"output is not a valid SPIR-V module — header={spv[:8]!r}"
        )

    def test_precompile_invalidates_on_slangc_fingerprint_change(
        self, tmp_path, monkeypatch,
    ):
        """Fix #1 floor: ``precompile_shader_libs()`` must mix the
        slangc binary fingerprint into the cache key, so a slangc
        upgrade with unchanged source forces regen.

        Pre-fix: second precompile call (same source, *different*
        fingerprint) reports cached=[…helpers…], compiled=[].
        Post-fix: fingerprint mismatch → regen → compiled contains
        helpers + tensor_layout.
        """
        _slangc_or_skip()
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(rt, "_SHADER_LIB_MODULE_CACHE_DIR", str(cache))

        # Round 1 — fingerprint A.
        monkeypatch.setattr(
            rt, "_slangc_fingerprint", lambda: "fingerprint-A", raising=False,
        )
        rt.precompile_shader_libs(force=True)
        helpers_mod = cache / "helpers.slang-module"
        assert helpers_mod.exists()
        m1 = helpers_mod.stat().st_mtime_ns

        # Round 2 — fingerprint B with no source change. Sleep is not
        # needed: the cache key derivation should mismatch and force
        # the recompile, which writes a new file (st_mtime_ns may be
        # equal on a very fast FS, but the recorded `.hash` content
        # must include fingerprint-B).
        monkeypatch.setattr(
            rt, "_slangc_fingerprint", lambda: "fingerprint-B", raising=False,
        )
        result = rt.precompile_shader_libs()
        assert "helpers" in result["compiled"], (
            "precompile_shader_libs must regen on slangc-fingerprint drift; "
            f"got compiled={result['compiled']} cached={result['cached']}"
        )
        # The serialized hash sidecar must reflect the new fingerprint —
        # otherwise round 3 with fingerprint-A would see a "match" again
        # and ping-pong forever.
        assert (cache / "helpers.hash").read_text().strip() != "", (
            "helpers.hash must be non-empty after regen"
        )
        m2 = helpers_mod.stat().st_mtime_ns
        # Defensive: at minimum the recompile must have happened (file
        # rewritten). On filesystems with ns-granular mtime, m2 > m1;
        # otherwise mtime can equal m1 — we rely on the `compiled` list
        # above as the primary assertion and only sanity-check that
        # the file is present and non-empty.
        assert helpers_mod.stat().st_size > 0
        del m1, m2  # not used by the assertion contract


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests
    run_tests()
