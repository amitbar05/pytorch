"""Reflection helpers — descriptor counts, binding layout, reflection JSON access.

These functions query the Slang reflection JSON cached during compilation
to determine binding counts, descriptor array sizes, and push-constant layout
without re-reading the source.
"""

import hashlib
from typing import Optional


def _get_reflection_cache():
    from .slangc import _reflection_cache
    return _reflection_cache


def _get_disk_reflection_read():
    from .slangc import _disk_reflection_read
    return _disk_reflection_read


def _get_normalize_slang_source():
    from .slangc import _normalize_slang_source
    return _normalize_slang_source


def get_reflection_json(hash_key: str) -> Optional[str]:
    """Look up the cached slangc reflection JSON blob for a given key.

    First checks the in-memory cache, then the on-disk sidecar
    (`<spirv_cache_dir>/<prefix>/<rest>.refl.json`). Returns ``None`` when
    no reflection has ever been emitted for that key (e.g. compiled
    pre-P0.4 or via a slangc that doesn't support `-reflection-json`).
    """
    _reflection_cache = _get_reflection_cache()
    _disk_reflection_read = _get_disk_reflection_read()
    hit = _reflection_cache.get(hash_key)
    if hit is not None:
        return hit
    blob = _disk_reflection_read(hash_key)
    if blob is not None:
        _reflection_cache[hash_key] = blob
    return blob


def _binding_descriptor_count(param: dict) -> int:
    """Return the ``descriptorCount`` for a single reflection parameter.

    Looks for one of the recognised array shapes the slangc reflection
    JSON emits for ``RWStructuredBuffer<T> name[N]`` / nested
    ``ParameterBlock`` array members:

    - ``param["type"]["kind"] == "array"`` with ``elementCount: N`` —
      flat top-level array binding (the common case post-N+1.5).
    - ``param["binding"]["size"] >= 1`` — slangc occasionally inlines
      the count there for `descriptorTableSlot` kinds.
    - ``param["binding"]["subBindings"][...]`` — nested layout where the
      array element count lives one level deep (older slangc shapes).

    Returns ``1`` when none of the above match (i.e. a plain scalar
    binding).
    """
    t = param.get("type") or {}
    if t.get("kind") == "array":
        ec = t.get("elementCount")
        if ec is not None:
            try:
                n = int(ec)
                return n if n >= 1 else 1
            except (TypeError, ValueError):
                pass
    b = param.get("binding") or {}
    sz = b.get("size")
    if sz is not None:
        try:
            n = int(sz)
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass
    sub = b.get("subBindings") or []
    for s in sub:
        sb = s.get("binding") or {}
        ssz = sb.get("size")
        if ssz is not None:
            try:
                n = int(ssz)
                if n >= 1:
                    return n
            except (TypeError, ValueError):
                continue
        st = s.get("type") or {}
        if st.get("kind") == "array":
            ec = st.get("elementCount")
            if ec is not None:
                try:
                    n = int(ec)
                    if n >= 1:
                        return n
                except (TypeError, ValueError):
                    continue
    return 1

def reflection_layout(reflection_json: str) -> dict:
    """Extract the descriptor + push-constant layout from a slangc reflection JSON.

    Returns ``{"bindings": [(set, index, name), ...],
    "descriptor_counts": [int, ...], "push_constant_size": int}``.
    The ``descriptor_counts`` list is parallel to ``bindings`` (same
    length, same order after sort) and carries the per-binding
    ``descriptorCount`` extracted from the slangc reflection. A flat
    binding has count ``1``; an array binding (e.g.
    ``RWStructuredBuffer<float> outs[4]``) has count ``4``.

    Lets callers populate `VkDescriptorSetLayoutBinding` arrays without
    having to count tensors or guess push-constant sizes manually.
    """
    import json

    data = json.loads(reflection_json)
    paired: list[tuple[tuple[int, int, str], int]] = []
    pc_size = 0
    for p in data.get("parameters", []):
        b = p.get("binding") or {}
        kind = b.get("kind")
        if kind == "descriptorTableSlot":
            key = (b.get("space", 0), b.get("index", 0), p.get("name", ""))
            paired.append((key, _binding_descriptor_count(p)))
        elif kind == "pushConstantBuffer":
            t = p.get("type", {})
            elv = t.get("elementVarLayout", {})
            elb = elv.get("binding", {})
            pc_size = max(pc_size, int(elb.get("size", 0)))
    paired.sort(key=lambda kv: kv[0])
    bindings = [k for k, _ in paired]
    descriptor_counts = [c for _, c in paired]
    return {
        "bindings": bindings,
        "descriptor_counts": descriptor_counts,
        "push_constant_size": pc_size,
    }

def get_reflected_binding_count(spv: bytes) -> int | None:
    """Extract the number of storage-buffer bindings from SPIR-V reflection.

    Returns ``None`` when no reflection JSON is cached for this SPV (e.g.
    compiled with a slangc that doesn't support ``-reflection-json``).
    Callers should fall back to the hand-counted ``n_buffers`` in that case.
    """
    hash_key = hashlib.sha256(spv).hexdigest()
    refl_json = get_reflection_json(hash_key)
    if refl_json is None:
        # Try the full-src hash — the SPV hash alone may not match if the
        # reflection was keyed by (entry + src), not by SPV bytes.
        return None
    layout = reflection_layout(refl_json)
    return len(layout["bindings"])

def _get_reflected_buffer_count_from_cache_key(
    src: str,
    entry: str = "computeMain",
    include_paths: tuple[str, ...] = (),
) -> int | None:
    """Like :func:`get_reflected_binding_count` but keyed by source hash."""
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    hash_key = hashlib.sha256(
        (entry + "\n" + _get_normalize_slang_source()(src) + inc_tag).encode()
    ).hexdigest()
    refl = get_reflection_json(hash_key)
    if refl is None:
        return None
    return len(reflection_layout(refl)["bindings"])

def get_reflected_descriptor_counts(spv: bytes) -> Optional[list[int]]:
    """N+1.5.a — extract per-binding ``descriptorCount`` from SPV reflection.

    Returns a list parallel to the binding list (same order as
    :func:`reflection_layout`'s ``bindings``). Each entry is the
    ``descriptorCount`` for that binding — ``1`` for a flat binding,
    ``N`` for an array binding (e.g. ``RWStructuredBuffer<T> arr[N]``).

    Returns ``None`` when no reflection JSON is cached for this SPV
    (e.g. compiled with a slangc that doesn't support
    ``-reflection-json``). Callers should fall back to ``[1] * n_buffers``
    in that case.
    """
    hash_key = hashlib.sha256(spv).hexdigest()
    refl_json = get_reflection_json(hash_key)
    if refl_json is None:
        return None
    layout = reflection_layout(refl_json)
    return list(layout.get("descriptor_counts") or [])

def _get_reflected_descriptor_counts_from_src(
    src: str,
    entry: str = "computeMain",
    include_paths: tuple[str, ...] = (),
) -> Optional[list[int]]:
    """Source-hash variant of :func:`get_reflected_descriptor_counts`.

    The reflection JSON is keyed by ``sha256(entry + normalized_src)``
    (see :func:`compile_slang_to_spirv`), so a SPV-hash lookup will
    miss. This helper performs the same lookup but using the source hash
    that ``compile_slang_to_spirv`` actually wrote.
    """
    inc_tag = "" if not include_paths else "\nINC=" + "|".join(include_paths)
    hash_key = hashlib.sha256(
        (entry + "\n" + _get_normalize_slang_source()(src) + inc_tag).encode()
    ).hexdigest()
    refl_json = get_reflection_json(hash_key)
    if refl_json is None:
        return None
    layout = reflection_layout(refl_json)
    return list(layout.get("descriptor_counts") or [])
