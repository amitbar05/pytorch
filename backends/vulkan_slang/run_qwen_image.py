"""Run Qwen-Image-2512 on Vulkan with sequential CPU offloading.

Architecture (model_index.json):
  text_encoder : Qwen2_5_VLForConditionalGeneration  (7 B,  ~14 GB bf16)
  transformer  : QwenImageTransformer2DModel          (60 blocks, ~36 GB bf16)
  vae          : AutoencoderKLQwenImage               (~0.5 GB bf16)
  scheduler    : FlowMatchEulerDiscreteScheduler

Sequential offloading pages each nn.Module that owns direct parameters to
Vulkan only for its own forward pass; activations stay on Vulkan throughout.
Peak VRAM ≈ largest single submodule params + activation memory (~600 MB for
one transformer block at 1664×928).

The flush model is deferred-batch: shaders accumulate in a command buffer and
are submitted as a batch.  Moving params back to CPU (post_hook) triggers the
pre_read callback which flushes only if dispatches are pending, so we pay one
GPU submit per submodule forward—correct and safe.

Usage:
    source .venv/bin/activate
    python run_qwen_image.py
"""
import torch
import torch_vulkan  # noqa: F401 — registers the vulkan PrivateUse1 backend
from diffusers import DiffusionPipeline

VULKAN_DEVICE = "vulkan:0"
MODEL_NAME = "Qwen/Qwen-Image-2512"


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _move_to(obj, device):
    """Recursively move tensors nested in tuples/lists/dicts to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to(x, device) for x in obj)
    return obj


def _move_own_to(module: torch.nn.Module, device) -> None:
    """Move only this module's direct parameters and registered buffers to device.

    Unlike module.to(device) this does NOT recurse into children, so each
    module can be paged independently without double-moving shared state.
    """
    for p in module.parameters(recurse=False):
        p.data = p.data.to(device)
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf is not None:
            module._buffers[name] = buf.to(device)


# ── Sequential CPU offloading ─────────────────────────────────────────────────

def _install_hook(module: torch.nn.Module, device: str) -> None:
    """Install forward pre/post hooks that page module's own weights to device."""
    def pre_hook(mod, args, kwargs):
        _move_own_to(mod, device)
        return _move_to(args, device), _move_to(kwargs, device)

    def post_hook(mod, inputs, output):
        # Moving params back to CPU triggers Stream::flush() if dispatches are
        # pending, guaranteeing the GPU is done reading them before we free.
        _move_own_to(mod, "cpu")
        return output

    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook)


def enable_vulkan_sequential_cpu_offload(pipe, device: str = VULKAN_DEVICE) -> None:
    """Per-module CPU offloading for Vulkan (replaces enable_sequential_cpu_offload).

    Hooks every nn.Module that owns direct parameters so that each module's
    weights are on `device` only during its own forward pass.  Intermediate
    activations stay on `device` throughout, eliminating large round-trips.

    This bounds peak VRAM to the largest single submodule's own parameter
    count plus activation tensors—well under 1 GB for one transformer block.
    """
    for name in list(pipe.components.keys()):
        component = getattr(pipe, name, None)
        if not isinstance(component, torch.nn.Module):
            continue
        for submod in component.modules():
            if (any(True for _ in submod.parameters(recurse=False))
                    or any(True for _ in submod.buffers(recurse=False))):
                _install_hook(submod, device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not torch_vulkan.is_available():
        raise RuntimeError("No Vulkan device found")

    pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    # Sequential CPU offloading: each module's weights live on Vulkan only
    # during its forward pass.  To load everything into VRAM instead (needs
    # ~50 GB), replace with: pipe.to(VULKAN_DEVICE)
    enable_vulkan_sequential_cpu_offload(pipe)

    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    width, height = aspect_ratios["16:9"]

    prompt = (
        "A 20-year-old East Asian girl with delicate, charming features and large, "
        "bright brown eyes—expressive and lively, with a cheerful or subtly smiling "
        "expression. Her naturally wavy long hair is either loose or tied in twin "
        "ponytails. She has fair skin and light makeup accentuating her youthful "
        "freshness. She wears a modern, cute dress or relaxed outfit in bright, soft "
        "colors—lightweight fabric, minimalist cut. She stands indoors at an anime "
        "convention, surrounded by banners, posters, or stalls. Lighting is typical "
        "indoor illumination—no staged lighting—and the image resembles a casual iPhone "
        "snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm."
    )
    negative_prompt = (
        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，"
        "画面具有AI感。构图混乱。文字模糊，扭曲。"
    )

    generator = torch.Generator(device=VULKAN_DEVICE).manual_seed(42)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=generator,
    ).images[0]

    image.save("example.png")
    print("Saved example.png")


if __name__ == "__main__":
    main()
