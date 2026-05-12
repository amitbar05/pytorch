"""Train a tiny diffusion UNet on Vulkan — verifies the backend end-to-end.

Model:  UNet2DModel  (2.8 M params, defined in code — nothing to download)
VRAM:   ~500 MB  (fp32 params + grads + AdamW states + activations, batch 8)
Data:   synthetic random images — no dataset needed

The training loop follows standard DDPM:
  1. Sample clean image x0 and noise ε
  2. Sample timestep t; compute noisy image x_t = scheduler.add_noise(x0, ε, t)
  3. Predict ε̂ = model(x_t, t)
  4. Loss = MSE(ε̂, ε);  backward;  AdamW step

Noise schedule (add_noise) runs on CPU to stay out of the Vulkan dispatch path.
Everything else — forward, backward, optimizer — runs on Vulkan.

Usage:
    source .venv/bin/activate
    python train_diffusion.py
"""
import torch
import torch_vulkan  # noqa: F401 — registers the vulkan PrivateUse1 backend
from diffusers import UNet2DModel, DDPMScheduler

DEVICE = "vulkan:0"
BATCH = 8
IMG = 32
STEPS = 200
LR = 1e-4
LOG = 20


def main():
    if not torch_vulkan.is_available():
        raise RuntimeError("No Vulkan device found")

    # ── Model ────────────────────────────────────────────────────────────────
    # 2.8 M params; fp32 training VRAM ~43 MB (params+grad+AdamW), plus ~300 MB
    # activations at batch 8 → well under 6 GB.
    model = UNet2DModel(
        sample_size=IMG,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(DEVICE)

    # ── Scheduler (CPU — only used for add_noise preprocessing) ──────────────
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # ── Optimizer (AdamW; states live on Vulkan alongside params) ────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ── Training loop ────────────────────────────────────────────────────────
    for step in range(STEPS):
        # Noise schedule on CPU — keeps scheduler internals off the GPU path
        x0 = torch.randn(BATCH, 3, IMG, IMG)
        eps = torch.randn_like(x0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (BATCH,))
        x_t = scheduler.add_noise(x0, eps, t)

        # Move inputs to Vulkan
        x_t = x_t.to(DEVICE)
        eps_target = eps.to(DEVICE)
        t_vk = t.to(DEVICE)

        # Forward → loss → backward → optimizer step
        eps_pred = model(x_t, t_vk).sample
        loss = (eps_pred - eps_target).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % LOG == 0:
            print(f"step {step:4d}  loss={loss.item():.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
