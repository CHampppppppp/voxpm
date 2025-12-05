#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import contextlib
from typing import Dict, Optional

import argbind
import torch
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, will use pytorch format")

from voxcpm.model import VoxCPMModel
from voxcpm.model.voxcpm import LoRAConfig
from voxcpm.training import (
    Accelerator,
    BatchProcessor,
    TrainingTracker,
    build_dataloader,
    load_audio_text_datasets,
)


@argbind.bind(without_prefix=True)
def train(
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str = "",
    sample_rate: int = 16_000,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    num_workers: int = 2,
    num_iters: int = 100_000,
    log_interval: int = 100,
    valid_interval: int = 1_000,
    save_interval: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    max_batch_tokens: int = 0,
    save_path: str = "checkpoints",
    tensorboard: str = "",
    lambdas: Dict[str, float] = {"loss/diff": 1.0, "loss/stop": 1.0},
    lora: dict = None,
    config_path: str = "",
):
    _ = config_path
    accelerator = Accelerator(amp=True)

    save_dir = Path(save_path)
    tb_dir = Path(tensorboard) if tensorboard else save_dir / "logs"

    # Only create directories on rank 0 to avoid race conditions
    if accelerator.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.barrier()  # Wait for directory creation

    writer = SummaryWriter(log_dir=str(tb_dir)) if accelerator.rank == 0 else None
    tracker = TrainingTracker(writer=writer, log_file=str(save_dir / "train.log"), rank=accelerator.rank)

    base_model = VoxCPMModel.from_local(pretrained_path, optimize=False, training=True, lora_config=LoRAConfig(**lora) if lora else None)
    tokenizer = base_model.text_tokenizer

    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        sample_rate=sample_rate,
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    if val_ds is not None:
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    dataset_cnt = int(max(train_ds["dataset_id"])) + 1 if "dataset_id" in train_ds.column_names else 1
    num_train_samples = len(train_ds)

    # ------------------------------------------------------------------ #
    # Optional: filter samples by estimated token count to avoid OOM
    # Enabled when max_batch_tokens > 0:
    #   max_sample_len = max_batch_tokens // batch_size
    #   Samples exceeding this length will be dropped
    # ------------------------------------------------------------------ #
    if max_batch_tokens and max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and accelerator.rank == 0:
            tracker.print(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={max_batch_tokens})."
            )
        train_ds = train_ds.select(keep_indices)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )
    del base_model.audio_vae
    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()


    # Only print param info on rank 0 to avoid cluttered output
    if accelerator.rank == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine + warmup scheduler from transformers:
    # - num_warmup_steps: warmup steps
    # - num_training_steps: total training steps (outer step count)
    total_training_steps = max_steps if max_steps > 0 else num_iters
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Manual epoch management instead of itertools.cycle to support DistributedSampler.set_epoch()
    grad_accum_steps = max(int(grad_accum_steps), 1)
    data_epoch = 0
    train_iter = iter(train_loader)

    def get_next_batch():
        """Get next batch, handles epoch boundary and DistributedSampler."""
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            # Key: set DistributedSampler epoch to ensure different data order each epoch
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    with tracker.live():
        for step in range(num_iters):
            tracker.step = step
            optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation: accumulate gradients over micro-batches before optimizer step
            loss_dict = {}
            for micro_step in range(grad_accum_steps):
                batch = get_next_batch()
                processed = batch_processor(batch)

                # Only sync gradients on the last micro-batch
                # Use no_sync() for intermediate steps to reduce communication overhead
                is_last_micro_step = (micro_step == grad_accum_steps - 1)
                sync_context = contextlib.nullcontext() if is_last_micro_step else accelerator.no_sync()

                with sync_context:
                    with accelerator.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            processed["text_tokens"],
                            processed["text_mask"],
                            processed["audio_feats"],
                            processed["audio_mask"],
                            processed["loss_mask"],
                            processed["position_ids"],
                            processed["labels"],
                            progress=step / max(1, num_iters),
                        )

                    total_loss = 0.0
                    for key, value in outputs.items():
                        if key.startswith("loss/"):
                            weight = lambdas.get(key, 1.0)
                            loss_value = value * weight / grad_accum_steps
                            total_loss = total_loss + loss_value
                            # Record raw loss from last micro-batch for logging
                            loss_dict[key] = value.detach()

                    # Accumulate gradients (normalized by grad_accum_steps)
                    accelerator.backward(total_loss)

            # After all micro-batches, do unscale / grad_norm / step
            scaler = getattr(accelerator, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Use large max_norm to only compute grad_norm without actual clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=1e9)

            accelerator.step(optimizer)
            accelerator.update()
            scheduler.step()

            if step % log_interval == 0:
                loss_values = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
                loss_values["lr"] = float(optimizer.param_groups[0]["lr"])
                # Approximate epoch: seen samples / total samples (considering grad_accum and batch_size)
                epoch = (step * grad_accum_steps * batch_size) / max(1, num_train_samples)
                loss_values["epoch"] = float(epoch)
                loss_values["grad_norm"] = float(grad_norm)
                tracker.log_metrics(loss_values, split="train")

            if val_loader is not None and step % valid_interval == 0 and step != 0:
                validate(model, val_loader, batch_processor, accelerator, tracker, lambdas)

            if step % save_interval == 0 and accelerator.rank == 0:
                save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path)

    if accelerator.rank == 0:
        save_checkpoint(model, optimizer, scheduler, save_dir, num_iters, pretrained_path)
    if writer:
        writer.close()


def validate(model, val_loader, batch_processor, accelerator, tracker, lambdas):
    model.eval()
    losses = []
    num_batches = 0
    max_val_batches = 10

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_val_batches:
                break
            processed = batch_processor(batch)
            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                    sample_generate=False,
                )
            total = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    total += lambdas.get(key, 1.0) * value
            losses.append(total.detach())
            num_batches += 1

    if losses:
        mean_loss = torch.stack(losses).mean()
        # All-reduce validation loss across processes for global average
        accelerator.all_reduce(mean_loss)
        tracker.log_metrics({"loss": mean_loss.item()}, split="val")
    model.train()


def save_checkpoint(model, optimizer, scheduler, save_dir: Path, step: int, pretrained_path: str = None):
    """
    Save checkpoint with different strategies for full finetune vs LoRA:
    - Full finetune: save non-vae weights to model.safetensors (or pytorch_model.bin if safetensors unavailable)
    - LoRA: save only lora weights to lora_weights.safetensors (or lora_weights.ckpt if safetensors unavailable)
    """
    import shutil
    
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = "latest" if step == 0 else f"step_{step:07d}"
    folder = save_dir / tag
    folder.mkdir(parents=True, exist_ok=True)
    
    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()
    lora_cfg = unwrapped.lora_config
    
    if lora_cfg is not None:
        # LoRA finetune: save only lora_A/lora_B weights
        state_dict = {k: v for k, v in full_state.items() if "lora_" in k}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "lora_weights.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "lora_weights.ckpt")
    else:
        # Full finetune: save non-vae weights to model.safetensors
        state_dict = {k: v for k, v in full_state.items() if not k.startswith("audio_vae.")}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "model.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "pytorch_model.bin")
        
        # Copy config files from pretrained path
        if pretrained_path:
            pretrained_dir = Path(pretrained_path)
            files_to_copy = ["config.json", "audiovae.pth", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
            for fname in files_to_copy:
                src = pretrained_dir / fname
                if src.exists():
                    shutil.copy2(src, folder / fname)
    
    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")


if __name__ == "__main__":
    from voxcpm.training.config import load_yaml_config

    args = argbind.parse_args()
    config_file = args.get("config_path")
    # If YAML config provided, use YAML args to call train
    if config_file:
        yaml_args = load_yaml_config(config_file)
        train(**yaml_args)
    else:
        # Otherwise use command line args (parsed by argbind)
        with argbind.scope(args):
            train()

