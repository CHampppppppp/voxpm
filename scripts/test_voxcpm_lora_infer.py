#!/usr/bin/env python3
"""
LoRA inference test script.

Usage:

    python scripts/test_voxcpm_lora_infer.py \
        --config_path conf/voxcpm/voxcpm_finetune_test.yaml \
        --lora_ckpt checkpoints/step_0002000 \
        --text "Hello, this is LoRA finetuned result." \
        --output lora_test.wav

With voice cloning:

    python scripts/test_voxcpm_lora_infer.py \
        --config_path conf/voxcpm/voxcpm_finetune_test.yaml \
        --lora_ckpt checkpoints/step_0002000 \
        --text "This is voice cloning result." \
        --prompt_audio path/to/ref.wav \
        --prompt_text "Reference audio transcript" \
        --output lora_clone.wav
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from voxcpm.model import VoxCPMModel
from voxcpm.model.voxcpm import LoRAConfig
from voxcpm.training.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser("VoxCPM LoRA inference test")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Training YAML config path (contains pretrained_path and lora config)",
    )
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        required=True,
        help="LoRA checkpoint directory (contains lora_weights.ckpt with lora_A/lora_B only)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Target text to synthesize",
    )
    parser.add_argument(
        "--prompt_audio",
        type=str,
        default="",
        help="Optional: reference audio path for voice cloning",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="",
        help="Optional: transcript of reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lora_test.wav",
        help="Output wav file path",
    )
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="CFG scale (default: 2.0)",
    )
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=10,
        help="Diffusion inference steps (default: 10)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=600,
        help="Max generation steps",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load YAML config
    cfg = load_yaml_config(args.config_path)
    pretrained_path = cfg["pretrained_path"]
    lora_cfg_dict = cfg.get("lora", {}) or {}
    lora_cfg = LoRAConfig(**lora_cfg_dict) if lora_cfg_dict else None

    # 2. Load base model (with LoRA structure and torch.compile)
    print(f"[1/3] Loading base model: {pretrained_path}")
    model = VoxCPMModel.from_local(
        pretrained_path,
        optimize=True,  # compile first, load_lora_weights uses named_parameters for compatibility
        training=False,
        lora_config=lora_cfg,
    )
    
    # Debug: check DiT param paths after compile
    dit_params = [n for n, _ in model.named_parameters() if 'feat_decoder' in n and 'lora' in n]
    print(f"[DEBUG] DiT LoRA param paths after compile (first 3): {dit_params[:3]}")

    # 3. Load LoRA weights (works after compile)
    ckpt_dir = Path(args.lora_ckpt)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_dir}")
    
    print(f"[2/3] Loading LoRA weights: {ckpt_dir}")
    loaded, skipped = model.load_lora_weights(str(ckpt_dir))
    print(f"       Loaded {len(loaded)} parameters")
    if skipped:
        print(f"[WARNING] Skipped {len(skipped)} parameters")
        print(f"       Skipped keys (first 5): {skipped[:5]}")

    # 4. Synthesize audio
    prompt_wav_path = args.prompt_audio or ""
    prompt_text = args.prompt_text or ""
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/3] Starting synthesis tests...")
    
    # === Test 1: With LoRA ===
    print(f"\n  [Test 1] Synthesize with LoRA...")
    with torch.inference_mode():
        audio = model.generate(
            target_text=args.text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            max_len=args.max_len,
            inference_timesteps=args.inference_timesteps,
            cfg_value=args.cfg_value,
        )
    audio_np = audio.squeeze(0).cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
    lora_output = out_path.with_stem(out_path.stem + "_with_lora")
    sf.write(str(lora_output), audio_np, model.sample_rate)
    print(f"           Saved: {lora_output}, duration: {len(audio_np) / model.sample_rate:.2f}s")

    # === Test 2: Disable LoRA (via set_lora_enabled) ===
    print(f"\n  [Test 2] Disable LoRA (set_lora_enabled=False)...")
    model.set_lora_enabled(False)
    with torch.inference_mode():
        audio = model.generate(
            target_text=args.text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            max_len=args.max_len,
            inference_timesteps=args.inference_timesteps,
            cfg_value=args.cfg_value,
        )
    audio_np = audio.squeeze(0).cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
    disabled_output = out_path.with_stem(out_path.stem + "_lora_disabled")
    sf.write(str(disabled_output), audio_np, model.sample_rate)
    print(f"           Saved: {disabled_output}, duration: {len(audio_np) / model.sample_rate:.2f}s")

    # === Test 3: Re-enable LoRA ===
    print(f"\n  [Test 3] Re-enable LoRA (set_lora_enabled=True)...")
    model.set_lora_enabled(True)
    with torch.inference_mode():
        audio = model.generate(
            target_text=args.text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            max_len=args.max_len,
            inference_timesteps=args.inference_timesteps,
            cfg_value=args.cfg_value,
        )
    audio_np = audio.squeeze(0).cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
    reenabled_output = out_path.with_stem(out_path.stem + "_lora_reenabled")
    sf.write(str(reenabled_output), audio_np, model.sample_rate)
    print(f"           Saved: {reenabled_output}, duration: {len(audio_np) / model.sample_rate:.2f}s")

    # === Test 4: Unload LoRA (reset_lora_weights) ===
    print(f"\n  [Test 4] Unload LoRA (reset_lora_weights)...")
    model.reset_lora_weights()
    with torch.inference_mode():
        audio = model.generate(
            target_text=args.text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            max_len=args.max_len,
            inference_timesteps=args.inference_timesteps,
            cfg_value=args.cfg_value,
        )
    audio_np = audio.squeeze(0).cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
    reset_output = out_path.with_stem(out_path.stem + "_lora_reset")
    sf.write(str(reset_output), audio_np, model.sample_rate)
    print(f"           Saved: {reset_output}, duration: {len(audio_np) / model.sample_rate:.2f}s")

    # === Test 5: Hot-reload LoRA (load_lora_weights) ===
    print(f"\n  [Test 5] Hot-reload LoRA (load_lora_weights)...")
    loaded, _ = model.load_lora_weights(str(ckpt_dir))
    print(f"           Reloaded {len(loaded)} parameters")
    with torch.inference_mode():
        audio = model.generate(
            target_text=args.text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            max_len=args.max_len,
            inference_timesteps=args.inference_timesteps,
            cfg_value=args.cfg_value,
        )
    audio_np = audio.squeeze(0).cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
    reload_output = out_path.with_stem(out_path.stem + "_lora_reloaded")
    sf.write(str(reload_output), audio_np, model.sample_rate)
    print(f"           Saved: {reload_output}, duration: {len(audio_np) / model.sample_rate:.2f}s")

    print(f"\n[Done] All tests completed!")
    print(f"  - with_lora:      {lora_output}")
    print(f"  - lora_disabled:  {disabled_output}")
    print(f"  - lora_reenabled: {reenabled_output}")
    print(f"  - lora_reset:     {reset_output}")
    print(f"  - lora_reloaded:  {reload_output}")


if __name__ == "__main__":
    main()


