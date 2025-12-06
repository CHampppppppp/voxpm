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

from voxcpm.core import VoxCPM
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
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable text normalization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load YAML config
    cfg = load_yaml_config(args.config_path)
    pretrained_path = cfg["pretrained_path"]
    lora_cfg_dict = cfg.get("lora", {}) or {}
    lora_cfg = LoRAConfig(**lora_cfg_dict) if lora_cfg_dict else None

    # 2. Check LoRA checkpoint
    ckpt_dir = args.lora_ckpt
    if not Path(ckpt_dir).exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_dir}")

    # 3. Load model with LoRA (no denoiser)
    print(f"[1/2] Loading model with LoRA: {pretrained_path}")
    print(f"      LoRA weights: {ckpt_dir}")
    model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=ckpt_dir,
    )

    # 4. Synthesize audio
    prompt_wav_path = args.prompt_audio if args.prompt_audio else None
    prompt_text = args.prompt_text if args.prompt_text else None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/2] Starting synthesis tests...")
    
    # === Test 1: With LoRA ===
    print(f"\n  [Test 1] Synthesize with LoRA...")
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    lora_output = out_path.with_stem(out_path.stem + "_with_lora")
    sf.write(str(lora_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {lora_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 2: Disable LoRA (via set_lora_enabled) ===
    print(f"\n  [Test 2] Disable LoRA (set_lora_enabled=False)...")
    model.set_lora_enabled(False)
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    disabled_output = out_path.with_stem(out_path.stem + "_lora_disabled")
    sf.write(str(disabled_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {disabled_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 3: Re-enable LoRA ===
    print(f"\n  [Test 3] Re-enable LoRA (set_lora_enabled=True)...")
    model.set_lora_enabled(True)
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    reenabled_output = out_path.with_stem(out_path.stem + "_lora_reenabled")
    sf.write(str(reenabled_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {reenabled_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 4: Unload LoRA (reset_lora_weights) ===
    print(f"\n  [Test 4] Unload LoRA (unload_lora)...")
    model.unload_lora()
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    reset_output = out_path.with_stem(out_path.stem + "_lora_reset")
    sf.write(str(reset_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {reset_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 5: Hot-reload LoRA (load_lora) ===
    print(f"\n  [Test 5] Hot-reload LoRA (load_lora)...")
    loaded, skipped = model.load_lora(str(ckpt_dir))
    print(f"           Reloaded {len(loaded)} parameters")
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    reload_output = out_path.with_stem(out_path.stem + "_lora_reloaded")
    sf.write(str(reload_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {reload_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    print(f"\n[Done] All tests completed!")
    print(f"  - with_lora:      {lora_output}")
    print(f"  - lora_disabled:  {disabled_output}")
    print(f"  - lora_reenabled: {reenabled_output}")
    print(f"  - lora_reset:     {reset_output}")
    print(f"  - lora_reloaded:  {reload_output}")


if __name__ == "__main__":
    main()
