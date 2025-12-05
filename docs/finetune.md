# VoxCPM Fine-tuning Guide

This guide covers how to fine-tune VoxCPM models with two approaches: full fine-tuning and LoRA fine-tuning.

### 🎓 SFT (Supervised Fine-Tuning)

Full fine-tuning updates all model parameters. Suitable for:
- 📊 Large, specialized datasets
- 🔄 Cases where significant behavior changes are needed

### ⚡ LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:
- 🎯 Trains only a small number of additional parameters
- 💾 Significantly reduces memory requirements and training time
- 🔀 Supports multiple LoRA adapters with hot-swapping



## Table of Contents

- [Data Preparation](#data-preparation)
- [Full Fine-tuning](#full-fine-tuning)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [Inference](#inference)
- [LoRA Hot-swapping](#lora-hot-swapping)
- [FAQ](#faq)

---

## Data Preparation

Training data should be prepared as a JSONL manifest file, with one sample per line:

```jsonl
{"audio": "path/to/audio1.wav", "text": "Transcript of audio 1."}
{"audio": "path/to/audio2.wav", "text": "Transcript of audio 2."}
{"audio": "path/to/audio3.wav", "text": "Optional duration field.", "duration": 3.5}
{"audio": "path/to/audio4.wav", "text": "Optional dataset_id for multi-dataset.", "dataset_id": 1}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `audio` | Path to audio file (absolute or relative) |
| `text` | Corresponding transcript |

### Optional Fields

| Field | Description |
|-------|-------------|
| `duration` | Audio duration in seconds (speeds up sample filtering) |
| `dataset_id` | Dataset ID for multi-dataset training (default: 0) |

### Requirements

- Audio format: WAV
- Sample rate: 16kHz for VoxCPM-0.5B, 44.1kHz for VoxCPM1.5
- Text: Transcript matching the audio content

See `examples/train_data_example.jsonl` for a complete example.

---

## Full Fine-tuning

Full fine-tuning updates all model parameters. Suitable for large datasets or when significant behavior changes are needed.

### Configuration

Create `conf/voxcpm_v1.5/voxcpm_finetune_all.yaml`:

```yaml
pretrained_path: /path/to/VoxCPM1.5/
train_manifest: /path/to/train.jsonl
val_manifest: ""

sample_rate: 44100
batch_size: 16
grad_accum_steps: 1
num_workers: 2
num_iters: 2000
log_interval: 10
valid_interval: 1000
save_interval: 1000

learning_rate: 0.00001   # Use smaller LR for full fine-tuning
weight_decay: 0.01
warmup_steps: 100
max_steps: 2000
max_batch_tokens: 8192

save_path: /path/to/checkpoints/finetune_all
tensorboard: /path/to/logs/finetune_all

lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
```

### Training

```bash
# Single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_all.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_all.yaml
```

### Checkpoint Structure

Full fine-tuning saves a complete model directory that can be loaded directly:

```
checkpoints/finetune_all/
└── step_0002000/
    ├── model.safetensors     # Model weights (excluding audio_vae)
    ├── config.json            # Model config
    ├── audiovae.pth           # Audio VAE weights
    ├── tokenizer.json         # Tokenizer
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── optimizer.pth
    └── scheduler.pth
```

---

## LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that trains only a small number of additional parameters, significantly reducing memory requirements.

### Configuration

Create `conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml`:

```yaml
pretrained_path: /path/to/VoxCPM1.5/
train_manifest: /path/to/train.jsonl
val_manifest: ""

sample_rate: 44100
batch_size: 16
grad_accum_steps: 1
num_workers: 2
num_iters: 2000
log_interval: 10
valid_interval: 1000
save_interval: 1000

learning_rate: 0.0001    # LoRA can use larger LR
weight_decay: 0.01
warmup_steps: 100
max_steps: 2000
max_batch_tokens: 8192

save_path: /path/to/checkpoints/finetune_lora
tensorboard: /path/to/logs/finetune_lora

lambdas:
  loss/diff: 1.0
  loss/stop: 1.0

# LoRA configuration
lora:
  enable_lm: true        # Apply LoRA to Language Model
  enable_dit: true       # Apply LoRA to Diffusion Transformer
  enable_proj: false     # Apply LoRA to projection layers (optional)
  
  r: 32                  # LoRA rank (higher = more capacity)
  alpha: 16              # LoRA alpha, scaling = alpha / r
  dropout: 0.0
  
  # Target modules
  target_modules_lm: ["q_proj", "v_proj", "k_proj", "o_proj"]
  target_modules_dit: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### LoRA Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `enable_lm` | Apply LoRA to LM (language model) | `true` |
| `enable_dit` | Apply LoRA to DiT (diffusion model) | `true` (required for voice cloning) |
| `r` | LoRA rank (higher = more capacity) | 16-64 |
| `alpha` | Scaling factor, `scaling = alpha / r` | Usually `r/2` or `r` |
| `target_modules_*` | Layer names to add LoRA | attention layers |

### Training

```bash
# Single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml
```

### Checkpoint Structure

LoRA training saves only LoRA parameters:

```
checkpoints/finetune_lora/
└── step_0002000/
    ├── lora_weights.safetensors    # Only lora_A, lora_B parameters
    ├── optimizer.pth
    └── scheduler.pth
```

---

## Inference

### Full Fine-tuning Inference

The checkpoint directory is a complete model, load it directly:

```bash
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir /path/to/checkpoints/finetune_all/step_0002000 \
    --text "Hello, this is the fine-tuned model." \
    --output output.wav
```

With voice cloning:

```bash
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir /path/to/checkpoints/finetune_all/step_0002000 \
    --text "This is voice cloning result." \
    --prompt_audio /path/to/reference.wav \
    --prompt_text "Reference audio transcript" \
    --output cloned_output.wav
```

### LoRA Inference

LoRA inference requires the training config (for LoRA structure) and LoRA checkpoint:

```bash
python scripts/test_voxcpm_lora_infer.py \
    --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml \
    --lora_ckpt /path/to/checkpoints/finetune_lora/step_0002000 \
    --text "Hello, this is LoRA fine-tuned result." \
    --output lora_output.wav
```

With voice cloning:

```bash
python scripts/test_voxcpm_lora_infer.py \
    --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml \
    --lora_ckpt /path/to/checkpoints/finetune_lora/step_0002000 \
    --text "This is voice cloning with LoRA." \
    --prompt_audio /path/to/reference.wav \
    --prompt_text "Reference audio transcript" \
    --output cloned_output.wav
```

---

## LoRA Hot-swapping

LoRA supports dynamic loading, unloading, and switching at inference time without reloading the entire model.

### API Reference

```python
from voxcpm.model import VoxCPMModel
from voxcpm.model.voxcpm import LoRAConfig

# 1. Load model with LoRA structure
lora_cfg = LoRAConfig(
    enable_lm=True, 
    enable_dit=True, 
    r=32, 
    alpha=16,
    target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
    target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model = VoxCPMModel.from_local(
    pretrained_path,
    optimize=True,       # Enable torch.compile acceleration
    lora_config=lora_cfg
)

# 2. Load LoRA weights (works after torch.compile)
loaded, skipped = model.load_lora_weights("/path/to/lora_checkpoint")
print(f"Loaded {len(loaded)} params, skipped {len(skipped)}")

# 3. Disable LoRA (use base model only)
model.set_lora_enabled(False)

# 4. Re-enable LoRA
model.set_lora_enabled(True)

# 5. Unload LoRA (reset weights to zero)
model.reset_lora_weights()

# 6. Hot-swap to another LoRA
model.load_lora_weights("/path/to/another_lora_checkpoint")

# 7. Get current LoRA weights
lora_state = model.get_lora_state_dict()
```

### Method Reference

| Method | Description | torch.compile Compatible |
|--------|-------------|--------------------------|
| `load_lora_weights(path)` | Load LoRA weights from file | ✅ |
| `set_lora_enabled(bool)` | Enable/disable LoRA | ✅ |
| `reset_lora_weights()` | Reset LoRA weights to initial values | ✅ |
| `get_lora_state_dict()` | Get current LoRA weights | ✅ |

---

## FAQ

### 1. Out of Memory (OOM)

- Increase `grad_accum_steps` (gradient accumulation)
- Decrease `batch_size`
- Use LoRA fine-tuning instead of full fine-tuning
- Decrease `max_batch_tokens` to filter long samples

### 2. Poor LoRA Performance

- Increase `r` (LoRA rank)
- Adjust `alpha` (try `alpha = r/2` or `alpha = r`)
- Ensure `enable_dit: true` (required for voice cloning)
- Increase training steps
- Add more target modules

### 3. Training Not Converging

- Decrease `learning_rate`
- Increase `warmup_steps`
- Check data quality

### 4. LoRA Not Taking Effect at Inference

- Ensure inference config matches training config LoRA parameters
- Check `load_lora_weights` return value - `skipped_keys` should be empty
- Verify `set_lora_enabled(True)` is called

### 5. Checkpoint Loading Errors

- Full fine-tuning: checkpoint directory should contain `model.safetensors`(or `pytorch_model.bin`), `config.json`, `audiovae.pth`
- LoRA: checkpoint directory should contain `lora_weights.safetensors` (or `lora_weights.ckpt`)
