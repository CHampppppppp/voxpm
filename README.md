# 🎙️ VoxCPM: 无需 Tokenizer 的上下文感知语音生成与高保真声音克隆模型

[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub-blue)](https://github.com/OpenBMB/VoxCPM/) [![Technical Report](https://img.shields.io/badge/Technical%20Report-Arxiv-red)](https://arxiv.org/abs/2509.24650)[![Live Playground](https://img.shields.io/badge/Live%20PlayGround-Demo-orange)](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo) [![Samples](https://img.shields.io/badge/Audio%20Samples-Page-green)](https://openbmb.github.io/VoxCPM-demopage)

#### VoxCPM1.5 模型权重

 [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OpenBMB-yellow)](https://huggingface.co/openbmb/VoxCPM1.5) [![ModelScope](https://img.shields.io/badge/ModelScope-OpenBMB-purple)](https://modelscope.cn/models/OpenBMB/VoxCPM1.5)  

<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="40%">
</div>

<div align="center">

👋 在 [微信](assets/wechat.png) 上联系我们

</div>

## 最新动态
* [2025.12.05] 🎉 🎉 🎉 我们开源了 VoxCPM1.5 [权重](https://huggingface.co/openbmb/VoxCPM1.5)！模型现在支持全参数微调和高效的 LoRA 微调，助您打造专属模型。详情请见 [发布说明](docs/release_note.md)。
* [2025.09.30] 🔥 🔥 🔥 我们发布了 VoxCPM [技术报告](https://arxiv.org/abs/2509.24650)！
* [2025.09.16] 🔥 🔥 🔥 我们开源了 VoxCPM-0.5B [权重](https://huggingface.co/openbmb/VoxCPM-0.5B)！
* [2025.09.16] 🎉 🎉 🎉 我们提供了 VoxCPM-0.5B 的 [Gradio 演示](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo)，快来试用吧！

## 模型简介

VoxCPM 是一款新颖的无需 Tokenizer 的文本转语音（TTS）系统，重新定义了语音合成的真实感。通过在连续空间中对语音进行建模，它克服了离散 Tokenization 的局限性，并实现了两大核心能力：上下文感知的语音生成和高保真的零样本声音克隆。

与将语音转换为离散 Token 的主流方法不同，VoxCPM 采用端到端的扩散自回归架构，直接从文本生成连续的语音表征。基于 [MiniCPM-4](https://huggingface.co/openbmb/MiniCPM4-0.5B) 骨干网络，它通过分层语言建模和 FSQ 约束实现了隐式的语义-声学解耦，极大地增强了表现力和生成稳定性。

<div align="center">
  <img src="assets/voxcpm_model.png" alt="VoxCPM Model Architecture" width="90%">
</div>

###  🚀 核心特性
- **上下文感知的高表现力语音生成** - VoxCPM 能够理解文本内容，推断并生成适当的韵律，提供极具表现力和自然流畅的语音。它能根据内容自发调整说话风格，生成与 180 万小时双语语料库训练结果高度契合的语音表达。
- **高保真声音克隆** - 仅需一段简短的参考音频，VoxCPM 即可执行精准的零样本声音克隆，不仅能捕捉说话者的音色，还能还原口音、情感基调、节奏和语速等细微特征，创造出忠实且自然的复刻。
- **高效合成** - VoxCPM 支持流式合成，在消费级 NVIDIA RTX 4090 GPU 上，实时率（RTF）低至 0.17，使实时应用成为可能。

### 📦 模型版本
详情请见 [发布说明](docs/release_note.md)
- **VoxCPM1.5** (最新): 
  - 模型参数: 800M
  - AudioVAE 采样率: 44100
  - LM 骨干网络 Token 率: 6.25Hz (patch-size=4)
  - 单卡 NVIDIA-RTX 4090 RTF: ~0.15

- **VoxCPM-0.5B** (初代):
  - 模型参数: 640M
  - AudioVAE 采样率: 16000
  - LM 骨干网络 Token 率: 12.5Hz (patch-size=2)
  - 单卡 NVIDIA-RTX 4090 RTF: 0.17

## 快速开始

### 🔧 安装
``` sh
pip install voxcpm
```

### 1. 模型下载 (可选)
默认情况下，首次运行脚本时会自动下载模型，但您也可以提前下载。
- 下载 VoxCPM1.5
    ```python
    from huggingface_hub import snapshot_download
    snapshot_download("openbmb/VoxCPM1.5")
    ```

- 或下载 VoxCPM-0.5B
    ```python
    from huggingface_hub import snapshot_download
    snapshot_download("openbmb/VoxCPM-0.5B")
    ```
- 下载 ZipEnhancer 和 SenseVoice-Small。我们使用 ZipEnhancer 增强语音提示，并在 Web 演示中使用 SenseVoice-Small 进行语音提示 ASR。
    ```python
    from modelscope import snapshot_download
    snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base')
    snapshot_download('iic/SenseVoiceSmall')
    ```

### 2. Python 代码调用
```python
import soundfile as sf
import numpy as np
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")

# 非流式生成
wav = model.generate(
    text="VoxCPM 是 ModelBest 推出的创新端到端 TTS 模型，旨在生成极具表现力的语音。",
    prompt_wav_path=None,      # 可选：用于声音克隆的参考音频路径
    prompt_text=None,          # 可选：参考文本
    cfg_value=2.0,             # LocDiT 的 LM 引导值，越高越贴合提示，但可能导致质量下降
    inference_timesteps=10,    # LocDiT 推理步数，越高效果越好，越低速度越快
    normalize=False,           # 启用外部 TN 工具，但会禁用原生原始文本支持
    denoise=False,             # 启用外部降噪工具，可能会导致失真并将采样率限制在 16kHz
    retry_badcase=True,        # 启用坏例重试模式（不可阻挡）
    retry_badcase_max_times=3,  # 最大重试次数
    retry_badcase_ratio_threshold=6.0, # 坏例检测的最大长度限制（简单但有效），语速较慢时可调整
)

sf.write("output.wav", wav, model.tts_model.sample_rate)
print("saved: output.wav")

# 流式生成
chunks = []
for chunk in model.generate_streaming(
    text = "使用 VoxCPM 进行流式语音合成非常简单！",
    # 支持与上述相同的参数
):
    chunks.append(chunk)
wav = np.concatenate(chunks)

sf.write("output_streaming.wav", wav, model.tts_model.sample_rate)
print("saved: output_streaming.wav")
```

### 3. 命令行工具 (CLI)

安装后，入口点为 `voxcpm` (或使用 `python -m voxcpm.cli`)。

```bash
# 1) 直接合成 (单条文本)
voxcpm --text "VoxCPM 是 ModelBest 推出的创新端到端 TTS 模型。" --output out.wav

# 2) 声音克隆 (参考音频 + 文本)
voxcpm --text "VoxCPM 是 ModelBest 推出的创新端到端 TTS 模型。" \
  --prompt-audio path/to/voice.wav \
  --prompt-text "参考音频的文本内容" \
  --output out.wav \
  # --denoise

# (可选) 声音克隆 (参考音频 + 文本文件)
voxcpm --text "VoxCPM 是 ModelBest 推出的创新端到端 TTS 模型。" \
  --prompt-audio path/to/voice.wav \
  --prompt-file "/path/to/text-file" \
  --output out.wav \
  # --denoise

# 3) 批量处理 (每行一条文本)
voxcpm --input examples/input.txt --output-dir outs
# (可选) 批量 + 克隆
voxcpm --input examples/input.txt --output-dir outs \
  --prompt-audio path/to/voice.wav \
  --prompt-text "参考音频的文本内容" \
  # --denoise

# 4) 推理参数 (质量/速度)
voxcpm --text "..." --output out.wav \
  --cfg-value 2.0 --inference-timesteps 10 --normalize

# 5) 模型加载
# 优先使用本地路径
voxcpm --text "..." --output out.wav --model-path /path/to/VoxCPM_model_dir
# 或从 Hugging Face 加载 (自动下载/缓存)
voxcpm --text "..." --output out.wav \
  --hf-model-id openbmb/VoxCPM1.5 --cache-dir ~/.cache/huggingface --local-files-only

# 6) 降噪器控制
voxcpm --text "..." --output out.wav \
  --no-denoiser --zipenhancer-path iic/speech_zipenhancer_ans_multiloss_16k_base

# 7) 帮助
voxcpm --help
python -m voxcpm.cli --help
```

### 4. Web 演示 (Gradio)

运行以下命令启动可视化界面，支持声音克隆、参考音频录制与自动 ASR 识别：

```bash
python app.py
```

### 5. WebSocket API 服务

VoxCPM 提供基于 WebSocket 的高性能异步 API 服务，支持并发请求与声音克隆，仅保留 TTS 能力。

**启动服务**:
```bash
python api.py
```
默认监听 `0.0.0.0:8080`。

**接口功能**:
*   `ws://host:port/ws/generate`: 标准语音合成接口。
    *   **特性**: 支持传入 `prompt_wav_path` 和 `prompt_text` 进行克隆，不再自动识别文本。
*   `ws://host:port/ws/health`: WebSocket 健康检查。
*   `ws://host:port/ws/models`: WebSocket 获取模型信息。

**HTTP 接口**:
*   `http://host:port/health`: HTTP 健康检查。
*   `http://host:port/models`: 获取模型信息。
*   `http://host:port/generate`: 生成 WAV 音频。

**请求示例**:
```json
{
    "text": "你好，这是一段测试语音。",
    "prompt_wav_path": "/path/to/voice.wav",
    "cfg_value": 2.0,
    "inference_timesteps": 25,
    "denoise": true
}
```

### 6. 微调 (Fine-tuning)

VoxCPM1.5 支持全量微调 (SFT) 和 LoRA 微调，允许您基于自有数据训练个性化语音模型。详细说明请参考 [微调指南](docs/finetune.md)。

**快速开始:**
```bash
# 全量微调
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v1.5/voxcpm_finetune_all.yaml

# LoRA 微调
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml
```

## � 文档

- **[使用指南](docs/usage_guide.md)** - 关于如何有效使用 VoxCPM 的详细指南，包括文本输入模式、声音克隆技巧和参数调优
- **[微调指南](docs/finetune.md)** - 使用 SFT 和 LoRA 微调 VoxCPM 模型的完整指南
- **[发布说明](docs/release_note.md)** - 版本历史和更新
- **[性能基准](docs/performance.md)** - 公共基准测试的详细性能对比

---
