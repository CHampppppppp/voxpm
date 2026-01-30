"""
VoxCPM TTS WebSocket Service
============================
提供高性能的异步 TTS WebSocket 服务，支持声音克隆和多客户端并发调用。

Usage:
    python api.py

WebSocket Endpoints:
    ws://host:port/ws/generate - 语音合成
    ws://host:port/ws/health - 健康检查
    ws://host:port/ws/models - 获取模型信息
"""

import os
import sys
import torch
import numpy as np
import voxcpm
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import tempfile
import threading
from contextlib import asynccontextmanager
import json

# 配置
MODEL_PATH = os.environ.get("VOXCPM_MODEL_PATH", "/home/zju/VoxCPM/models/openbmb__VoxCPM1.5")
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8080))
# 默认参考声音配置
DEFAULT_PROMPT_WAV_PATH = os.environ.get("DEFAULT_PROMPT_WAV_PATH", "")
DEFAULT_PROMPT_TEXT = os.environ.get("DEFAULT_PROMPT_TEXT", "")


# 全局模型实例（懒加载）
_tts_model: Optional[voxcpm.VoxCPM] = None
_model_lock = threading.Lock()


def get_model() -> voxcpm.VoxCPM:
    """获取或初始化模型（线程安全懒加载）"""
    global _tts_model
    if _tts_model is None:
        with _model_lock:
            if _tts_model is None:
                print(f"Loading model from: {MODEL_PATH}", file=sys.stderr)
                _tts_model = voxcpm.VoxCPM(
                    voxcpm_model_path=MODEL_PATH,
                    enable_denoiser=True,  # 启用降噪以提升克隆质量
                    optimize=False,
                )
                print("Model loaded successfully!", file=sys.stderr)
    return _tts_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时在后台线程预加载模型，避免阻塞服务启动
    # 这样 /health 接口可以立即响应，而模型在后台加载
    print("Starting background model loading...", file=sys.stderr)
    threading.Thread(target=get_model, daemon=True).start()
    yield
    # 关闭时清理
    pass


app = FastAPI(
    title="VoxCPM TTS WebSocket Service",
    description="基于 VoxCPM 的本地语音合成服务，支持声音克隆",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "VoxCPM TTS WebSocket Service",
        "docs_url": "/docs",
        "endpoints": {
            "http_generate": "/generate",
            "http_health": "/health",
            "http_models": "/models",
            "ws_generate": "/ws/generate",
            "ws_health": "/ws/health",
            "ws_models": "/ws/models"
        }
    }


# ============ 请求模型 ============

class TTSRequest(BaseModel):
    """语音合成请求"""
    text: str
    prompt_wav_path: Optional[str] = None
    prompt_text: Optional[str] = None
    cfg_value: float = 2.0
    inference_timesteps: int = 25
    normalize: bool = False
    denoise: bool = False
    seed: Optional[int] = None  # 新增: 随机种子





def set_seed(seed: int):
    """设置随机种子以保证生成结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WebSocketManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        await websocket.send_json(data)
    
    async def send_bytes(self, websocket: WebSocket, data: bytes):
        await websocket.send_bytes(data)
    
    async def send_error(self, websocket: WebSocket, message: str):
        await self.send_json(websocket, {"status": "error", "message": message})

ws_manager = WebSocketManager()


# ============ HTTP Endpoints ============

@app.get("/health")
async def health_check():
    """健康检查接口"""
    is_loaded = _tts_model is not None
    return {
        "status": "ok", 
        "message": "TTS service is running", 
        "model_loaded": is_loaded
    }


@app.get("/models")
async def get_models():
    """获取模型信息"""
    model = get_model()
    return {
        "model_path": MODEL_PATH,
        "sample_rate": model.tts_model.sample_rate,
        "device": str(model.tts_model.device),
        "dtype": str(model.tts_model.config.dtype),
    }


@app.post("/generate")
async def http_generate(request: TTSRequest):
    """
    HTTP 语音合成接口
    返回 WAV 格式音频文件
    """
    # 验证必填参数
    if not request.text.strip():
        return {"status": "error", "message": "Text cannot be empty"}
    
    # 处理参考音频参数
    prompt_wav_path = request.prompt_wav_path
    prompt_text = request.prompt_text

    # 验证参考音频文件是否存在（如果提供了的话）
    if prompt_wav_path:
        if not os.path.exists(prompt_wav_path):
            return {"status": "error", "message": f"Prompt audio file not found: {prompt_wav_path}"}

    # 处理默认参考音频
    if prompt_wav_path is None and prompt_text is None:
        if DEFAULT_PROMPT_WAV_PATH and os.path.exists(DEFAULT_PROMPT_WAV_PATH):
            prompt_wav_path = DEFAULT_PROMPT_WAV_PATH
            prompt_text = DEFAULT_PROMPT_TEXT if DEFAULT_PROMPT_TEXT else None

    # 设置随机种子
    if request.seed is not None:
        set_seed(request.seed)

    # 生成语音
    try:
        model = get_model()
        wav = model.generate(
            text=request.text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=request.cfg_value,
            inference_timesteps=request.inference_timesteps,
            normalize=request.normalize,
            denoise=request.denoise,
        )

        # 保存到临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_tensor = torch.from_numpy(wav).unsqueeze(0)
        torchaudio.save(temp_file.name, audio_tensor, sample_rate=model.tts_model.sample_rate, format="wav")
        temp_file.close()

        # 返回文件流
        from fastapi.responses import FileResponse
        from starlette.background import BackgroundTask

        def cleanup():
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

        return FileResponse(
            temp_file.name, 
            media_type="audio/wav", 
            filename="output.wav",
            background=BackgroundTask(cleanup)
        )

    except Exception as e:
        return {"status": "error", "message": f"Generation failed: {str(e)}"}


@app.websocket("/")
async def websocket_root(websocket: WebSocket):
    """
    Handle WebSocket connection to root path to avoid 403 error.
    Sends service info and closes connection.
    """
    await websocket.accept()
    await websocket.send_json({
        "status": "connected",
        "message": "VoxCPM TTS WebSocket Service",
        "endpoints": {
            "generate": "/ws/generate",
            "health": "/ws/health",
            "models": "/ws/models"
        }
    })
    await websocket.close()





# ============ WebSocket 辅助端点 ============

@app.websocket("/ws/health")
async def websocket_health(websocket: WebSocket):
    """
    WebSocket 健康检查
    """
    await ws_manager.connect(websocket)
    try:
        await ws_manager.send_json(websocket, {
            "status": "ok",
            "message": "TTS WebSocket service is running"
        })
    except Exception:
        pass
    finally:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/models")
async def websocket_models(websocket: WebSocket):
    """
    WebSocket 获取模型信息
    """
    await ws_manager.connect(websocket)
    try:
        model = get_model()
        await ws_manager.send_json(websocket, {
            "model_path": MODEL_PATH,
            "sample_rate": model.tts_model.sample_rate,
            "device": model.tts_model.device,
            "dtype": model.tts_model.config.dtype,
        })
    except Exception as e:
        await ws_manager.send_error(websocket, f"Failed to get model info: {str(e)}")
    finally:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket 语音合成接口
    
    客户端发送 JSON 参数，服务器返回二进制音频数据
    
    发送格式:
    {
        "text": "要合成的文本",
        "prompt_wav_path": "/path/to/reference.wav",  // 可选
        "prompt_text": "参考音频对应的文本",           // 可选
        "cfg_value": 2.0,                              // 可选，默认 2.0 (推荐)
        "inference_timesteps": 25,                     // 可选，默认 25 (更高质量可设为 30-50)
        "normalize": false,                            // 可选，默认 false
        "denoise": false,                               // 可选，默认 false (设为 false 可保留呼吸声增加拟人感)
        "stream": true,                                // 可选，默认 true。若为 true，则分块发送 PCM 音频数据
        "seed": 12345                                  // 可选，设置随机种子以固定音色
    }
    
    接收格式:
    - 成功: 
        - stream=false: 先发送完整 WAV 音频 bytes，然后发送 {"status": "success", "audio_size": N}
        - stream=true:  连续发送 PCM 音频 bytes 块，最后发送 {"status": "success", "audio_size": N, "mode": "stream"}
    - 失败: 发送 {"status": "error", "message": "错误信息"}
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            try:
                # 接收 JSON 数据
                data = await websocket.receive_json()
                
                # 验证必填参数
                if not data.get("text", "").strip():
                    await ws_manager.send_error(websocket, "Text cannot be empty")
                    continue
                
                # 处理参考音频参数
                prompt_wav_path = data.get("prompt_wav_path")
                prompt_text = data.get("prompt_text")
                stream = data.get("stream", True)

                # 验证参考音频文件是否存在（如果提供了的话）
                if prompt_wav_path:
                    if not os.path.exists(prompt_wav_path):
                        await ws_manager.send_error(websocket, f"Prompt audio file not found: {prompt_wav_path}")
                        continue

                # 处理默认参考音频（当用户没有提供任何参数时）
                if prompt_wav_path is None and prompt_text is None:
                    if DEFAULT_PROMPT_WAV_PATH and os.path.exists(DEFAULT_PROMPT_WAV_PATH):
                        prompt_wav_path = DEFAULT_PROMPT_WAV_PATH
                        prompt_text = DEFAULT_PROMPT_TEXT if DEFAULT_PROMPT_TEXT else None
                    # 如果默认值也没有配置，则 prompt_wav_path 和 prompt_text 保持为 None（不使用声音克隆）

                # 设置随机种子
                seed = data.get("seed")
                if seed is not None:
                    set_seed(int(seed))

                # 生成语音
                model = get_model()
                
                if stream:
                    # 流式生成
                    generator = model.generate_streaming(
                        text=data["text"],
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        cfg_value=data.get("cfg_value", 2.0),
                        inference_timesteps=data.get("inference_timesteps", 25),
                        normalize=data.get("normalize", False),
                        denoise=data.get("denoise", False),
                        show_progress=False,
                    )
                    
                    total_bytes = 0
                    
                    for i, wav_chunk in enumerate(generator):
                        # Convert float32 numpy to int16 bytes (PCM)
                        audio_int16 = (wav_chunk * 32767).clip(-32768, 32767).astype(np.int16)
                        chunk_bytes = audio_int16.tobytes()
                        
                        await ws_manager.send_bytes(websocket, chunk_bytes)
                        total_bytes += len(chunk_bytes)
                    # 发送完成消息
                    await ws_manager.send_json(websocket, {
                        "status": "success", 
                        "audio_size": total_bytes,
                        "sample_rate": model.tts_model.sample_rate,
                        "mode": "stream"
                    })
                else:
                    wav = model.generate(
                        text=data["text"],
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        cfg_value=data.get("cfg_value", 2.0),
                        inference_timesteps=data.get("inference_timesteps", 25),
                        normalize=data.get("normalize", False),
                        denoise=data.get("denoise", False),
                    )
    
                    # 保存到临时文件并读取
                    temp_path = tempfile.mktemp(suffix=".wav")
                    audio_tensor = torch.from_numpy(wav).unsqueeze(0)
                    torchaudio.save(temp_path, audio_tensor, sample_rate=model.tts_model.sample_rate, format="wav")
                    
                    with open(temp_path, "rb") as f:
                        audio_bytes = f.read()
                    
                    # 发送音频数据
                    await ws_manager.send_bytes(websocket, audio_bytes)
                    
                    # 发送完成消息
                    await ws_manager.send_json(websocket, {
                        "status": "success", 
                        "audio_size": len(audio_bytes),
                        "sample_rate": model.tts_model.sample_rate
                    })
                
            except json.JSONDecodeError:
                await ws_manager.send_error(websocket, "Invalid JSON format")
            except Exception as e:
                await ws_manager.send_error(websocket, f"Generation failed: {str(e)}")
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============ 启动入口 ============

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("VoxCPM TTS WebSocket Service")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"WebSocket endpoints:")
    print(f"  - ws://{SERVER_HOST}:{SERVER_PORT}/ws/generate")
    print(f"  - ws://{SERVER_HOST}:{SERVER_PORT}/ws/health")
    print(f"  - ws://{SERVER_HOST}:{SERVER_PORT}/ws/models")
    print("=" * 60)

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        workers=1,  # GPU 模型建议单 worker
        log_level="info",
    )
