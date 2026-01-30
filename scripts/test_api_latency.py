import asyncio
import websockets
import json
import time
import os
import sys
import argparse
import math

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_AUDIO_PATH = "/home/zju/VoxCPM/examples/hailan07.wav"
DEFAULT_TEXT = "江苏现货市场里的费用分摊，主要有几种情况。一类是成本补偿，比如启停机组的费用，会按月向用户和售电公司按用电量比例分摊。然后就是市场不平衡费用，这部分涉及k值返还、结构性偏差这些，会根据电量或电价差来分摊。还有低负荷运行补偿，这类费用风电、光伏、核电按上网电量分摊，剩下部分再和其他发电或者用户按比例分。最后还有像超额收益回收这些，也是按照上网或用电量的比例分摊的。"
DEFAULT_PROMPT_TEXT = "感谢您的耐心，我这就去核实一下，在江苏电力现货市场里，费用分摊主要涉及几类。"
STEP_LENGTHS = [5, 15, 25]


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d = k - f
    return values[f] * (1 - d) + values[c] * d


def summarize_stats(label, values, unit):
    if not values:
        print(f"{label}: 无数据")
        return
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    p99 = percentile(values, 0.99)
    vmin = min(values)
    vmax = max(values)
    print(f"{label}: count={len(values)} min={vmin:.2f}{unit} p50={p50:.2f}{unit} p95={p95:.2f}{unit} p99={p99:.2f}{unit} max={vmax:.2f}{unit}")


def build_text(base_text, length):
    if length <= 0:
        return ""
    if len(base_text) >= length:
        return base_text[:length]
    repeat = (length // len(base_text)) + 1
    return (base_text * repeat)[:length]


async def test_health(uri):
    print(f"\n[Health] {uri}")
    try:
        start_time = time.perf_counter()
        async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
            resp = await websocket.recv()
        end_time = time.perf_counter()
        data = json.loads(resp)
        total_ms = (end_time - start_time) * 1000
        print(f"Response: {data}")
        print(f"Latency: {total_ms:.2f} ms")
    except Exception as e:
        print(f"Error: {e}")


async def test_models(uri):
    print(f"\n[Models] {uri}")
    try:
        start_time = time.perf_counter()
        async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
            resp = await websocket.recv()
        end_time = time.perf_counter()
        data = json.loads(resp)
        total_ms = (end_time - start_time) * 1000
        print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print(f"Latency: {total_ms:.2f} ms")
    except Exception as e:
        print(f"Error: {e}")


async def run_asr_once(uri, audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return None
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    start_time = time.perf_counter()
    async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
        await websocket.send(audio_bytes)
        resp = await websocket.recv()
    end_time = time.perf_counter()
    data = json.loads(resp)
    if data.get("status") == "error":
        return None
    return {
        "total_ms": (end_time - start_time) * 1000,
        "text": data.get("text", ""),
    }


async def run_asr_batch(label, uri, audio_path, runs, warmup):
    print(f"\n[{label}]")
    if not audio_path or not os.path.exists(audio_path):
        print("Audio file not found, ASR test skipped.")
        return
    print(f"audio_path={audio_path} runs={runs} warmup={warmup}")

    for _ in range(warmup):
        try:
            await run_asr_once(uri, audio_path)
        except Exception:
            pass

    results = []
    for i in range(runs):
        try:
            result = await run_asr_once(uri, audio_path)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"run {i + 1} error: {e}")

    if not results:
        print("No successful runs.")
        return

    total = [r["total_ms"] for r in results]
    summarize_stats("Total(ms)", total, "ms")
    sample_text = next((r["text"] for r in results if r.get("text")), "")
    if sample_text:
        print(f"sample_text={sample_text[:80]}")


async def run_vad_once(uri, audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return None
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    start_time = time.perf_counter()
    async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
        await websocket.send(audio_bytes)
        resp = await websocket.recv()
    end_time = time.perf_counter()
    data = json.loads(resp)
    if data.get("status") == "error":
        return None
    segments = data.get("vad_segments") or []
    return {
        "total_ms": (end_time - start_time) * 1000,
        "has_speech": bool(data.get("has_speech")),
        "segments_len": len(segments),
    }


async def run_vad_batch(label, uri, audio_path, runs, warmup):
    print(f"\n[{label}]")
    if not audio_path or not os.path.exists(audio_path):
        print("Audio file not found, VAD test skipped.")
        return
    print(f"audio_path={audio_path} runs={runs} warmup={warmup}")

    for _ in range(warmup):
        try:
            await run_vad_once(uri, audio_path)
        except Exception:
            pass

    results = []
    for i in range(runs):
        try:
            result = await run_vad_once(uri, audio_path)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"run {i + 1} error: {e}")

    if not results:
        print("No successful runs.")
        return

    total = [r["total_ms"] for r in results]
    summarize_stats("Total(ms)", total, "ms")
    has_speech_count = sum(1 for r in results if r["has_speech"])
    segments_avg = sum(r["segments_len"] for r in results) / len(results)
    print(f"has_speech={has_speech_count}/{len(results)} segments_avg={segments_avg:.2f}")


async def run_generate_once(
    uri,
    text,
    prompt_wav,
    prompt_text,
    stream,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
):
    req_data = {
        "text": text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "normalize": normalize,
        "denoise": denoise,
        "stream": stream,
    }
    if prompt_wav:
        req_data["prompt_wav_path"] = prompt_wav
    if prompt_text:
        req_data["prompt_text"] = prompt_text

    start_time = time.perf_counter()
    async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
        await websocket.send(json.dumps(req_data, ensure_ascii=False))
        send_time = time.perf_counter()
        first_chunk_time = None
        total_audio_bytes = 0
        sample_rate = 44100
        while True:
            msg = await websocket.recv()
            if isinstance(msg, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                total_audio_bytes += len(msg)
            else:
                end_time = time.perf_counter()
                res_json = json.loads(msg)
                if res_json.get("status") == "error":
                    return None
                if "sample_rate" in res_json:
                    sample_rate = res_json["sample_rate"]
                break

    first_latency_ms = (first_chunk_time - send_time) * 1000 if first_chunk_time else 0
    total_latency_ms = (end_time - send_time) * 1000
    total_latency_s = total_latency_ms / 1000.0
    bytes_per_sample = 2
    audio_duration_s = total_audio_bytes / (sample_rate * bytes_per_sample) if sample_rate > 0 else 0
    rtf = total_latency_s / audio_duration_s if audio_duration_s > 0 else 0

    return {
        "ttfb_ms": first_latency_ms,
        "total_ms": total_latency_ms,
        "audio_s": audio_duration_s,
        "rtf": rtf,
        "audio_bytes": total_audio_bytes,
        "sample_rate": sample_rate,
    }


async def run_generate_batch(
    label,
    uri,
    text,
    prompt_wav,
    prompt_text,
    stream,
    cfg_value,
    inference_timesteps,
    normalize,
    denoise,
    runs,
    warmup,
):
    mode_str = "Streaming" if stream else "Non-streaming"
    print(f"\n[{label} | {mode_str}]")
    print(f"text_len={len(text)} runs={runs} warmup={warmup}")
    if prompt_wav:
        print(f"prompt_wav_path={prompt_wav}")
    if prompt_text:
        print(f"prompt_text_len={len(prompt_text)}")
    print(f"cfg_value={cfg_value} inference_timesteps={inference_timesteps} normalize={normalize} denoise={denoise}")

    for _ in range(warmup):
        try:
            await run_generate_once(
                uri,
                text,
                prompt_wav,
                prompt_text,
                stream,
                cfg_value,
                inference_timesteps,
                normalize,
                denoise,
            )
        except Exception:
            pass

    results = []
    for i in range(runs):
        try:
            result = await run_generate_once(
                uri,
                text,
                prompt_wav,
                prompt_text,
                stream,
                cfg_value,
                inference_timesteps,
                normalize,
                denoise,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"run {i + 1} error: {e}")

    if not results:
        print("No successful runs.")
        return

    ttfb = [r["ttfb_ms"] for r in results]
    total = [r["total_ms"] for r in results]
    rtf = [r["rtf"] for r in results]
    audio_s = [r["audio_s"] for r in results]
    audio_bytes = [r["audio_bytes"] for r in results]

    summarize_stats("TTFB(ms)", ttfb, "ms")
    summarize_stats("Total(ms)", total, "ms")
    summarize_stats("RTF", rtf, "")
    summarize_stats("Audio(s)", audio_s, "s")
    summarize_stats("Audio(bytes)", audio_bytes, "B")


async def main():
    parser = argparse.ArgumentParser(description="VoxCPM API Latency Test Script")
    parser.add_argument("--host", default=DEFAULT_HOST, help="API Server Host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="API Server Port")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text for fixed-length test")
    parser.add_argument("--prompt-wav", default=DEFAULT_AUDIO_PATH, help="Path to prompt wav for cloning")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT, help="Prompt text for cloning")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per test")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before statistics")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="CFG value")
    parser.add_argument("--inference-timesteps", type=int, default=25, help="Inference timesteps")
    parser.add_argument("--normalize", action="store_true", help="Enable text normalization")
    parser.add_argument("--denoise", action="store_true", help="Enable denoiser")
    parser.add_argument("--step-lengths", default="5,15,25", help="Comma-separated lengths for step test")
    args = parser.parse_args()

    base_uri = f"ws://{args.host}:{args.port}"
    print(f"Target Server: {base_uri}")

    await test_health(f"{base_uri}/ws/health")
    await test_models(f"{base_uri}/ws/models")
    await run_asr_batch("ASR Test", f"{base_uri}/ws/asr", args.prompt_wav, args.runs, args.warmup)
    await run_vad_batch("VAD Test", f"{base_uri}/ws/vad", args.prompt_wav, args.runs, args.warmup)

    modes = [True, False]

    prompt_wav = os.path.abspath(args.prompt_wav) if args.prompt_wav and os.path.exists(args.prompt_wav) else ""
    prompt_text = args.prompt_text.strip() if args.prompt_text else ""

    scenarios = [("No Cloning", "", "")]
    if prompt_wav and prompt_text:
        scenarios.append(("Cloning", prompt_wav, prompt_text))
    else:
        print("Cloning test skipped: prompt_wav and prompt_text are both required.")

    for scenario_label, scenario_wav, scenario_text in scenarios:
        for stream in modes:
            await run_generate_batch(
                f"Fixed Text Test | {scenario_label}",
                f"{base_uri}/ws/generate",
                args.text,
                scenario_wav,
                scenario_text,
                stream,
                args.cfg_value,
                args.inference_timesteps,
                args.normalize,
                args.denoise,
                args.runs,
                args.warmup,
            )

    step_lengths = [int(x) for x in args.step_lengths.split(",") if x.strip().isdigit()]
    if not step_lengths:
        step_lengths = STEP_LENGTHS

    for length in step_lengths:
        step_text = build_text(args.text, length)
        for scenario_label, scenario_wav, scenario_text in scenarios:
            for stream in modes:
                await run_generate_batch(
                    f"Step Text Test ({length}) | {scenario_label}",
                    f"{base_uri}/ws/generate",
                    step_text,
                    scenario_wav,
                    scenario_text,
                    stream,
                    args.cfg_value,
                    args.inference_timesteps,
                    args.normalize,
                    args.denoise,
                    args.runs,
                    args.warmup,
                )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except ConnectionRefusedError:
        print("\nError: Connection refused. Is the server running? (python api.py)")
