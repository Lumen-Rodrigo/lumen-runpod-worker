import json, os, subprocess, tempfile
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import requests, runpod
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator

MAX_INPUT_BYTES = int(os.getenv("LUMEN_MAX_INPUT_BYTES", 500 * 1024 * 1024))
MAX_DURATION_SECONDS = int(os.getenv("LUMEN_MAX_DURATION_SECONDS", "120"))
MODEL_ID = os.getenv("LUMEN_MODEL_ID", "Wan-AI/Wan2.1-VACE-1.3B-diffusers")
FPS, SEGMENT_FRAMES = 16, 81
ALLOWED_STYLES = {"realistic", "comic", "manga", "hybrid"}
STYLE_PROMPTS = {
    "realistic": "photorealistic cinematic footage, natural skin texture, coherent lighting",
    "comic": "high quality western graphic novel, expressive ink lines, cinematic color",
    "manga": "high quality manga animation, precise line art, coherent shading",
    "hybrid": "cinematic live action fused with refined illustrated detail",
}
NEGATIVE_PROMPT = "low quality, blurry, flicker, jitter, warped anatomy, duplicate limbs, deformed face, subtitles, watermark, logo, compression artifacts, abrupt camera motion"
_PIPELINE, _PIPELINE_LOCK = None, Lock()


class JobInput(BaseModel):
    video_url: HttpUrl
    prompt: str = Field(min_length=3, max_length=1200)
    style: str = "realistic"
    preserve_audio: bool = True
    mode: str = "clip"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    inference_steps: int = Field(default=24, ge=12, le=40)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=10.0)
    conditioning_scale: float = Field(default=0.85, ge=0.1, le=1.5)

    @field_validator("style")
    @classmethod
    def valid_style(cls, value):
        if value not in ALLOWED_STYLES:
            raise ValueError("unsupported style")
        return value

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, value):
        if value not in {"clip", "extended"}:
            raise ValueError("mode must be clip or extended")
        return value


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def download(url, destination):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("video_url must be a public HTTPS URL")
    size = 0
    with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as response:
        response.raise_for_status()
        if int(response.headers.get("content-length", "0") or 0) > MAX_INPUT_BYTES:
            raise ValueError("input exceeds the configured size limit")
        with destination.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    size += len(chunk)
                    if size > MAX_INPUT_BYTES:
                        raise ValueError("input exceeds the configured size limit")
                    output.write(chunk)
    if not size:
        raise ValueError("downloaded video is empty")


def probe(path):
    raw = run("ffprobe", "-v", "error", "-show_entries", "format=duration,format_name,size:stream=codec_type,codec_name,width,height", "-of", "json", str(path))
    data = json.loads(raw)
    duration, streams = float(data.get("format", {}).get("duration") or 0), data.get("streams", [])
    video = next((x for x in streams if x.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("input has no video stream")
    if duration < 2 or duration > MAX_DURATION_SECONDS:
        raise ValueError(f"video duration must be between 2 and {MAX_DURATION_SECONDS} seconds")
    return {"duration": duration, "format": data.get("format", {}).get("format_name", "unknown"), "size": int(data.get("format", {}).get("size") or path.stat().st_size), "width": int(video.get("width") or 0), "height": int(video.get("height") or 0), "has_audio": any(x.get("codec_type") == "audio" for x in streams)}


def normalize(source, target):
    run("ffmpeg", "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-vf", "scale=832:480:force_original_aspect_ratio=decrease,pad=832:480:(ow-iw)/2:(oh-ih)/2,fps=16", "-c:v", "libx264", "-profile:v", "main", "-level:v", "4.0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-movflags", "+faststart", str(target))


def load_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            import torch
            from diffusers import AutoencoderKLWan, WanVACEPipeline
            if not torch.cuda.is_available():
                raise RuntimeError("Lumen requires a CUDA GPU")
            vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
            _PIPELINE = WanVACEPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
            _PIPELINE.enable_model_cpu_offload()
            _PIPELINE.vae.enable_tiling()
    return _PIPELINE


def read_frames(path):
    import cv2
    from PIL import Image
    capture, frames = cv2.VideoCapture(str(path)), []
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if not frames:
        raise ValueError("normalized video contains no readable frames")
    return frames


def segment_frames(frames, extended):
    if not extended and len(frames) > SEGMENT_FRAMES * 2:
        raise ValueError("clip mode accepts up to 10 seconds; select extended mode")
    segments = []
    for start in range(0, len(frames), SEGMENT_FRAMES):
        count = min(SEGMENT_FRAMES, len(frames) - start)
        segment = list(frames[start:start + SEGMENT_FRAMES])
        segment.extend([segment[-1]] * (SEGMENT_FRAMES - len(segment)))
        segments.append((segment, count))
    return segments


def generate_segments(source, silent_output, job):
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image
    pipeline, generated = load_pipeline(), []
    prompt = f"{STYLE_PROMPTS[job.style]}. {job.prompt.strip()}"
    for index, (segment, count) in enumerate(segment_frames(read_frames(source), job.mode == "extended")):
        masks = [Image.new("L", segment[0].size, 255) for _ in segment]
        result = pipeline(video=segment, mask=masks, prompt=prompt, negative_prompt=NEGATIVE_PROMPT, height=480, width=832, num_frames=SEGMENT_FRAMES, num_inference_steps=job.inference_steps, guidance_scale=job.guidance_scale, conditioning_scale=job.conditioning_scale, generator=torch.Generator(device="cpu").manual_seed(job.seed + index)).frames[0]
        generated.extend(result[:count])
    export_to_video(generated, str(silent_output), fps=FPS, quality=8)


def mux_audio(video, original, output, preserve_audio, has_audio):
    command = ["ffmpeg", "-y", "-i", str(video)]
    if preserve_audio and has_audio:
        command += ["-i", str(original), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest"]
    else:
        command += ["-map", "0:v:0", "-c:v", "copy", "-an"]
    run(*(command + ["-movflags", "+faststart", str(output)]))


def validate_output(path, expected_duration):
    metadata = probe(path)
    if "mp4" not in metadata["format"] or abs(metadata["duration"] - expected_duration) > max(1.0, expected_duration * .05):
        raise RuntimeError("generated file failed MP4 or duration validation")
    return metadata


def handler(event):
    try:
        job = JobInput.model_validate((event or {}).get("input", {}))
        with tempfile.TemporaryDirectory(prefix="lumen-") as folder:
            root = Path(folder)
            downloaded, normalized = root / "input", root / "normalized.mp4"
            silent, generated = root / "generated-silent.mp4", root / "lumen-result.mp4"
            download(str(job.video_url), downloaded)
            input_metadata = probe(downloaded)
            normalize(downloaded, normalized)
            normalized_metadata = probe(normalized)
            generate_segments(normalized, silent, job)
            mux_audio(silent, normalized, generated, job.preserve_audio, normalized_metadata["has_audio"])
            output_metadata = validate_output(generated, normalized_metadata["duration"])
           url = runpod.serverless.utils.rp_upload.upload_file_to_bucket(generated.name, str(generated))
            return {"status": "completed", "video_url": url, "input": input_metadata, "output": output_metadata, "model": MODEL_ID}
    except ValidationError as error:
        return {"status": "failed", "code": "invalid_input", "error": error.errors(include_url=False)}
    except subprocess.CalledProcessError as error:
        return {"status": "failed", "code": "media_error", "error": (error.stderr or "media processing failed")[-2000:]}
    except Exception as error:
        return {"status": "failed", "code": "processing_failed", "error": str(error)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
