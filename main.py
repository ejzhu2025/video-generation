from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import anthropic
import httpx
import fal_client
import os, json, base64, asyncio, uuid
from dotenv import load_dotenv

load_dotenv()

# fal_client reads FAL_KEY; mirror our FAL_API_KEY into it (strip to remove HF Spaces trailing newlines)
if os.getenv("FAL_API_KEY") and not os.getenv("FAL_KEY"):
    os.environ["FAL_KEY"] = os.getenv("FAL_API_KEY").strip()

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Ad Video Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ============================================================
# CONFIG
# ============================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
FAL_API_KEY       = os.getenv("FAL_API_KEY", "").strip()
FAL_T2V_MODEL     = os.getenv("FAL_T2V_MODEL", "fal-ai/cogvideox-5b")
FAL_I2V_MODEL     = os.getenv("FAL_I2V_MODEL", "fal-ai/wan/v2.1/1.3b/image-to-video")
LUMA_API_KEY      = os.getenv("LUMA_API_KEY", "")
VIDEO_BACKEND     = os.getenv("VIDEO_BACKEND", "mock")   # fal | luma | mock
SERVER_URL        = os.getenv("SERVER_URL", "").rstrip("/")

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

jobs: dict = {}

# ============================================================
# PYDANTIC MODELS
# ============================================================

class ScrapeRequest(BaseModel):
    url: str

class StoryboardRequest(BaseModel):
    brief: str
    platform: str = "instagram_reel"
    photo_url: Optional[str] = None
    brand_name: Optional[str] = None
    duration: int = 5

class SuggestMotionRequest(BaseModel):
    photo_url: str
    brand_name: Optional[str] = None
    platform: str = "instagram_reel"

class GenerateRequest(BaseModel):
    brief: str
    mode: str = "t2v"               # "t2v" | "i2v"
    storyboard: Optional[dict] = None   # t2v only
    duration: int = 5               # video length in seconds (5 or 10)
    photo_url: Optional[str] = None
    model_id: Optional[str] = None      # override env default
    brand_concept: Optional[str] = None # I2V brand concept for music generation
    platform: str = "instagram_reel"

# ============================================================
# URL SCRAPING
# ============================================================

async def scrape_brand_page(url: str) -> dict:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url, headers=SCRAPE_HEADERS)
            html, base_url = resp.text, str(resp.url)
    except Exception as e:
        return {"error": f"Could not fetch page: {e}"}

    soup = BeautifulSoup(html, "html.parser")

    def meta(prop=None, name=None):
        tag = soup.find("meta", property=prop) if prop else soup.find("meta", attrs={"name": name})
        return (tag.get("content") or "").strip() if tag else ""

    brand_name  = meta(prop="og:title") or (soup.title.text.strip() if soup.title else "")
    description = meta(prop="og:description") or meta(name="description")

    logo_url = ""
    for link in soup.find_all("link", rel=True):
        rel = " ".join(link.get("rel", []))
        if "icon" in rel.lower() and link.get("href"):
            raw = link["href"]
            logo_url = raw if raw.startswith("http") else (
                "https:" + raw if raw.startswith("//") else urljoin(base_url, raw))
            break

    images, seen = [], set()
    og_img = meta(prop="og:image")
    if og_img:
        u = og_img if og_img.startswith("http") else urljoin(base_url, og_img)
        images.append(u); seen.add(u)

    for img in soup.find_all("img", src=True):
        src = img.get("src", "").strip()
        if not src or src.startswith("data:"): continue
        src = "https:" + src if src.startswith("//") else (src if src.startswith("http") else urljoin(base_url, src))
        if src in seen: continue
        seen.add(src)
        lower = src.lower()
        if any(k in lower for k in ["icon","favicon","avatar","logo",".svg",".gif","pixel","track"]): continue
        try:
            w, h = int(img.get("width",0)), int(img.get("height",0))
            if (w and w < 100) or (h and h < 100): continue
        except ValueError: pass
        images.append(src)
        if len(images) >= 12: break

    return {"brand_name": brand_name[:80], "description": description[:300],
            "logo_url": logo_url, "images": images, "page_url": base_url}

# ============================================================
# IMAGE HELPERS
# ============================================================

def _media_type(ext: str) -> str:
    return {".jpg":"image/jpeg",".jpeg":"image/jpeg",
            ".png":"image/png",".webp":"image/webp",".gif":"image/gif"
            }.get(ext.lower(), "image/jpeg")

async def _read_image_bytes(photo_url: str) -> tuple[bytes, str]:
    """Returns (bytes, content_type). Handles /uploads/ or external URL."""
    if photo_url.startswith("/uploads/"):
        fp = UPLOAD_DIR / photo_url.removeprefix("/uploads/")
        if not fp.exists():
            raise FileNotFoundError(f"Upload not found: {fp.name}")
        return fp.read_bytes(), _media_type(fp.suffix)
    async with httpx.AsyncClient(timeout=15) as hc:
        r = await hc.get(photo_url, headers=SCRAPE_HEADERS)
        r.raise_for_status()
        ct = r.headers.get("content-type","image/jpeg").split(";")[0].strip()
        return r.content, ct


async def upload_to_fal_storage(photo_url: str) -> str:
    """Upload image to fal.ai CDN storage via fal_client, return the CDN URL."""
    raw, ct = await _read_image_bytes(photo_url)
    loop = asyncio.get_event_loop()
    cdn_url = await loop.run_in_executor(None, lambda: fal_client.upload(raw, ct))
    return cdn_url


async def get_public_image_url(photo_url: str) -> str:
    """Return a publicly accessible URL for fal.ai to fetch."""
    if photo_url.startswith("http"):
        return photo_url                             # already external
    if SERVER_URL:
        return f"{SERVER_URL}{photo_url}"            # serve from our server
    return await upload_to_fal_storage(photo_url)   # upload to fal CDN

# ============================================================
# CLAUDE VISION — IMAGE ANALYSIS
# ============================================================

def _parse_json(raw: str) -> dict:
    """Robustly extract JSON from Claude's response, handling common formatting issues."""
    from json_repair import repair_json
    text = raw.strip()
    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            part = part.strip()
            if part.startswith("{"):
                text = part
                break
    # Extract outermost { ... } in case Claude added extra prose
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    # Try strict parse first, fall back to auto-repair
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        raise ValueError("Repaired JSON is not a dict")
    except Exception as e:
        print(f"[JSON parse] failed even after repair: {e}\nRaw (first 500): {raw[:500]}")
        raise ValueError(f"Claude returned invalid JSON: {e}") from e


async def analyze_image(photo_url: str) -> str:
    if not ANTHROPIC_API_KEY:
        return "A professional product photo with clean composition, appealing colors, and clear brand identity."
    try:
        raw, ct = await _read_image_bytes(photo_url)
    except Exception as e:
        return f"Photo analysis skipped: {e}"
    if len(raw) > 4 * 1024 * 1024:
        return "High-resolution product photo. Apply a clean, professional video style."
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=400,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":ct,"data":base64.standard_b64encode(raw).decode()}},
            {"type":"text","text":(
                "Analyze this product photo for video ad generation. In 3-4 sentences describe:\n"
                "1. The product and its key visual features\n"
                "2. Color palette and lighting mood\n"
                "3. Overall aesthetic style (e.g. 'warm & cozy', 'bold & energetic', 'clean & minimal')\n"
                "4. How this style should translate into a short video ad"
            )},
        ]}],
    )
    return msg.content[0].text.strip()

# ============================================================
# CLAUDE AGENT — T2V STORYBOARD
# ============================================================

async def generate_storyboard(brief: str, platform: str,
                               image_analysis: str = "", brand_name: str = "the brand", duration: int = 5) -> dict:
    specs = {
        "instagram_reel": {"ratio":"9:16","name":"Instagram Reel"},
        "tiktok":         {"ratio":"9:16","name":"TikTok"},
        "youtube":        {"ratio":"16:9","name":"YouTube Short"},
    }
    spec = {**specs.get(platform, specs["instagram_reel"]), "duration": duration}
    if not ANTHROPIC_API_KEY:
        return _mock_storyboard(brand_name, spec, image_analysis)

    photo_ctx = (
        f"\nProduct photo analysis:\n{image_analysis}\n"
        "Use the exact visual style, colors, and product details from this analysis in all scenes."
        if image_analysis else ""
    )
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        system="You are a creative director for short-form video ads. Return ONLY valid JSON, no other text.",
        messages=[{"role":"user","content":f"""Create a {spec['duration']}-second video ad storyboard.

Brand: {brand_name}
Brief: {brief}
Platform: {spec['name']} ({spec['ratio']}){photo_ctx}

Return this exact JSON:
{{
  "title": "Short catchy title",
  "detected_style": "1-line description of auto-detected visual style",
  "style_reasoning": "Why this style was chosen (1-2 sentences)",
  "total_duration": {spec['duration']},
  "platform": "{spec['name']}",
  "ratio": "{spec['ratio']}",
  "overall_prompt": "Single comprehensive AI video generation prompt for the full {spec['duration']}s video",
  "scenes": [
    {{"id":1,"start_time":0.0,"end_time":1.5,"duration":1.5,
      "title":"Scene title","description":"What viewer sees and feels",
      "visual_prompt":"Detailed visual prompt","camera_note":"Camera movement"}}
  ]
}}
Generate 3-4 scenes totaling {spec['duration']} seconds."""}],
    )
    raw = msg.content[0].text.strip()
    return _parse_json(raw)


def _mock_storyboard(brand_name: str, spec: dict, image_analysis: str) -> dict:
    return {
        "title": f"{brand_name} — {spec['duration']}s Video Ad",
        "detected_style": "Warm & inviting with natural product focus",
        "style_reasoning": "Detected from product photo: clean lighting, warm tones, lifestyle composition.",
        "total_duration": spec["duration"], "platform": spec["name"], "ratio": spec["ratio"],
        "overall_prompt": f"Cinematic product ad for {brand_name}. Hero product in warm light, slow reveal, brand identity fade-in.",
        "scenes": [
            {"id":1,"start_time":0.0,"end_time":1.5,"duration":1.5,"title":"Hero Reveal",
             "description":"Product placed center frame with dramatic lighting.",
             "visual_prompt":f"Hero product shot for {brand_name}, cinematic lighting, slow reveal",
             "camera_note":"Slow push-in"},
            {"id":2,"start_time":1.5,"end_time":3.0,"duration":1.5,"title":"Detail Close-up",
             "description":"Macro close-up highlighting textures and quality.",
             "visual_prompt":"Macro product texture, bokeh background, warm light, premium feel",
             "camera_note":"Rack focus from background to product"},
            {"id":3,"start_time":3.0,"end_time":4.5,"duration":1.5,"title":"Lifestyle",
             "description":"Product in real-world context, aspirational.",
             "visual_prompt":"Lifestyle shot, golden hour, natural setting, brand colors",
             "camera_note":"Medium shot, gentle pan"},
            {"id":4,"start_time":4.5,"end_time":5.0,"duration":0.5,"title":"Brand End Card",
             "description":"Brand identity fades in.",
             "visual_prompt":"Brand logo on warm bokeh, elegant fade",
             "camera_note":"Static, fade in"},
        ],
        "_mock": True,
    }

# ============================================================
# CLAUDE — SUGGEST MOTION PROMPT (for i2v)
# ============================================================

async def generate_ad_script(photo_url: str, brand_name: str, platform: str) -> dict:
    """Generate both a human-readable ad concept and a technical I2V motion prompt."""
    analysis = await analyze_image(photo_url)
    if not ANTHROPIC_API_KEY:
        return {
            "concept": (
                "Premium product hero shot bathed in warm cinematic light. "
                "The product commands center frame as soft bokeh elements drift in the background. "
                "Confident, aspirational mood — quality you can feel."
            ),
            "motion_prompt": (
                "Slow cinematic zoom in toward the product. Warm golden light sweeps gently left to right. "
                "Soft bokeh particles drift upward. Product remains sharp and centered."
            ),
        }
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=400,
        system='Return ONLY valid JSON, no other text.',
        messages=[{"role":"user","content":f"""You are both a creative director and a technical AI video prompt engineer.

Analyze this product photo and return JSON with two fields:

1. "concept": A 2-3 sentence COMMERCIAL AD CONCEPT describing the brand story, emotion, and product appeal. Written for a human creative brief.

2. "motion_prompt": A precise TECHNICAL MOTION PROMPT for an image-to-video AI model (max 40 words). Focus ONLY on: camera movement, lighting changes, physics of objects visible in the photo, and atmosphere. This must be achievable by animating the provided image — no new elements.

Product photo analysis: {analysis}
Brand: {brand_name or 'the brand'}
Platform: {platform}

Rules for motion_prompt:
- Use verbs: "zooms", "drifts", "sweeps", "rises", "rotates", "pulses"
- Reference only what's already in the photo
- No narrative or emotion words — only physical motion and light

Return JSON only: {{"concept": "...", "motion_prompt": "..."}}"""}],
    )
    raw = msg.content[0].text.strip()
    return _parse_json(raw)

# ============================================================
# FAL.AI — TEXT TO VIDEO
# ============================================================

def _fal_error(resp) -> str:
    """Extract human-readable error from fal.ai response."""
    try:
        detail = resp.json().get("detail", "")
        if detail:
            return f"fal.ai: {detail}"
    except Exception:
        pass
    return f"fal.ai HTTP {resp.status_code}"

async def _fal_submit(endpoint: str, payload: dict) -> dict:
    """Submit to fal.ai queue. Returns dict with request_id, status_url, response_url."""
    async with httpx.AsyncClient(timeout=30) as hc:
        resp = await hc.post(
            f"https://queue.fal.run/{endpoint}",
            headers={"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        if not resp.is_success:
            raise RuntimeError(_fal_error(resp))
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"fal.ai non-JSON response [{resp.status_code}]: {resp.text[:300]}")
        if "request_id" not in data:
            raise RuntimeError(f"fal.ai no request_id in response: {str(data)[:300]}")
        rid = data["request_id"]
        # Use URLs returned by fal.ai directly — avoids constructing wrong paths per model
        status_url   = data.get("status_url",   f"https://queue.fal.run/{endpoint}/requests/{rid}/status")
        response_url = data.get("response_url", f"https://queue.fal.run/{endpoint}/requests/{rid}")
        print(f"[fal submit] {endpoint.split('/')[-1]} rid={rid[:8]} status_url={status_url}")
        return {"request_id": rid, "status_url": status_url, "response_url": response_url}


# ── Music generation ────────────────────────────────────────

async def _generate_music_prompt(brand_concept: str, platform: str) -> str:
    """Ask Claude to write a Stable Audio prompt based on brand concept."""
    if not ANTHROPIC_API_KEY or not brand_concept:
        return "uplifting cinematic background music, warm and inviting, no vocals, product advertisement"
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=80,
        messages=[{"role":"user","content":
            f"Write a 1-sentence Stable Audio music prompt (max 20 words, no vocals, background music) "
            f"for a {platform} ad with this brand concept: {brand_concept[:300]}\n"
            f"Reply with the prompt only, no quotes."}],
    )
    return msg.content[0].text.strip()


async def generate_music(brand_concept: str, platform: str, duration: int) -> str:
    """Generate background music via fal.ai Stable Audio. Returns audio URL."""
    music_prompt = await _generate_music_prompt(brand_concept, platform)
    seconds = min(47, max(5, duration + 2))   # slightly longer than video, capped at 47s
    print(f"[music] prompt='{music_prompt}' seconds={seconds}")
    job = await _fal_submit("fal-ai/stable-audio", {"prompt": music_prompt, "seconds_total": seconds, "steps": 100})
    for _ in range(60):
        await asyncio.sleep(3)
        async with httpx.AsyncClient(timeout=20) as hc:
            sr = await hc.get(job["status_url"], headers={"Authorization": f"Key {FAL_API_KEY}"})
            data = sr.json()
        st = data.get("status", "")
        if st == "COMPLETED":
            result = data.get("output") or {}
            af = result.get("audio_file") or {}
            url = af.get("url") or result.get("audio_url", "")
            if url:
                return url
            # fetch from response_url
            async with httpx.AsyncClient(timeout=20) as hc:
                rr = await hc.get(job["response_url"], headers={"Authorization": f"Key {FAL_API_KEY}"})
                out = rr.json()
            af = out.get("audio_file") or {}
            return af.get("url", "")
        if st in ("FAILED", "ERROR"):
            raise RuntimeError(f"Music generation failed: {data.get('error','unknown')}")
    raise RuntimeError("Music generation timed out")


async def merge_video_audio(video_url: str, audio_url: str, job_id: str) -> str:
    """Download video+audio, merge with ffmpeg, serve merged file. Returns local path."""
    import subprocess
    out_dir = UPLOAD_DIR / "merged"
    out_dir.mkdir(exist_ok=True)
    vid_path   = out_dir / f"{job_id}_video.mp4"
    aud_path   = out_dir / f"{job_id}_audio.mp3"
    merged_path = out_dir / f"{job_id}_merged.mp4"

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as hc:
        vr = await hc.get(video_url)
        vid_path.write_bytes(vr.content)
        ar = await hc.get(audio_url)
        aud_path.write_bytes(ar.content)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(vid_path),
        "-i", str(aud_path),
        "-map", "0:v:0",         # video stream from video file
        "-map", "1:a:0",         # audio stream from music file
        "-c:v", "copy",          # no re-encode video
        "-c:a", "aac",
        "-shortest",             # cut to video length
        str(merged_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    vid_path.unlink(missing_ok=True)
    aud_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()[-300:]}")
    return f"/uploads/merged/{job_id}_merged.mp4"


async def _start_fal_t2v(prompt: str, model_id: str = "", aspect_ratio: str = "9:16", duration: int = 5) -> dict:
    return await _fal_submit(model_id or FAL_T2V_MODEL, {"prompt": prompt, "aspect_ratio": aspect_ratio, "duration": str(duration)})


def _i2v_payload(prompt: str, image_url: str, model_id: str, duration: int) -> dict:
    """Build model-specific I2V payload — each model uses different duration param."""
    base = {"image_url": image_url, "prompt": prompt}
    if "wan/v2.2" in model_id or "wan/v2.1/1.3b" in model_id:
        # Wan 2.x uses num_frames at 16 fps; clamp to 17-161
        frames = max(17, min(161, round(duration * 16)))
        base["num_frames"] = frames
    elif "kling" in model_id:
        base["duration"] = str(duration)      # Kling: string '5' or '10'
    # MiniMax: no duration param, omit entirely
    return base


async def _start_fal_i2v(prompt: str, image_url: str, model_id: str = "", duration: int = 5) -> dict:
    mid = model_id or FAL_I2V_MODEL
    return await _fal_submit(mid, _i2v_payload(prompt, image_url, mid, duration))


async def _poll_fal(job: dict) -> dict:
    """Poll using status_url/response_url from the fal.ai submit response."""
    async with httpx.AsyncClient(timeout=30) as hc:
        sr = await hc.get(job["status_url"], headers={"Authorization": f"Key {FAL_API_KEY}"})
        try:
            status_data = sr.json()
        except Exception:
            print(f"[fal poll] parse error HTTP {sr.status_code}: {sr.text[:200]}")
            return {"status": "pending"}

        status = status_data.get("status", "")
        print(f"[fal poll] {job['request_id'][:8]}… → {status}")

        if status == "COMPLETED":
            rr = await hc.get(job["response_url"], headers={"Authorization": f"Key {FAL_API_KEY}"})
            try:
                result = rr.json()
            except Exception:
                return {"status": "failed", "error": f"Result parse error: {rr.text[:300]}"}
            print(f"[fal result] keys={list(result.keys())}")
            outputs = result.get("outputs") or []
            video_url = (
                (result.get("video") or {}).get("url")
                or result.get("video_url")
                or (result.get("output") or {}).get("video_url")
                or (outputs[0].get("url") if outputs else None)
            )
            if video_url:
                return {"status": "completed", "video_url": video_url}
            return {"status": "failed", "error": f"No video URL. keys={list(result.keys())} | {str(result)[:300]}"}

        if status in ("IN_QUEUE", "IN_PROGRESS"):
            return {"status": "pending"}
        return {"status": "failed", "error": f"Unexpected fal status: {status}"}

# ============================================================
# LUMAAI (premium fallback)
# ============================================================

async def _start_luma(prompt: str, aspect_ratio: str, photo_url: Optional[str]) -> str:
    payload: dict = {"prompt": prompt, "aspect_ratio": aspect_ratio, "loop": False}
    if photo_url and SERVER_URL:
        from urllib.parse import quote
        img_url = f"{SERVER_URL}{photo_url}" if photo_url.startswith("/uploads/") else photo_url
        payload["keyframes"] = {"frame0": {"type": "image", "url": img_url}}
    async with httpx.AsyncClient(timeout=30) as hc:
        resp = await hc.post(
            "https://api.lumalabs.ai/dream-machine/v1/generations",
            headers={"Authorization": f"Bearer {LUMA_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["id"]


async def _poll_luma(generation_id: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as hc:
        resp = await hc.get(
            f"https://api.lumalabs.ai/dream-machine/v1/generations/{generation_id}",
            headers={"Authorization": f"Bearer {LUMA_API_KEY}"},
        )
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", "")
        if state == "completed": return {"status":"completed","video_url":data["video"]["url"]}
        if state == "failed":    return {"status":"failed","error":data.get("failure_reason","unknown")}
        return {"status":"pending"}

# ============================================================
# JOB RUNNER
# ============================================================

async def run_generation_job(
    job_id: str, prompt: str, aspect_ratio: str,
    photo_url: Optional[str], mode: str = "t2v", chosen_model: str = "", duration: int = 5
):
    jobs[job_id].update({"status":"generating","message":"Sending to video model...","progress":10})
    try:
        backend = VIDEO_BACKEND.lower()

        # ── fal.ai ──────────────────────────────────────────
        if backend == "fal":
            if not FAL_API_KEY:
                jobs[job_id].update({"status":"failed","error":"FAL_API_KEY not set in .env"}); return

            if mode == "i2v":
                if not photo_url:
                    jobs[job_id].update({"status":"failed","error":"No photo provided for image-to-video"}); return
                jobs[job_id]["message"] = "Uploading image to fal.ai..."
                try:
                    public_img_url = await get_public_image_url(photo_url)
                except Exception as e:
                    jobs[job_id].update({"status":"failed","error":f"Image upload failed: {e}"}); return
                model_id = chosen_model or FAL_I2V_MODEL
                jobs[job_id]["message"] = f"Generating with {model_id.split('/')[-1]}..."
                fal_job = await _start_fal_i2v(prompt, public_img_url, model_id, duration)
            else:
                model_id = chosen_model or FAL_T2V_MODEL
                jobs[job_id]["message"] = f"Generating with {model_id.split('/')[-1]}..."
                fal_job = await _start_fal_t2v(prompt, model_id, aspect_ratio, duration)

            for i in range(120):
                await asyncio.sleep(5)
                jobs[job_id]["progress"] = min(12 + i, 75)
                result = await _poll_fal(fal_job)
                if result["status"] == "completed":
                    video_url = result["video_url"]
                    # ── Add background music ──────────────────
                    brand_concept = jobs[job_id].get("brand_concept", "")
                    jobs[job_id].update({"message": "Generating background music...", "progress": 80})
                    try:
                        audio_url = await generate_music(brand_concept, jobs[job_id].get("platform","instagram_reel"), duration)
                        jobs[job_id].update({"message": "Mixing music into video...", "progress": 90})
                        video_url = await merge_video_audio(video_url, audio_url, job_id)
                    except Exception as e:
                        print(f"[music] skipped: {e}")  # non-fatal, use original video
                    jobs[job_id].update({"status":"completed","video_url":video_url,
                                         "message":"Video ready!","progress":100,"mock":False}); return
                if result["status"] == "failed":
                    jobs[job_id].update({"status":"failed","error":result["error"]}); return

        # ── LumaAI ──────────────────────────────────────────
        elif backend == "luma":
            if not LUMA_API_KEY:
                jobs[job_id].update({"status":"failed","error":"LUMA_API_KEY not set."}); return
            # LumaAI supports i2v natively via keyframes
            gen_id = await _start_luma(prompt, aspect_ratio, photo_url if mode == "i2v" else None)
            jobs[job_id]["message"] = "LumaAI generating... (2–4 min)"
            for i in range(80):
                await asyncio.sleep(3)
                jobs[job_id]["progress"] = min(12 + i, 88)
                result = await _poll_luma(gen_id)
                if result["status"] == "completed":
                    jobs[job_id].update({"status":"completed","video_url":result["video_url"],
                                         "message":"Video ready!","progress":100,"mock":False}); return
                if result["status"] == "failed":
                    jobs[job_id].update({"status":"failed","error":result["error"]}); return

        # ── Mock ─────────────────────────────────────────────
        else:
            label = "i2v mock" if mode == "i2v" else "t2v mock"
            jobs[job_id]["message"] = f"Mock mode ({label}) — simulating..."
            for i in range(5):
                await asyncio.sleep(2)
                jobs[job_id]["progress"] = 20 + i * 15
            jobs[job_id].update({"status":"completed","video_url":None,
                                  "message":"Mock complete! Set VIDEO_BACKEND=fal in .env.",
                                  "progress":100,"mock":True,"mode":mode})
            return

        jobs[job_id].update({"status":"failed","error":"Generation timed out."})
    except Exception as e:
        jobs[job_id].update({"status":"failed","error":str(e)})

# ============================================================
# API ROUTES
# ============================================================

@app.post("/api/scrape")
async def scrape_url(req: ScrapeRequest):
    if not req.url.startswith("http"): req.url = "https://" + req.url
    result = await scrape_brand_page(req.url)
    if "error" in result: raise HTTPException(400, result["error"])
    return result

@app.post("/api/upload")
async def upload_photo(file: UploadFile = File(...)):
    allowed = {"image/jpeg","image/png","image/webp","image/gif"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported: {file.content_type}")
    ext = {"image/jpeg":".jpg","image/png":".png","image/webp":".webp","image/gif":".gif"}.get(file.content_type,".jpg")
    fname = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / fname).write_bytes(await file.read())
    return {"filename": fname, "url": f"/uploads/{fname}"}

@app.post("/api/storyboard")
async def create_storyboard(req: StoryboardRequest):
    try:
        analysis = await analyze_image(req.photo_url) if req.photo_url else ""
        result   = await generate_storyboard(req.brief, req.platform, analysis, req.brand_name or "the brand", req.duration)
        result["image_analysis"]  = analysis
        if req.photo_url: result["reference_photo"] = req.photo_url
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/suggest-motion")
async def suggest_motion(req: SuggestMotionRequest):
    try:
        result = await generate_ad_script(req.photo_url, req.brand_name or "", req.platform)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/generate")
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    if req.mode == "t2v":
        prompt = (req.storyboard or {}).get("overall_prompt", req.brief)
        ratio  = (req.storyboard or {}).get("ratio", "9:16")
    else:  # i2v
        prompt = req.brief
        ratio  = "9:16"

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status":"pending","message":"Queued...","progress":0,
                    "video_url":None,"error":None,"mock":False,"mode":req.mode,
                    "brand_concept": req.brand_concept or prompt[:200],
                    "platform": req.platform}
    background_tasks.add_task(
        run_generation_job, job_id, prompt, ratio, req.photo_url, req.mode, req.model_id or "", req.duration
    )
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs: raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

# ============================================================
# HTML TEMPLATE
# ============================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Ad Video Generator</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"></script>
  <style>
    body{background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
    .pcard{transition:transform .15s,box-shadow .15s;cursor:pointer}
    .pcard:hover{transform:translateY(-3px);box-shadow:0 12px 28px rgba(0,0,0,.13)}
    .pcard.sel{transform:translateY(-3px)}
    .mcard{transition:transform .2s,box-shadow .2s,border-color .2s;cursor:pointer}
    .mcard:hover{transform:translateY(-4px);box-shadow:0 16px 36px rgba(0,0,0,.12)}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
    .spin{display:inline-block;animation:spin 1s linear infinite}
    .bob{animation:bob 1.4s ease-in-out infinite}
    [x-cloak]{display:none!important}
    .dropzone{border:2px dashed #cbd5e1;transition:border-color .2s,background .2s}
    .dropzone.over{border-color:#6366f1;background:#eef2ff}
    .scene-bar{border-left:4px solid #6366f1}
    .i2v-bar{border-left:4px solid #0ea5e9}
  </style>
</head>
<body>
<div x-data="adAgent()" class="min-h-screen">

  <!-- ══ NAVBAR ══════════════════════════════════════════ -->
  <nav class="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-10">
    <div class="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-2xl">🎬</span>
        <div>
          <h1 class="text-lg font-bold text-slate-800">Ad Video Generator</h1>
          <p class="text-xs text-slate-400">AI-powered short video ads from your product</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <!-- Mode badge -->
        <span x-show="mode==='t2v'"   class="px-2.5 py-1 bg-indigo-100 text-indigo-700 text-xs font-semibold rounded-full">📝 Text → Video</span>
        <span x-show="mode==='i2v'"   class="px-2.5 py-1 bg-sky-100 text-sky-700 text-xs font-semibold rounded-full">🖼 Image → Video</span>
        <button @click="reset()" class="text-xs text-slate-400 hover:text-slate-600 px-3 py-1.5 rounded-lg hover:bg-slate-100 transition-colors">
          ↺ <span x-text="mode ? 'Change Mode' : 'Reset'"></span>
        </button>
      </div>
    </div>
  </nav>

  <!-- ══ STEP INDICATOR (only when mode set) ═══════════ -->
  <div x-show="mode" class="max-w-3xl mx-auto px-6 pt-6 pb-2">
    <!-- T2V: 5 steps -->
    <div x-show="mode==='t2v'" class="flex items-center">
      <template x-for="(s,i) in ['Brand URL','Photos','Brief','Plan','Video']" :key="i">
        <div class="flex items-center flex-1">
          <div class="flex flex-col items-center gap-1 shrink-0">
            <div :class="step>i+1?'bg-indigo-500 text-white':step===i+1?'bg-indigo-500 text-white':'bg-slate-200 text-slate-400'"
                 class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all">
              <span x-show="step>i+1">✓</span><span x-show="step<=i+1" x-text="i+1"></span>
            </div>
            <span :class="step>=i+1?'text-indigo-600':'text-slate-400'" class="text-xs font-medium" x-text="s"></span>
          </div>
          <div x-show="i<4" :class="step>i+1?'bg-indigo-400':'bg-slate-200'" class="flex-1 h-0.5 mx-1 mb-5 transition-all"></div>
        </div>
      </template>
    </div>
    <!-- I2V: 3 steps -->
    <div x-show="mode==='i2v'" class="flex items-center">
      <template x-for="(s,i) in ['Upload Photo','Ad Script','Video']" :key="i">
        <div class="flex items-center flex-1">
          <div class="flex flex-col items-center gap-1 shrink-0">
            <div :class="step>i+1?'bg-sky-500 text-white':step===i+1?'bg-sky-500 text-white':'bg-slate-200 text-slate-400'"
                 class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all">
              <span x-show="step>i+1">✓</span><span x-show="step<=i+1" x-text="i+1"></span>
            </div>
            <span :class="step>=i+1?'text-sky-600':'text-slate-400'" class="text-xs font-medium" x-text="s"></span>
          </div>
          <div x-show="i<2" :class="step>i+1?'bg-sky-400':'bg-slate-200'" class="flex-1 h-0.5 mx-1 mb-5 transition-all"></div>
        </div>
      </template>
    </div>
  </div>

  <!-- ══ MAIN CONTENT ══════════════════════════════════ -->
  <div class="max-w-3xl mx-auto px-6 pb-16">

    <!-- ERROR -->
    <div x-show="error" x-cloak class="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm flex justify-between">
      <span><strong>Error:</strong> <span x-text="error"></span></span>
      <button @click="error=null" class="ml-4 text-red-400 hover:text-red-600">✕</button>
    </div>

    <!-- ╔══════════════════════════════════════════════╗ -->
    <!-- ║  LANDING — MODE PICKER                      ║ -->
    <!-- ╚══════════════════════════════════════════════╝ -->
    <div x-show="!mode" class="pt-4 space-y-8">
      <div class="text-center py-6">
        <h2 class="text-3xl font-bold text-slate-800">Generate AI Video Ads</h2>
        <p class="text-slate-500 mt-2">Choose your creation mode</p>
      </div>

      <div class="grid grid-cols-2 gap-4">

        <!-- T2V Card -->
        <div @click="selectMode('t2v')"
             class="mcard bg-white rounded-2xl p-5 border-2 border-slate-200 hover:border-indigo-400 shadow-sm">
          <div class="text-3xl mb-3">📝</div>
          <h3 class="text-base font-bold text-slate-800">Text → Video</h3>
          <p class="text-slate-500 text-xs mt-2 leading-relaxed">
            Brand URL + brief. Claude writes the storyboard and picks visual style.
          </p>
          <ul class="mt-3 space-y-1 text-xs text-slate-400">
            <li class="flex items-center gap-1.5"><span class="text-indigo-500">✓</span> URL scraping</li>
            <li class="flex items-center gap-1.5"><span class="text-indigo-500">✓</span> AI storyboard</li>
            <li class="flex items-center gap-1.5"><span class="text-indigo-500">✓</span> Multi-scene narrative</li>
          </ul>
          <div class="mt-5 w-full py-2.5 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl text-sm font-semibold text-center transition-colors">
            Start →
          </div>
        </div>

        <!-- I2V Card -->
        <div @click="selectMode('i2v')"
             class="mcard bg-white rounded-2xl p-5 border-2 border-slate-200 hover:border-sky-400 shadow-sm">
          <div class="text-3xl mb-3">🖼</div>
          <h3 class="text-base font-bold text-slate-800">Image → Video</h3>
          <p class="text-slate-500 text-xs mt-2 leading-relaxed">
            Upload your product photo and animate it with AI motion.
          </p>
          <ul class="mt-3 space-y-1 text-xs text-slate-400">
            <li class="flex items-center gap-1.5"><span class="text-sky-500">✓</span> Real product image</li>
            <li class="flex items-center gap-1.5"><span class="text-sky-500">✓</span> AI motion prompts</li>
            <li class="flex items-center gap-1.5"><span class="text-sky-500">✓</span> Brand consistency</li>
          </ul>
          <div class="mt-5 w-full py-2.5 bg-sky-500 hover:bg-sky-600 text-white rounded-xl text-sm font-semibold text-center transition-colors">
            Start →
          </div>
        </div>

      </div>

      <div class="bg-slate-50 rounded-xl p-4 border border-slate-200 text-center">
        <p class="text-xs text-slate-500">
          <strong>Text→Video</strong> generates new visuals from your brief &nbsp;·&nbsp;
          <strong>Image→Video</strong> animates your real product photo
        </p>
      </div>
    </div>

    <!-- ╔══════════════════════════════════════════════╗ -->
    <!-- ║  T2V STEPS                                   ║ -->
    <!-- ╚══════════════════════════════════════════════╝ -->

    <!-- T2V Step 1 — Brand URL -->
    <div x-show="mode==='t2v' && step===1" class="space-y-5 pt-2">
      <div class="text-center py-4">
        <h2 class="text-2xl font-bold text-slate-800">Enter your product webpage</h2>
        <p class="text-slate-500 mt-1 text-sm">Auto-extract brand name, description, and product photos</p>
      </div>
      <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
        <div class="flex gap-2">
          <div class="flex-1 relative">
            <span class="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm">🌐</span>
            <input x-model="pageUrl" type="url" placeholder="https://your-brand.com"
                   @keydown.enter="scrapePage()"
                   class="w-full pl-9 pr-4 py-3 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"/>
          </div>
          <button @click="scrapePage()" :disabled="!pageUrl.trim()||scraping"
                  :class="pageUrl.trim()&&!scraping?'bg-indigo-500 hover:bg-indigo-600 text-white':'bg-slate-200 text-slate-400 cursor-not-allowed'"
                  class="px-5 py-3 rounded-xl font-semibold text-sm transition-all flex items-center gap-2 shrink-0">
            <span x-show="scraping" class="spin">↻</span>
            <span x-text="scraping?'Extracting...':'Extract'"></span>
          </button>
        </div>
        <div x-show="brandInfo" x-cloak class="bg-indigo-50 border border-indigo-200 rounded-xl p-4 space-y-2">
          <div class="flex items-center gap-3">
            <img x-show="brandInfo&&brandInfo.logo_url" :src="brandInfo&&brandInfo.logo_url"
                 class="w-10 h-10 rounded-lg object-contain bg-white border border-slate-200 p-1"
                 @error="$el.style.display='none'"/>
            <div>
              <p class="font-semibold text-slate-800" x-text="brandInfo&&brandInfo.brand_name"></p>
              <p class="text-xs text-slate-500" x-text="brandInfo&&brandInfo.page_url"></p>
            </div>
          </div>
          <p class="text-sm text-slate-600 leading-relaxed" x-text="brandInfo&&brandInfo.description"></p>
          <p class="text-xs text-indigo-600 font-medium"
             x-text="brandInfo?'📸 '+brandInfo.images.length+' product images found':''"></p>
        </div>
        <div class="relative flex items-center gap-3">
          <div class="flex-1 border-t border-slate-200"></div>
          <span class="text-xs text-slate-400 shrink-0">or</span>
          <div class="flex-1 border-t border-slate-200"></div>
        </div>
        <button @click="step=2" class="w-full py-3 border-2 border-dashed border-slate-300 hover:border-indigo-400 text-slate-500 hover:text-indigo-600 rounded-xl text-sm font-medium transition-all">
          ↑ Skip — upload photos directly
        </button>
      </div>
      <div class="flex justify-end">
        <button @click="step=2" x-show="brandInfo" x-cloak
                class="px-7 py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl font-semibold text-sm">
          Continue with these photos →
        </button>
      </div>
    </div>

    <!-- T2V Step 2 — Photos -->
    <div x-show="mode==='t2v' && step===2" class="space-y-5 pt-2">
      <div class="text-center py-4">
        <h2 class="text-2xl font-bold text-slate-800">Select a reference photo</h2>
        <p class="text-slate-500 mt-1 text-sm">Claude analyzes its style to anchor your video visuals</p>
      </div>
      <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
        <div class="dropzone rounded-xl p-6 text-center"
             @dragover.prevent="$el.classList.add('over')"
             @dragleave="$el.classList.remove('over')"
             @drop.prevent="$el.classList.remove('over'); handleDrop($event)">
          <div class="text-3xl mb-2">📁</div>
          <p class="text-sm font-medium text-slate-600">Drag & drop photos here</p>
          <input type="file" accept="image/*" multiple x-ref="t2vFile" class="hidden" @change="handleFileUpload($event)"/>
          <button @click="$refs.t2vFile.click()" class="mt-2 px-4 py-2 bg-indigo-500 hover:bg-indigo-600 text-white text-sm rounded-lg font-medium transition-colors">
            Choose Files
          </button>
        </div>
        <div x-show="uploading" class="mt-3 flex items-center gap-2 text-sm text-indigo-600"><span class="spin">↻</span> Uploading...</div>
      </div>
      <div x-show="allPhotos.length>0" class="space-y-3">
        <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Available photos (<span x-text="allPhotos.length"></span>)</p>
        <div class="grid grid-cols-3 gap-3">
          <template x-for="photo in allPhotos" :key="photo.url">
            <div @click="selectedPhotoUrl=photo.url"
                 :class="selectedPhotoUrl===photo.url?'sel ring-4 ring-indigo-500 ring-offset-2':''"
                 class="pcard bg-white rounded-xl overflow-hidden border border-slate-200 shadow-sm">
              <div class="relative">
                <img :src="photo.url" class="w-full h-32 object-cover bg-slate-100" @error="photo.failed=true; $el.style.display='none'"/>
                <div x-show="photo.failed" class="w-full h-32 bg-slate-100 flex items-center justify-center text-slate-400 text-xs">Load failed</div>
                <div x-show="selectedPhotoUrl===photo.url" class="absolute top-2 right-2 w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center text-white text-xs font-bold shadow">✓</div>
                <span x-show="photo.source==='upload'" class="absolute top-2 left-2 px-1.5 py-0.5 bg-green-500 text-white text-xs rounded font-medium">Uploaded</span>
              </div>
              <div class="p-2"><p class="text-xs text-slate-600 truncate" x-text="photo.label"></p></div>
            </div>
          </template>
        </div>
      </div>
      <div x-show="selectedPhotoUrl" x-cloak class="bg-indigo-50 border border-indigo-200 rounded-xl p-3 flex items-center gap-3">
        <img :src="selectedPhotoUrl" class="w-14 h-14 rounded-lg object-cover shrink-0"/>
        <p class="text-xs text-indigo-600">✓ Selected — Claude Vision will analyze this when building your plan.</p>
      </div>
      <div class="flex justify-between items-center">
        <button @click="step=1" class="px-5 py-3 text-slate-400 hover:text-slate-600 text-sm">← Back</button>
        <div class="flex gap-3">
          <button @click="step=3; selectedPhotoUrl=null" class="px-4 py-2.5 border border-slate-200 text-slate-500 hover:bg-slate-50 rounded-xl text-sm">Skip photo</button>
          <button @click="step=3"
                  :class="selectedPhotoUrl?'bg-indigo-500 hover:bg-indigo-600 text-white':'bg-slate-200 text-slate-400 cursor-not-allowed'"
                  class="px-7 py-3 rounded-xl font-semibold text-sm">Continue →</button>
        </div>
      </div>
    </div>

    <!-- T2V Step 3 — Brief -->
    <div x-show="mode==='t2v' && step===3" class="space-y-5 pt-2">
      <div class="text-center py-4">
        <h2 class="text-2xl font-bold text-slate-800">Describe your video ad</h2>
        <p class="text-slate-500 mt-1 text-sm">Style is auto-detected from your photo — just describe the story</p>
      </div>
      <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 space-y-4">
        <div>
          <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Brand / Product name</label>
          <input x-model="brandName" type="text" placeholder="e.g. Tong Sui, Nike, Apple..."
                 class="mt-1 w-full px-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"/>
        </div>
        <div>
          <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider">What should this video say?</label>
          <textarea x-model="brief" rows="4"
                    placeholder="e.g. 'A 5-second cozy video showcasing our coconut drink. Warm, inviting, perfect for a summer afternoon. Drive foot traffic to our Bay Area stores.'"
                    class="mt-1 w-full p-4 border border-slate-200 rounded-xl text-slate-700 placeholder-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-300 resize-none text-sm leading-relaxed">
          </textarea>
        </div>
        <div>
          <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Platform</label>
          <div class="mt-1 flex gap-5">
            <label class="flex items-center gap-2 cursor-pointer text-sm text-slate-600">
              <input type="radio" x-model="platform" value="instagram_reel" class="accent-indigo-500"> 📱 Instagram Reel
            </label>
            <label class="flex items-center gap-2 cursor-pointer text-sm text-slate-600">
              <input type="radio" x-model="platform" value="tiktok" class="accent-indigo-500"> 🎵 TikTok
            </label>
            <label class="flex items-center gap-2 cursor-pointer text-sm text-slate-600">
              <input type="radio" x-model="platform" value="youtube" class="accent-indigo-500"> ▶️ YouTube Short
            </label>
          </div>
        </div>
      </div>
      <!-- Duration selector — before plan generation so Claude writes correct total duration -->
      <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 space-y-3">
        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Video Duration</p>
        <div class="grid grid-cols-2 gap-3">
          <template x-for="d in [5,10]" :key="d">
            <div @click="videoDuration=d"
                 :class="videoDuration===d?'border-indigo-500 bg-indigo-50':'border-slate-200 hover:border-indigo-300 bg-white'"
                 class="cursor-pointer border-2 rounded-xl p-3 text-center transition-all">
              <p class="text-sm font-bold text-slate-800" x-text="d+'s'"></p>
            </div>
          </template>
        </div>
      </div>

      <div class="flex justify-between">
        <button @click="step=2" class="px-5 py-3 text-slate-400 hover:text-slate-600 text-sm">← Back</button>
        <button @click="generateStoryboard()" :disabled="!brief.trim()||loading"
                :class="brief.trim()&&!loading?'bg-indigo-500 hover:bg-indigo-600 text-white':'bg-slate-200 text-slate-400 cursor-not-allowed'"
                class="px-7 py-3 rounded-xl font-semibold text-sm flex items-center gap-2">
          <span x-show="loading" class="spin">↻</span>
          <span x-text="loading?(selectedPhotoUrl?'Analyzing photo...':'Building plan...'):'Generate Plan →'"></span>
        </button>
      </div>
    </div>

    <!-- T2V Step 4 — Plan -->
    <div x-show="mode==='t2v' && step===4 && storyboard" class="space-y-5 pt-2">
      <div class="text-center py-4">
        <h2 class="text-2xl font-bold text-slate-800">Your Video Plan</h2>
        <p class="text-slate-600 font-medium mt-1 text-sm" x-text="storyboard&&storyboard.title"></p>
        <div class="flex justify-center flex-wrap gap-2 mt-3">
          <span class="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-medium" x-text="storyboard&&storyboard.total_duration+'s'"></span>
          <span class="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-medium" x-text="storyboard&&storyboard.platform"></span>
          <span x-show="storyboard&&storyboard._mock" class="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-xs">⚠ Mock</span>
        </div>
      </div>
      <!-- Auto-detected style -->
      <div class="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-xl p-4">
        <div class="flex items-start gap-3">
          <img x-show="selectedPhotoUrl" :src="selectedPhotoUrl" class="w-14 h-14 rounded-lg object-cover shadow-sm shrink-0"/>
          <div>
            <p class="text-xs font-semibold text-indigo-600 uppercase tracking-wider mb-1">✨ Auto-detected style from your photo</p>
            <p class="text-sm font-semibold text-slate-800" x-text="storyboard&&storyboard.detected_style"></p>
            <p class="text-xs text-slate-500 mt-1 leading-relaxed" x-text="storyboard&&storyboard.style_reasoning"></p>
          </div>
        </div>
      </div>
      <!-- Overall prompt -->
      <div class="bg-slate-50 rounded-xl p-4 border border-slate-200">
        <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2">Video Generation Prompt</p>
        <p class="text-sm text-slate-600 italic leading-relaxed" x-text="storyboard&&storyboard.overall_prompt"></p>
      </div>
      <!-- Scenes -->
      <div class="space-y-3">
        <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Scenes</p>
        <template x-for="scene in (storyboard?storyboard.scenes:[])" :key="scene.id">
          <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden scene-bar">
            <div class="p-5">
              <div class="flex items-center justify-between mb-3">
                <div class="flex items-center gap-3">
                  <span class="w-7 h-7 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs font-bold" x-text="scene.id"></span>
                  <h3 class="font-semibold text-slate-800 text-sm" x-text="scene.title"></h3>
                </div>
                <span class="text-xs text-slate-400 bg-slate-100 px-2 py-1 rounded-full font-mono" x-text="scene.start_time+'s – '+scene.end_time+'s'"></span>
              </div>
              <p class="text-slate-600 text-sm leading-relaxed" x-text="scene.description"></p>
              <div class="flex items-start gap-2 mt-3 pt-3 border-t border-slate-100">
                <span class="text-slate-300 text-xs">🎥</span>
                <p class="text-slate-400 text-xs italic" x-text="scene.camera_note"></p>
              </div>
            </div>
          </div>
        </template>
      </div>
      <!-- T2V Model selector -->
      <div class="space-y-2">
        <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Video Model</p>
        <div class="grid grid-cols-2 gap-2">
          <template x-for="m in t2vModels" :key="m.id">
            <div @click="selectedT2VModel=m.id"
                 :class="selectedT2VModel===m.id?'border-indigo-500 bg-indigo-50':'border-slate-200 hover:border-indigo-300 bg-white'"
                 class="cursor-pointer border-2 rounded-xl p-3 transition-all relative">
              <div x-show="m.recommended" class="absolute -top-2 right-3 px-2 py-0.5 bg-indigo-500 text-white text-xs rounded-full font-semibold">Recommended</div>
              <p class="text-xs font-bold text-slate-800" x-text="m.name"></p>
              <div class="flex items-center gap-1 mt-1">
                <template x-for="n in 5" :key="n">
                  <span :class="n<=m.quality?'text-amber-400':'text-slate-200'" class="text-xs">★</span>
                </template>
              </div>
              <div class="flex items-center justify-between mt-2">
                <span class="text-xs text-slate-500" x-text="m.price"></span>
                <span :class="m.speedColor" class="text-xs font-medium px-1.5 py-0.5 rounded-full" x-text="m.speed"></span>
              </div>
              <p class="text-xs text-slate-400 mt-1" x-text="m.desc"></p>
            </div>
          </template>
        </div>
      </div>

      <div class="flex justify-between">
        <button @click="step=3" class="px-5 py-3 text-slate-400 hover:text-slate-600 text-sm">← Adjust Brief</button>
        <button @click="startGeneration('t2v')" class="px-7 py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl font-semibold text-sm flex items-center gap-2">
          ✅ Generate Video →
        </button>
      </div>
    </div>

    <!-- T2V Step 5 — Video -->
    <div x-show="mode==='t2v' && step===5" class="pt-2">
      <template x-if="!videoReady">
        <div class="text-center py-16 space-y-6">
          <div class="bob inline-block text-5xl">🎬</div>
          <div>
            <h2 class="text-2xl font-bold text-slate-800">Generating your video</h2>
            <p class="text-slate-500 mt-2 text-sm" x-text="genMessage||'AI is working...'"></p>
          </div>
          <div class="max-w-xs mx-auto">
            <div class="bg-slate-200 rounded-full h-2.5 overflow-hidden">
              <div class="bg-indigo-500 h-2.5 rounded-full transition-all duration-700" :style="'width:'+Math.max(5,genProgress)+'%'"></div>
            </div>
            <p class="text-xs text-slate-400 mt-2" x-text="genProgress+'%'"></p>
          </div>
        </div>
      </template>
      <template x-if="videoReady">
        <div class="space-y-6">
          <div class="text-center py-4"><div class="text-4xl mb-3">🎉</div>
            <h2 class="text-2xl font-bold text-slate-800">Your video is ready!</h2></div>
          <div x-show="mockMode" class="bg-yellow-50 border border-yellow-200 rounded-xl p-4 text-sm text-yellow-800">
            <strong>Mock mode</strong> — Set <code class="bg-yellow-100 px-1 rounded">VIDEO_BACKEND=fal</code> + <code class="bg-yellow-100 px-1 rounded">FAL_API_KEY</code> for real video.
          </div>
          <div x-show="videoUrl" class="bg-slate-900 rounded-2xl overflow-hidden shadow-xl max-w-xs mx-auto">
            <video :src="videoUrl" controls autoplay loop muted playsinline class="w-full aspect-[9/16] object-cover"></video>
          </div>
          <div x-show="!videoUrl&&mockMode" class="max-w-xs mx-auto h-64 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl flex flex-col items-center justify-center gap-3 border-2 border-dashed border-indigo-300">
            <span class="text-4xl">🎬</span><p class="text-sm font-medium text-slate-600">Video preview (mock)</p>
          </div>
          <div class="flex justify-center gap-3">
            <a x-show="videoUrl" :href="videoUrl" download="ad-video-t2v.mp4" target="_blank"
               class="px-6 py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl font-semibold text-sm">⬇️ Download MP4</a>
            <button @click="reset()" class="px-6 py-3 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-semibold text-sm">🔄 Create Another</button>
          </div>
        </div>
      </template>
    </div>

    <!-- ╔══════════════════════════════════════════════╗ -->
    <!-- ║  I2V STEPS                                   ║ -->
    <!-- ╚══════════════════════════════════════════════╝ -->

    <!-- I2V Step 1 — Upload Photo -->
    <div x-show="mode==='i2v' && step===1" class="space-y-5 pt-2">
      <div class="text-center py-4">
        <h2 class="text-2xl font-bold text-slate-800">Upload your product photo</h2>
        <p class="text-slate-500 mt-1 text-sm">This image will be sent <strong>directly</strong> to the video model as the first frame</p>
      </div>

      <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
        <div class="dropzone rounded-xl p-8 text-center"
             @dragover.prevent="$el.classList.add('over')"
             @dragleave="$el.classList.remove('over')"
             @drop.prevent="$el.classList.remove('over'); handleI2VDrop($event)">
          <div class="text-5xl mb-3">📸</div>
          <p class="text-sm font-medium text-slate-600 mb-1">Drag & drop your product image here</p>
          <p class="text-xs text-slate-400">JPG · PNG · WEBP · up to 10MB</p>
          <input type="file" accept="image/*" x-ref="i2vFile" class="hidden" @change="handleI2VUpload($event)"/>
          <button @click="$refs.i2vFile.click()" class="mt-4 px-5 py-2.5 bg-sky-500 hover:bg-sky-600 text-white text-sm rounded-xl font-semibold transition-colors">
            Choose Image
          </button>
        </div>
        <div x-show="uploading" class="flex items-center gap-2 text-sm text-sky-600"><span class="spin">↻</span> Uploading...</div>
      </div>

      <!-- Preview -->
      <div x-show="i2vPhotoUrl" x-cloak class="bg-white rounded-2xl border border-sky-200 shadow-sm p-4">
        <div class="flex items-start gap-4">
          <img :src="i2vPhotoUrl" class="w-28 h-28 rounded-xl object-cover shadow-sm shrink-0"/>
          <div class="min-w-0 flex-1">
            <p class="text-sm font-semibold text-slate-800">✓ Photo ready</p>
            <p class="text-xs text-slate-500 mt-1">This image will be used as the visual anchor for your video. The AI will animate it according to your motion description.</p>
            <button @click="i2vPhotoUrl=null" class="mt-2 text-xs text-red-400 hover:text-red-600">Remove</button>
          </div>
        </div>
      </div>

      <div class="flex justify-between items-center">
        <button @click="reset()" class="px-5 py-3 text-slate-400 hover:text-slate-600 text-sm">← Change Mode</button>
        <button @click="step=2" :disabled="!i2vPhotoUrl"
                :class="i2vPhotoUrl?'bg-sky-500 hover:bg-sky-600 text-white':'bg-slate-200 text-slate-400 cursor-not-allowed'"
                class="px-7 py-3 rounded-xl font-semibold text-sm">Continue →</button>
      </div>
    </div>

    <!-- I2V Step 2 — Ad Script -->
    <div x-show="mode==='i2v' && step===2" class="space-y-5 pt-2">
      <div class="text-center py-4">
        <h2 class="text-2xl font-bold text-slate-800">Generate your ad script</h2>
        <p class="text-slate-500 mt-1 text-sm">Claude reads your photo's brand tonality and writes a commercial-ready video concept</p>
      </div>

      <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
        <!-- Photo thumbnail -->
        <div class="flex items-center gap-3 pb-4 border-b border-slate-100">
          <img :src="i2vPhotoUrl" class="w-16 h-16 rounded-xl object-cover shadow-sm"/>
          <div>
            <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Brand Reference</p>
            <p class="text-sm text-slate-600 mt-0.5">Claude will detect this photo's brand tonality and generate a commercial ad script</p>
          </div>
        </div>

        <!-- Brand name -->
        <div>
          <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Brand / Product name</label>
          <input x-model="brandName" type="text" placeholder="e.g. Tong Sui, Nike..."
                 class="mt-1 w-full px-4 py-2.5 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-sky-300"/>
        </div>

        <!-- Ad script -->
        <div class="space-y-4">
          <!-- Auto-generate button -->
          <div class="flex items-center justify-between">
            <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Ad Script</label>
            <button @click="autoSuggestMotion()"
                    :disabled="suggestingMotion"
                    class="flex items-center gap-1.5 px-3 py-1 bg-sky-50 hover:bg-sky-100 text-sky-600 text-xs font-medium rounded-lg border border-sky-200 transition-colors disabled:opacity-50">
              <span x-show="suggestingMotion" class="spin">↻</span>
              <span>✨ <span x-text="suggestingMotion?'Generating...':'Auto-generate from photo'"></span></span>
            </button>
          </div>

          <!-- Analyzing progress hint -->
          <div x-show="suggestingMotion" x-cloak
               class="flex items-start gap-3 px-4 py-3 bg-sky-50 border border-sky-200 rounded-xl text-sm text-sky-700">
            <span class="spin shrink-0 mt-0.5">↻</span>
            <div>
              <p class="font-medium">Analyzing photo with Claude Vision...</p>
              <p class="text-xs text-sky-500 mt-0.5">Detecting brand tonality, color palette &amp; product style — usually <strong>10–20 seconds</strong></p>
            </div>
          </div>

          <!-- Brand Concept (read-only, shown after auto-generate) -->
          <div x-show="adConcept" x-cloak class="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
            <p class="text-xs font-semibold text-emerald-600 uppercase tracking-wider mb-2">✦ Brand Concept (creative brief)</p>
            <p class="text-sm text-emerald-900 leading-relaxed" x-text="adConcept"></p>
            <p class="text-xs text-emerald-500 mt-2">This is the creative vision for your ad. The motion prompt below is what gets sent to the video AI.</p>
          </div>

          <!-- Motion Prompt (editable, sent to model) -->
          <div>
            <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Motion Prompt <span class="text-slate-400 font-normal normal-case tracking-normal">— sent to video AI</span>
            </label>
            <textarea x-model="motionPrompt" rows="4"
                      :disabled="suggestingMotion"
                      placeholder="e.g. 'Slow cinematic zoom in toward the product. Warm golden light sweeps gently left to right. Steam rises softly. Camera drifts forward with a shallow depth of field.'"
                      class="mt-1 w-full p-4 border border-slate-200 rounded-xl text-slate-700 placeholder-slate-300 focus:outline-none focus:ring-2 focus:ring-sky-300 resize-none text-sm leading-relaxed disabled:opacity-50 disabled:bg-slate-50">
            </textarea>
            <p class="text-xs text-slate-400 mt-1">Tip: use motion verbs like <em>zoom, drift, sweep, rise, rotate</em>. Keep it under 40 words for best results.</p>
          </div>
        </div>

        <!-- Platform -->
        <div>
          <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Platform</label>
          <div class="mt-1 flex gap-5">
            <label class="flex items-center gap-2 cursor-pointer text-sm text-slate-600">
              <input type="radio" x-model="platform" value="instagram_reel" class="accent-sky-500"> 📱 Instagram Reel
            </label>
            <label class="flex items-center gap-2 cursor-pointer text-sm text-slate-600">
              <input type="radio" x-model="platform" value="tiktok" class="accent-sky-500"> 🎵 TikTok
            </label>
            <label class="flex items-center gap-2 cursor-pointer text-sm text-slate-600">
              <input type="radio" x-model="platform" value="youtube" class="accent-sky-500"> ▶️ YouTube Short
            </label>
          </div>
        </div>
      </div>

      <!-- I2V Model selector -->
      <div class="space-y-2">
        <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Video Model</p>
        <div class="grid grid-cols-2 gap-2">
          <template x-for="m in i2vModels" :key="m.id">
            <div @click="selectedI2VModel=m.id"
                 :class="selectedI2VModel===m.id?'border-sky-500 bg-sky-50':'border-slate-200 hover:border-sky-300 bg-white'"
                 class="cursor-pointer border-2 rounded-xl p-3 transition-all relative">
              <div x-show="m.recommended" class="absolute -top-2 right-3 px-2 py-0.5 bg-sky-500 text-white text-xs rounded-full font-semibold">Recommended</div>
              <p class="text-xs font-bold text-slate-800" x-text="m.name"></p>
              <div class="flex items-center gap-1 mt-1">
                <template x-for="n in 5" :key="n">
                  <span :class="n<=m.quality?'text-amber-400':'text-slate-200'" class="text-xs">★</span>
                </template>
              </div>
              <div class="flex items-center justify-between mt-2">
                <span class="text-xs text-slate-500" x-text="m.price"></span>
                <span :class="m.speedColor" class="text-xs font-medium px-1.5 py-0.5 rounded-full" x-text="m.speed"></span>
              </div>
              <p class="text-xs text-slate-400 mt-1" x-text="m.desc"></p>
            </div>
          </template>
        </div>
      </div>

      <!-- I2V Duration selector -->
      <div class="space-y-2">
        <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Video Duration</p>
        <div class="grid grid-cols-2 gap-2">
          <template x-for="d in [5,10]" :key="d">
            <div @click="videoDuration=d"
                 :class="videoDuration===d?'border-sky-500 bg-sky-50':'border-slate-200 hover:border-sky-300 bg-white'"
                 class="cursor-pointer border-2 rounded-xl p-3 text-center transition-all">
              <p class="text-sm font-bold text-slate-800" x-text="d+'s'"></p>
            </div>
          </template>
        </div>
      </div>

      <!-- I2V technical note -->
      <div class="bg-sky-50 border border-sky-200 rounded-xl p-4 text-sm text-sky-800">
        <strong>How it works:</strong> Claude reads your photo's brand tonality and generates two things — a <em>creative concept</em> (what the ad communicates) and a <em>motion prompt</em> (physics instructions for the video AI). The video model animates your image using the motion prompt.
        <div class="mt-1.5 text-xs text-sky-600">Model: <code class="bg-sky-100 px-1 rounded" x-text="selectedI2VModel"></code></div>
      </div>

      <div class="flex justify-between">
        <button @click="step=1" class="px-5 py-3 text-slate-400 hover:text-slate-600 text-sm">← Back</button>
        <button @click="startGeneration('i2v')"
                :class="motionPrompt.trim()?'bg-sky-500 hover:bg-sky-600 text-white':'bg-slate-200 text-slate-400 cursor-not-allowed'"
                class="px-7 py-3 rounded-xl font-semibold text-sm flex items-center gap-2">
          🎬 Generate Ad →
        </button>
      </div>
    </div>

    <!-- I2V Step 3 — Video -->
    <div x-show="mode==='i2v' && step===3" class="pt-2">
      <template x-if="!videoReady">
        <div class="text-center py-16 space-y-6">
          <div class="bob inline-block text-5xl">🖼</div>
          <div>
            <h2 class="text-2xl font-bold text-slate-800">Animating your image</h2>
            <p class="text-slate-500 mt-2 text-sm" x-text="genMessage||'Uploading image and starting generation...'"></p>
          </div>
          <div class="max-w-xs mx-auto">
            <div class="bg-slate-200 rounded-full h-2.5 overflow-hidden">
              <div class="bg-sky-500 h-2.5 rounded-full transition-all duration-700" :style="'width:'+Math.max(5,genProgress)+'%'"></div>
            </div>
            <p class="text-xs text-slate-400 mt-2" x-text="genProgress+'%'"></p>
          </div>
          <div class="max-w-sm mx-auto bg-sky-50 rounded-xl p-4 text-left">
            <ul class="space-y-2 text-sm">
              <li :class="genProgress>=0?'text-sky-600':'text-slate-400'" class="flex gap-2">
                <span x-text="genProgress>=0?'✓':'○'"></span> Preparing your image
              </li>
              <li :class="genProgress>=10?'text-sky-600':'text-slate-400'" class="flex gap-2">
                <span x-text="genProgress>=10?'✓':'○'"></span> Uploading to fal.ai
              </li>
              <li :class="genProgress>=20?'text-sky-500':'text-slate-400'" class="flex items-center gap-2">
                <span x-show="genProgress>=20&&genProgress<95" class="spin text-sky-400">↻</span>
                <span x-show="genProgress>=95">✓</span>
                <span x-show="genProgress<20">○</span>
                Animating image (1–3 min)
              </li>
              <li :class="genProgress>=95?'text-sky-600':'text-slate-400'" class="flex gap-2">
                <span x-text="genProgress>=95?'✓':'○'"></span> Finalizing video
              </li>
            </ul>
          </div>
        </div>
      </template>
      <template x-if="videoReady">
        <div class="space-y-6">
          <div class="text-center py-4"><div class="text-4xl mb-3">🎉</div>
            <h2 class="text-2xl font-bold text-slate-800">Your video is ready!</h2>
            <p class="text-slate-500 mt-1 text-sm">Your product photo — now animated</p>
          </div>
          <div x-show="mockMode" class="bg-yellow-50 border border-yellow-200 rounded-xl p-4 text-sm text-yellow-800">
            <strong>Mock mode</strong> — Set <code class="bg-yellow-100 px-1 rounded">VIDEO_BACKEND=fal</code> + <code class="bg-yellow-100 px-1 rounded">FAL_API_KEY</code> for real video.
          </div>
          <!-- Side by side: original photo + video -->
          <div class="grid grid-cols-2 gap-4 max-w-sm mx-auto">
            <div class="space-y-2">
              <p class="text-xs text-slate-400 font-medium text-center">Original Photo</p>
              <img :src="i2vPhotoUrl" class="w-full aspect-[9/16] object-cover rounded-xl shadow-sm"/>
            </div>
            <div class="space-y-2">
              <p class="text-xs text-slate-400 font-medium text-center">Generated Video</p>
              <div x-show="videoUrl" class="bg-slate-900 rounded-xl overflow-hidden shadow">
                <video :src="videoUrl" controls autoplay loop muted playsinline class="w-full aspect-[9/16] object-cover"></video>
              </div>
              <div x-show="!videoUrl&&mockMode" class="w-full aspect-[9/16] bg-gradient-to-br from-sky-100 to-blue-100 rounded-xl flex flex-col items-center justify-center gap-2 border-2 border-dashed border-sky-300">
                <span class="text-3xl">🎬</span><p class="text-xs text-slate-400 text-center px-2">Real video with FAL_API_KEY</p>
              </div>
            </div>
          </div>
          <div class="flex justify-center gap-3">
            <a x-show="videoUrl" :href="videoUrl" download="ad-video-i2v.mp4" target="_blank"
               class="px-6 py-3 bg-sky-500 hover:bg-sky-600 text-white rounded-xl font-semibold text-sm">⬇️ Download MP4</a>
            <button @click="reset()" class="px-6 py-3 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-semibold text-sm">🔄 Create Another</button>
          </div>
        </div>
      </template>
    </div>

  </div><!-- content -->
</div><!-- x-data -->

<script>
function adAgent() {
  return {
    // ── Shared state ─────────────────────────────
    mode: null,       // null | 't2v' | 'i2v'
    step: 1,
    error: null,
    platform: 'instagram_reel',
    brandName: '',
    jobId: null,
    videoUrl: null,
    videoReady: false,
    mockMode: false,
    genProgress: 0,
    genMessage: '',
    pollInterval: null,

    // ── T2V state ────────────────────────────────
    pageUrl: '',
    brandInfo: null,
    scraping: false,
    allPhotos: [],
    selectedPhotoUrl: null,
    uploading: false,
    brief: '',
    storyboard: null,
    loading: false,
    selectedT2VModel: 'fal-ai/wan/v2.1/t2v-14b',
    videoDuration: 5,

    // ── I2V state ────────────────────────────────
    i2vPhotoUrl: null,
    adConcept: '',
    motionPrompt: '',
    suggestingMotion: false,
    selectedI2VModel: 'fal-ai/wan/v2.2-a14b/image-to-video',

    // ── Model catalogues ─────────────────────────
    t2vModels: [
      { id: 'fal-ai/wan/v2.1/t2v-14b',                     name: 'Wan 2.1 T2V 14B',  quality: 4, price: '~$0.05/video', speed: 'Medium', speedColor: 'text-blue-600 bg-blue-50',    desc: 'Balanced quality & cost',  recommended: true  },
      { id: 'fal-ai/kling-video/v1.6/pro/text-to-video',   name: 'Kling 1.6 Pro',    quality: 5, price: '~$0.14/video', speed: 'Slow',   speedColor: 'text-purple-600 bg-purple-50', desc: 'Best commercial quality',  recommended: false },
      { id: 'fal-ai/minimax/video-01',                     name: 'MiniMax Video-01',  quality: 4, price: '~$0.10/video', speed: 'Medium', speedColor: 'text-orange-600 bg-orange-50', desc: 'High fidelity motion',     recommended: false },
    ],
    i2vModels: [
      { id: 'fal-ai/wan/v2.2-a14b/image-to-video',       name: 'Wan 2.2 14B',    quality: 4, price: '~$0.08/video', speed: 'Medium', speedColor: 'text-blue-600 bg-blue-50',    desc: 'Balanced quality & cost',  recommended: true  },
      { id: 'fal-ai/kling-video/v1.6/pro/image-to-video', name: 'Kling 1.6 Pro',  quality: 5, price: '~$0.14/video', speed: 'Slow',   speedColor: 'text-purple-600 bg-purple-50', desc: 'Best commercial quality', recommended: false },
      { id: 'fal-ai/minimax/video-01-live',               name: 'MiniMax V01',    quality: 4, price: '~$0.10/video', speed: 'Medium', speedColor: 'text-orange-600 bg-orange-50', desc: 'High fidelity motion',    recommended: false },
    ],

    // ── Mode selection ────────────────────────────
    selectMode(m) {
      this.mode = m;
      this.step = 1;
      this.error = null;
    },

    // ── T2V: scrape brand page ────────────────────
    async scrapePage() {
      if (!this.pageUrl.trim() || this.scraping) return;
      this.scraping = true; this.error = null;
      try {
        const r = await fetch('/api/scrape', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({url: this.pageUrl})
        });
        if (!r.ok) { const e=await r.json(); throw new Error(e.detail||'Scrape failed'); }
        this.brandInfo = await r.json();
        this.brandName = this.brandInfo.brand_name || '';
        if (this.brandInfo.description && !this.brief)
          this.brief = `Create a compelling 5-second video ad for ${this.brandInfo.brand_name}. ${this.brandInfo.description}`;
        const scraped = (this.brandInfo.images||[]).map(url => ({url, label:'From website', source:'scraped', failed:false}));
        this.allPhotos = [...scraped, ...this.allPhotos.filter(p=>p.source==='upload')];
      } catch(e) { this.error = e.message; }
      finally { this.scraping = false; }
    },

    // ── T2V: photo upload ─────────────────────────
    async handleFileUpload(e) { await this._uploadFiles(Array.from(e.target.files)); e.target.value=''; },
    async handleDrop(e) { await this._uploadFiles(Array.from(e.dataTransfer.files).filter(f=>f.type.startsWith('image/'))); },
    async _uploadFiles(files) {
      if (!files.length) return;
      this.uploading = true; this.error = null;
      for (const f of files) {
        try {
          const fd = new FormData(); fd.append('file', f);
          const r = await fetch('/api/upload', {method:'POST', body:fd});
          if (!r.ok) { const e=await r.json(); throw new Error(e.detail); }
          const d = await r.json();
          this.allPhotos = [{url:d.url, label:f.name, source:'upload', failed:false}, ...this.allPhotos];
          this.selectedPhotoUrl = d.url;
        } catch(e) { this.error = `Upload failed: ${e.message}`; }
      }
      this.uploading = false;
    },

    // ── T2V: generate storyboard ──────────────────
    async generateStoryboard() {
      if (!this.brief.trim() || this.loading) return;
      this.loading = true; this.error = null;
      try {
        const r = await fetch('/api/storyboard', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({brief:this.brief, platform:this.platform,
                                photo_url:this.selectedPhotoUrl, brand_name:this.brandName||'the brand',
                                duration:this.videoDuration})
        });
        if (!r.ok) { const e=await r.json(); throw new Error(e.detail||'Failed'); }
        this.storyboard = await r.json();
        this.step = 4;
      } catch(e) { this.error = e.message; }
      finally { this.loading = false; }
    },

    // ── I2V: upload ───────────────────────────────
    async handleI2VUpload(e) {
      const f = e.target.files[0]; if (!f) return;
      await this._uploadI2VFile(f); e.target.value='';
    },
    async handleI2VDrop(e) {
      const f = Array.from(e.dataTransfer.files).find(f=>f.type.startsWith('image/')); if (!f) return;
      await this._uploadI2VFile(f);
    },
    async _uploadI2VFile(f) {
      this.uploading = true; this.error = null;
      try {
        const fd = new FormData(); fd.append('file', f);
        const r = await fetch('/api/upload', {method:'POST', body:fd});
        if (!r.ok) { const e=await r.json(); throw new Error(e.detail); }
        const d = await r.json();
        this.i2vPhotoUrl = d.url;
      } catch(e) { this.error = `Upload failed: ${e.message}`; }
      finally { this.uploading = false; }
    },

    // ── I2V: auto-suggest motion prompt ──────────
    async autoSuggestMotion() {
      if (!this.i2vPhotoUrl || this.suggestingMotion) return;
      this.suggestingMotion = true; this.error = null;
      try {
        const r = await fetch('/api/suggest-motion', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({photo_url:this.i2vPhotoUrl, brand_name:this.brandName, platform:this.platform})
        });
        if (!r.ok) { const e=await r.json(); throw new Error(e.detail||'Failed'); }
        const d = await r.json();
        this.adConcept    = d.concept      || '';
        this.motionPrompt = d.motion_prompt || d.concept || '';
      } catch(e) { this.error = e.message; }
      finally { this.suggestingMotion = false; }
    },

    // ── Shared: start generation ──────────────────
    async startGeneration(m) {
      const targetStep = m === 'i2v' ? 3 : 5;
      this.step = targetStep;
      this.genProgress = 5; this.videoReady = false;
      this.genMessage = 'Preparing...'; this.error = null;
      try {
        const isI2V = m === 'i2v';
        const body = {
          mode: isI2V ? 'i2v' : 't2v',
          brief: isI2V ? this.motionPrompt : this.brief,
          photo_url: isI2V ? this.i2vPhotoUrl : this.selectedPhotoUrl,
          storyboard: m === 't2v' ? this.storyboard : null,
          model_id: isI2V ? this.selectedI2VModel : this.selectedT2VModel,
          duration: this.videoDuration,
          brand_concept: isI2V ? this.adConcept : (this.storyboard?.overall_prompt || this.brief),
          platform: this.platform,
        };
        const r = await fetch('/api/generate', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify(body)
        });
        if (!r.ok) { const e=await r.json(); throw new Error(e.detail||'Failed'); }
        const d = await r.json();
        this.jobId = d.job_id;
        this.genProgress = 10;
        this.startPolling(m === 'i2v' ? 3 : 5);
      } catch(e) {
        this.error = e.message;
        this.step = m === 'i2v' ? 2 : 4;
      }
    },

    startPolling(doneStep) {
      this.pollInterval = setInterval(async () => {
        try {
          const r = await fetch('/api/status/'+this.jobId);
          if (!r.ok) {
            clearInterval(this.pollInterval);
            const msg = r.status === 404
              ? 'Job not found — server may have restarted. Please generate again.'
              : `Status check failed (HTTP ${r.status})`;
            this.error = msg;
            this.step = doneStep === 3 ? 2 : 4;
            return;
          }
          const d = await r.json();
          if (d.message) this.genMessage = d.message;
          if (d.progress > this.genProgress) this.genProgress = d.progress;
          else if (this.genProgress < 88) this.genProgress += 1;
          if (d.status === 'completed') {
            clearInterval(this.pollInterval);
            this.genProgress = 100;
            this.videoUrl = d.video_url;
            this.mockMode = d.mock||false;
            setTimeout(() => { this.videoReady = true; }, 500);
          } else if (d.status === 'failed') {
            clearInterval(this.pollInterval);
            this.error = d.error||'Generation failed';
            this.step = doneStep === 3 ? 2 : 4;
          }
        } catch(e) { console.error('poll error', e); }
      }, 3000);
    },

    // ── Reset ─────────────────────────────────────
    reset() {
      if (this.pollInterval) clearInterval(this.pollInterval);
      Object.assign(this, {
        mode:null, step:1, error:null, platform:'instagram_reel', brandName:'',
        jobId:null, videoUrl:null, videoReady:false, mockMode:false,
        genProgress:0, genMessage:'', pollInterval:null,
        // t2v
        pageUrl:'', brandInfo:null, scraping:false,
        allPhotos:[], selectedPhotoUrl:null, uploading:false, brief:'', storyboard:null, loading:false,
        selectedT2VModel:'fal-ai/wan/v2.1/t2v-14b', videoDuration:5,
        // i2v
        i2vPhotoUrl:null, adConcept:'', motionPrompt:'', suggestingMotion:false,
        selectedI2VModel:'fal-ai/wan/v2.2-a14b/image-to-video',
      });
    },
  };
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=(port == 8000))
