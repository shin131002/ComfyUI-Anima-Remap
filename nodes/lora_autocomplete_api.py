"""
Backend for the LoRA autocomplete shown in the Anima LoRA Tag Loader text box
(web/anima_lora_autocomplete.js).

Three read-only routes:
  GET /anima_remap/loras                 -> every LoRA ComfyUI can see, for filtering in the browser
  GET /anima_remap/lora_info?path=...    -> trigger words + whether a preview image exists
  GET /anima_remap/lora_preview?path=... -> the preview image itself

`path` is always the relative path exactly as folder_paths.get_filename_list("loras")
reports it (forward slashes). It is resolved through folder_paths.get_full_path(), which
only ever looks inside the configured loras folders, so a crafted `path` cannot be used
to read arbitrary files.
"""

import json
import logging
import os

import folder_paths

from .anima_common import is_remap_cache_filename

logger = logging.getLogger("AnimaRemap")

LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt", ".sft")

# Checked in this order. ".preview.*" is what LoRA Manager writes; the bare
# "<name>.png/.jpg" form is the older Civitai Helper / pysssss convention.
# Still images win over videos (same priority as the random loaders' preview).
PREVIEW_SUFFIXES = (
    ".preview.png", ".preview.webp", ".preview.jpg", ".preview.jpeg",
    ".png", ".webp", ".jpg", ".jpeg",
    ".preview.mp4", ".preview.webm", ".mp4", ".webm",
)
VIDEO_EXTS = (".mp4", ".webm")


def _strip_ext(name):
    low = name.lower()
    for ext in LORA_EXTENSIONS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _full_path(rel_path):
    if not rel_path:
        return None
    return folder_paths.get_full_path("loras", rel_path)


def _find_preview(full_path):
    base = _strip_ext(full_path)
    for suffix in PREVIEW_SUFFIXES:
        candidate = base + suffix
        if os.path.isfile(candidate):
            return candidate
    return None


def _split_trigger_patterns(patterns):
    """
    trainedWords is a list of patterns, each of which may itself be a comma-separated
    group ("aaa, bbb"). Flatten, trim, and de-duplicate case-insensitively while keeping
    the original order -- same rule RandomLoRALoader uses for its combined mode.
    """
    out, seen = [], set()
    for pattern in patterns or []:
        if not isinstance(pattern, str):
            continue
        for tag in pattern.split(","):
            tag = tag.strip()
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                out.append(tag)
    return out


def _trigger_words_from_json(data):
    if not isinstance(data, dict):
        return []
    # LoRA Manager (.metadata.json) nests the Civitai payload under "civitai";
    # Civitai Helper (.civitai.info / .info) stores the payload itself, so
    # trainedWords is top-level there.
    words = (data.get("civitai") or {}).get("trainedWords")
    if not words:
        words = data.get("trainedWords")
    return _split_trigger_patterns(words)


def _trigger_words_from_embedded(full_path):
    """
    Only fields that are explicitly trigger words. ss_tag_frequency (training-tag
    statistics) and ss_output_name (the model's own name) are deliberately ignored:
    they aren't trigger words, and inserting them into a prompt just adds noise.
    """
    if not full_path.lower().endswith(".safetensors"):
        return []
    try:
        from safetensors import safe_open
        with safe_open(full_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
    except Exception as e:
        logger.debug(f"[AnimaRemap autocomplete] embedded metadata read failed for {full_path}: {e}")
        return []
    trigger = meta.get("modelspec.trigger_word")
    return _split_trigger_patterns([trigger]) if trigger else []


def get_trigger_words(full_path):
    base = _strip_ext(full_path)
    for sidecar in (base + ".metadata.json", base + ".civitai.info", base + ".info"):
        if not os.path.isfile(sidecar):
            continue
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                words = _trigger_words_from_json(json.load(f))
        except Exception as e:
            logger.debug(f"[AnimaRemap autocomplete] failed to read {sidecar}: {e}")
            continue
        if words:
            return words
    return _trigger_words_from_embedded(full_path)


def list_loras():
    """
    [{"n": basename-without-extension, "p": relative path, "d": basename is shared by
      more than one file}, ...]

    "d" lets the browser insert the bare filename (the canonical form) normally, and fall
    back to "subfolder/name" only when the bare name would be ambiguous.
    """
    try:
        rel_paths = folder_paths.get_filename_list("loras")
    except Exception as e:
        logger.warning(f"[AnimaRemap autocomplete] could not list loras: {e}")
        return []

    entries, counts = [], {}
    for rel in rel_paths:
        rel_norm = rel.replace("\\", "/")
        stem = _strip_ext(rel_norm)
        base = os.path.basename(stem)
        if is_remap_cache_filename(base):
            continue
        entries.append((base, rel_norm))
        counts[base.lower()] = counts.get(base.lower(), 0) + 1

    return [{"n": b, "p": p, "d": counts[b.lower()] > 1} for b, p in entries]


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as e:
        logger.warning(f"[AnimaRemap autocomplete] routes not registered (no PromptServer): {e}")
        return

    routes = PromptServer.instance.routes

    @routes.get("/anima_remap/loras")
    async def _loras(request):
        return web.json_response(list_loras())

    @routes.get("/anima_remap/lora_info")
    async def _lora_info(request):
        full = _full_path(request.query.get("path", ""))
        if full is None:
            return web.json_response({"error": "not found"}, status=404)
        preview = _find_preview(full)
        if preview is None:
            preview_type = None
        elif preview.lower().endswith(VIDEO_EXTS):
            # The browser shows the first frame of a paused <video>, so no server-side
            # decoding (and no opencv dependency) is needed, unlike the random loaders,
            # which must turn the frame into an IMAGE tensor.
            preview_type = "video"
        else:
            preview_type = "image"
        return web.json_response({
            "trigger_words": get_trigger_words(full),
            "preview_type": preview_type,
        })

    @routes.get("/anima_remap/lora_preview")
    async def _lora_preview(request):
        full = _full_path(request.query.get("path", ""))
        preview = _find_preview(full) if full else None
        if preview is None:
            return web.Response(status=404)
        return web.FileResponse(preview, headers={"Cache-Control": "max-age=300"})


_register_routes()
