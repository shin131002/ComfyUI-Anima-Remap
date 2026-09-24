"""
anima_filtered_random_lora_loader.py

Anima-only port of shin131002/RandomLoRALoader's "Filtered Random LoRA
Loader" (single folder + keyword filtering), with automatic old-Anima
(28-block) / Anima-2.9B (40-block) detection and block-index remapping
reused from anima_common.py / lora_remap_extended_anima.py.

See anima_random_lora_loader.py's module docstring for the design notes
shared with that node (no LBW, no save_remapped/on-disk caching, folder
scan excludes "_29Bremap"/"_29Bremap_ext" cache files, remap settings are
a single shared set rather than per-selection).
"""

import os
import json
import random
import re

import folder_paths
import comfy.sd
import comfy.utils

from .anima_common import (
    computable,
    get_lora_block_count,
    remap_key,
    build_base_to_target,
    build_insertion_neighbors,
    list_manifest_choices,
    resolve_manifest,
    get_model_block_count,
    is_remap_cache_filename,
    AnimaBlockMismatchError,
)
from .lora_remap_extended_anima import (
    group_by_base_index,
    build_blended_extension,
)

try:
    from safetensors.torch import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("[AnimaFilteredRandomLoRALoader] safetensors not available, embedded metadata reading disabled")


class AnimaFilteredRandomLoRALoader:
    """Anima 28/40-block random LoRA selector/applier (1 folder + keyword filter)."""

    _metadata_cache = {}
    _opencv_warning_shown = False

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_manifest_choices()
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "additional_prompt_positive": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "Additional positive prompt"
                }),
                "additional_prompt_negative": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "Additional negative prompt"
                }),
                "lora_folder_path": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "Path to LoRA folder"
                }),
                "include_subfolders": ("BOOLEAN", {"default": True}),
                "unique_by_filename": ("BOOLEAN", {"default": True}),
                "keyword_filter": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "Keywords (space-separated, e.g., 'style anime' or \"anime style\" red)"
                }),
                "filter_mode": (["AND", "OR"], {"default": "AND"}),
                "search_in_metadata": ("BOOLEAN", {"default": False}),
                "model_strength": ("STRING", {
                    "default": "1.0", "multiline": False,
                    "placeholder": "e.g., 1.0 or 0.6-0.9"
                }),
                "clip_strength": ("STRING", {
                    "default": "1.0", "multiline": False,
                    "placeholder": "e.g., 1.0 or 0.6-0.9"
                }),
                "num_loras": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1}),
                "trigger_word_source": (
                    ["json_combined", "json_random", "json_sample_prompt", "metadata"],
                    {"default": "json_combined"}
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # -- Anima block remap --
                "auto_remap": ("BOOLEAN", {"default": True}),
                "extend_to_new_layers": ("BOOLEAN", {"default": False}),
                "blend_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "extend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "manifest": (manifests,),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "CONDITIONING", "CONDITIONING", "IMAGE", "STRING")
    # Slots 0-6 keep their pre-v1.4.0 positions and types, so saved workflows stay wired
    # correctly (they link outputs by slot index). What changed in v1.4.0:
    #   positive_text (slot 2) no longer contains <lora:...> tags -- it is exactly the text
    #     encoded into `positive`, so it can go straight into a text encoder.
    #   lora_text (slot 7, new) carries the old positive_text content, tags included,
    #     for recording which LoRAs were picked.
    RETURN_NAMES = ("MODEL", "CLIP", "positive_text", "negative_text", "positive", "negative", "preview", "lora_text")
    FUNCTION = "load_loras"
    CATEGORY = "loaders/anima/random"

    # ------------------------------------------------------------------
    # Folder scan / filter / selection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_remap_cache_file(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return is_remap_cache_filename(stem)

    def _find_lora_files(self, folder_path, include_subfolders):
        if not os.path.exists(folder_path):
            print(f"[AnimaFilteredRandomLoRALoader] Folder does not exist: {folder_path}")
            return []
        extensions = ('.safetensors', '.pt', '.ckpt')
        found = []
        if include_subfolders:
            # followlinks=True matches what ComfyUI's own folder scan
            # (folder_paths.recursive_search) does. Without it, a subfolder that
            # is a symlink is skipped entirely, so LoRAs that show up fine in
            # ComfyUI's native dropdown would silently never be picked here.
            #
            # The realpath set is the price of following links: it stops a
            # symlink loop from hanging the scan, and also keeps a folder
            # reachable through two different links from being scanned twice.
            seen_dirs = set()
            for root, dirs, files in os.walk(folder_path, followlinks=True):
                real_root = os.path.realpath(root)
                if real_root in seen_dirs:
                    dirs[:] = []
                    continue
                seen_dirs.add(real_root)
                for file in files:
                    if file.lower().endswith(extensions):
                        found.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(extensions):
                    found.append(file_path)
        lora_files = [f for f in found if not self._is_remap_cache_file(f)]
        excluded = len(found) - len(lora_files)
        if excluded:
            print(f"[AnimaFilteredRandomLoRALoader] Excluded {excluded} cached remap file(s) (_animaremap*) from candidates")
        # Sorted so a given seed picks the same LoRAs on any machine (filesystem
        # enumeration order differs between OSes/filesystems). Same as
        # RandomLoRALoader v1.3.0.
        return sorted(lora_files)

    def _unique_by_filename(self, lora_files):
        seen, unique_files = {}, []
        for file_path in lora_files:
            filename = os.path.basename(file_path)
            if filename not in seen:
                seen[filename] = file_path
                unique_files.append(file_path)
            else:
                print(f"[AnimaFilteredRandomLoRALoader] Duplicate filename skipped: {filename}")
        return unique_files

    def _parse_keywords(self, keyword_filter):
        if not keyword_filter.strip():
            return []
        pattern = r'"([^"]+)"|(\S+)'
        matches = re.findall(pattern, keyword_filter)
        keywords = [m[0] if m[0] else m[1] for m in matches]
        return [kw.lower() for kw in keywords if kw]

    def _filter_lora_files(self, lora_files, keyword_filter, filter_mode, search_in_metadata):
        if not keyword_filter.strip():
            return lora_files
        keywords = self._parse_keywords(keyword_filter)
        if not keywords:
            return lora_files

        if not search_in_metadata:
            filtered = []
            for lora_path in lora_files:
                filename = os.path.splitext(os.path.basename(lora_path))[0].lower()
                match = all(kw in filename for kw in keywords) if filter_mode == "AND" else any(kw in filename for kw in keywords)
                if match:
                    filtered.append(lora_path)
            return filtered

        total = len(lora_files)
        filtered = []
        print(f"[AnimaFilteredRandomLoRALoader] Building metadata cache for {total} files...")
        for i, lora_path in enumerate(lora_files):
            filename = os.path.splitext(os.path.basename(lora_path))[0].lower()
            search_target = filename
            metadata_keywords = self._get_metadata_keywords(lora_path)
            if metadata_keywords:
                search_target = f"{filename} {metadata_keywords}"
            match = all(kw in search_target for kw in keywords) if filter_mode == "AND" else any(kw in search_target for kw in keywords)
            if match:
                filtered.append(lora_path)
        print(f"[AnimaFilteredRandomLoRALoader] Filtered {len(filtered)}/{total} files.")
        return filtered

    def _get_metadata_keywords(self, lora_path):
        if lora_path in self._metadata_cache:
            return self._metadata_cache[lora_path]
        metadata = self._load_json_metadata(lora_path)
        parts = []
        if metadata:
            if metadata.get("model_name"):
                parts.append(metadata["model_name"])
            civitai = metadata.get("civitai", {})
            if civitai.get("name"):
                parts.append(civitai["name"])
            for pattern in civitai.get("trainedWords", []):
                parts.extend(w.strip() for w in pattern.split(','))
            model_info = civitai.get("model", {})
            if model_info.get("name"):
                parts.append(model_info["name"])
            parts.extend(model_info.get("tags", []))
            parts.extend(metadata.get("tags", []))
        unique = list(dict.fromkeys(parts))
        keywords = " ".join(unique).lower()
        self._metadata_cache[lora_path] = keywords
        return keywords

    def _select_random_loras(self, lora_files, num_loras, seed):
        if not lora_files:
            return []
        random.seed(seed)
        if num_loras <= len(lora_files):
            return random.sample(lora_files, num_loras)
        selected = lora_files.copy()
        for _ in range(num_loras - len(lora_files)):
            selected.append(random.choice(lora_files))
        random.shuffle(selected)
        return selected

    def _parse_strength(self, strength_str):
        strength_str = str(strength_str).strip()
        range_pattern = r'^(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)$'
        match = re.match(range_pattern, strength_str)
        if match:
            try:
                min_val, max_val = round(float(match.group(1)), 1), round(float(match.group(2)), 1)
                if min_val > max_val:
                    raise ValueError("min must be <= max")
                values, current = [], min_val
                while current <= max_val + 0.01:
                    values.append(round(current, 1))
                    current += 0.1
                return random.choice(values) if values else 1.0
            except Exception as e:
                print(f"[AnimaFilteredRandomLoRALoader] Strength parse error ({strength_str}): {e}, using 1.0")
                return 1.0
        try:
            return float(strength_str)
        except Exception:
            print(f"[AnimaFilteredRandomLoRALoader] Could not parse strength '{strength_str}', using 1.0")
            return 1.0

    # ------------------------------------------------------------------
    # Metadata / trigger words (architecture-agnostic, ported as-is)
    # ------------------------------------------------------------------

    def _load_json_metadata(self, lora_path):
        base_name = os.path.splitext(lora_path)[0]
        for suffix in (".metadata.json", ".info"):
            path = f"{base_name}{suffix}"
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[AnimaFilteredRandomLoRALoader] JSON read error ({path}): {e}")
        return self._load_embedded_metadata(lora_path)

    def _load_embedded_metadata(self, lora_path):
        if not SAFETENSORS_AVAILABLE or not os.path.exists(lora_path):
            return None
        # safe_open only reads .safetensors; .pt/.ckpt candidates would otherwise
        # log a spurious read error every time they are picked. Same as
        # RandomLoRALoader v1.3.0.
        if not lora_path.lower().endswith(".safetensors"):
            return None
        try:
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if not metadata:
                    return None
                if "ss_tag_frequency" in metadata:
                    try:
                        tag_freq = json.loads(metadata.get("ss_tag_frequency", "{}"))
                        all_tags = []
                        for dataset_tags in tag_freq.values():
                            all_tags.extend(dataset_tags.keys())
                        unique_tags = list(dict.fromkeys(all_tags))
                        if unique_tags:
                            return {"civitai": {"trainedWords": [", ".join(unique_tags[:20])]}}
                    except json.JSONDecodeError:
                        pass
                if "modelspec.trigger_word" in metadata:
                    return {"civitai": {"trainedWords": [metadata["modelspec.trigger_word"]]}}
                if "ss_output_name" in metadata:
                    return {"civitai": {"trainedWords": [metadata["ss_output_name"]]}}
                return None
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Embedded metadata read error ({lora_path}): {e}")
            return None

    def _get_trigger_words_combined(self, lora_path):
        data = self._load_json_metadata(lora_path)
        if not data:
            return ""
        trained_words = data.get("civitai", {}).get("trainedWords", [])
        all_tags = []
        for pattern in trained_words:
            all_tags.extend(t.strip() for t in pattern.split(','))
        seen, unique = set(), []
        for tag in all_tags:
            if tag and tag.lower() not in seen:
                unique.append(tag)
                seen.add(tag.lower())
        return ", ".join(unique)

    def _get_trigger_words_random(self, lora_path):
        data = self._load_json_metadata(lora_path)
        if not data:
            return ""
        trained_words = data.get("civitai", {}).get("trainedWords", [])
        if not trained_words:
            return ""
        tags = [t.strip() for t in random.choice(trained_words).split(',')]
        seen, unique = set(), []
        for tag in tags:
            if tag and tag.lower() not in seen:
                unique.append(tag)
                seen.add(tag.lower())
        return ", ".join(unique)

    def _get_sample_prompt_from_json(self, lora_path):
        data = self._load_json_metadata(lora_path)
        if not data:
            return "", ""
        images = data.get("civitai", {}).get("images", [])
        valid = [img for img in images if img.get("meta")]
        if not valid:
            return "", ""
        meta = random.choice(valid).get("meta", {})
        positive = self._remove_lora_syntax(meta.get("prompt", ""))
        negative = self._remove_lora_syntax(meta.get("negativePrompt", ""))
        return positive.strip(), negative.strip()

    def _get_trigger_words_from_embedded(self, lora_path):
        data = self._load_embedded_metadata(lora_path)
        if not data:
            return ""
        trained_words = data.get("civitai", {}).get("trainedWords", [])
        return trained_words[0] if trained_words else ""

    def _remove_lora_syntax(self, text):
        if not text:
            return text
        text = re.sub(r'<lora:[^>]+>', '', text)
        text = re.sub(r',(\s*,)+', ',', text)
        return text.strip().strip(',').strip()

    # ------------------------------------------------------------------
    # Preview images (architecture-agnostic, ported as-is)
    # ------------------------------------------------------------------

    def _load_preview_image_as_tensor(self, lora_path):
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            return None
        base_name = os.path.splitext(os.path.basename(lora_path))[0]
        folder = os.path.dirname(lora_path)
        static_exts = ('.png', '.jpg', '.jpeg')
        animated_exts = ('.gif', '.webp')
        video_exts = ('.mp4', '.webm', '.avi', '.mov')
        try:
            files = [f for f in os.listdir(folder) if f.lower().startswith(base_name.lower())]
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Folder read error: {e}")
            return None
        if not files:
            return None

        def priority(fn):
            lower = fn.lower()
            if lower.endswith(static_exts):
                return 0
            if lower.endswith(animated_exts):
                return 1
            if lower.endswith(video_exts):
                return 2
            return 999
        files.sort(key=priority)

        for file in files:
            path = os.path.join(folder, file)
            lower = file.lower()
            if lower.endswith(static_exts):
                img = self._load_static_image(path)
            elif lower.endswith(animated_exts):
                img = self._load_animated_first_frame(path)
            elif lower.endswith(video_exts):
                img = self._load_video_first_frame(path)
            else:
                continue
            if img is not None:
                return img
        return None

    def _load_static_image(self, path):
        try:
            from PIL import Image
            return self._resize_and_convert(Image.open(path).convert('RGB'))
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Static image load error ({path}): {e}")
            return None

    def _load_animated_first_frame(self, path):
        try:
            from PIL import Image
            img = Image.open(path)
            if hasattr(img, 'seek'):
                img.seek(0)
            return self._resize_and_convert(img.convert('RGB'))
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Animated image load error ({path}): {e}")
            return None

    def _load_video_first_frame(self, path):
        try:
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            return self._resize_and_convert(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except ImportError:
            if not AnimaFilteredRandomLoRALoader._opencv_warning_shown:
                print("[AnimaFilteredRandomLoRALoader] Video preview requires 'pip install opencv-python'; "
                      "static/animated images still work without it.")
                AnimaFilteredRandomLoRALoader._opencv_warning_shown = True
            return None
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Video load error ({path}): {e}")
            return None

    def _resize_and_convert(self, img):
        try:
            from PIL import Image
            import numpy as np
            import torch
            max_size = 1240
            width, height = img.size
            if max(width, height) != max_size:
                if width > height:
                    new_w, new_h = max_size, int(height * (max_size / width))
                else:
                    new_h, new_w = max_size, int(width * (max_size / height))
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                img = img.resize((new_w, new_h), resample)
            return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Image resize error: {e}")
            return None

    def _generate_preview_batch(self, preview_images):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return None
        if not preview_images:
            return torch.zeros((1, 1240, 1240, 3), dtype=torch.float32)
        target = 1240
        padded = []
        for img in preview_images:
            h, w, c = img.shape
            if h == target and w == target:
                padded.append(img)
                continue
            pad_h, pad_w = target - h, target - w
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2
            chw = img.permute(2, 0, 1)
            chw = F.pad(chw, (left, right, top, bottom), value=0)
            padded.append(chw.permute(1, 2, 0))
        return torch.stack(padded, dim=0)

    # ------------------------------------------------------------------
    # Anima block remap (28<->40), reusing anima_common / extended-blend logic
    # ------------------------------------------------------------------

    def _apply_lora_with_remap(self, model, clip, lora_path, model_strength, clip_strength,
                                auto_remap, manifest_choice, model_block_count,
                                extend_to_new_layers, blend_ratio, extend_strength):
        try:
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Failed to load LoRA {lora_path}: {e}")
            return model, clip

        lora_block_count = get_lora_block_count(lora_sd.keys())

        # A LoRA that references more blocks than the connected model has
        # cannot be remapped DOWN (there's nowhere for its high-index
        # tensors to go). Checked regardless of auto_remap -- it's about
        # whether the LoRA can be safely applied at all -- so this is a
        # hard stop rather than a silent partial application. See
        # AnimaBlockMismatchError.
        if (model_block_count is not None and lora_block_count is not None
                and lora_block_count > model_block_count):
            raise AnimaBlockMismatchError(
                f"[AnimaFilteredRandomLoRALoader] '{os.path.basename(lora_path)}' references "
                f"{lora_block_count} blocks but the connected model only has "
                f"{model_block_count}. It cannot be remapped down onto a smaller model, so "
                f"this run is stopped rather than silently applying a partial/broken LoRA. "
                f"Remove this file from the folder/filter, or connect the matching model."
            )

        # Resolve the manifest for THIS SPECIFIC LoRA's (block_count -> model's
        # block_count) pair -- a mixed-generation folder can hold e.g. both
        # 28-block and 40-block LoRAs that each need a different manifest
        # against the same connected model.
        manifest_data = None
        needs_remap = (
            auto_remap
            and model_block_count is not None
            and lora_block_count is not None
            and lora_block_count < model_block_count
        )
        if needs_remap:
            manifest_data = resolve_manifest(manifest_choice, lora_block_count, model_block_count)

        base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
        neighbors = build_insertion_neighbors(manifest_data) if manifest_data else {}

        try:
            if base_to_target:
                remapped = {}
                for k, v in lora_sd.items():
                    new_k, _ = remap_key(k, base_to_target)
                    if new_k is None:
                        continue
                    remapped[new_k] = v

                if extend_to_new_layers and neighbors:
                    groups = group_by_base_index(lora_sd)
                    extension = build_blended_extension(groups, neighbors, blend_ratio)
                    for k, v in extension.items():
                        remapped[k] = computable(v) * extend_strength

                lora_sd = remapped

            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, lora_sd, model_strength, clip_strength
            )
            return model_lora, clip_lora
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Failed to apply LoRA {lora_path}: {e}")
            return model, clip

    def _encode_prompt(self, clip, text):
        try:
            from nodes import CLIPTextEncode
            return CLIPTextEncode().encode(clip=clip, text=text)[0]
        except Exception as e:
            print(f"[AnimaFilteredRandomLoRALoader] Prompt encode error: {e}")
            return [[clip.encode(""), {}]]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def load_loras(self, model, clip, additional_prompt_positive, additional_prompt_negative,
                   lora_folder_path, include_subfolders, unique_by_filename,
                   keyword_filter, filter_mode, search_in_metadata,
                   model_strength, clip_strength, num_loras,
                   trigger_word_source, seed,
                   auto_remap, extend_to_new_layers, blend_ratio, extend_strength, manifest):

        # Model block count is the same for every LoRA this run; the manifest
        # itself is resolved PER LoRA inside _apply_lora_with_remap, since a
        # mixed-generation folder can hold LoRAs from different Anima
        # generations that each need a different manifest.
        model_block_count = get_model_block_count(model) if auto_remap else None
        if auto_remap and model_block_count is None:
            print("[AnimaFilteredRandomLoRALoader] Could not detect connected model's block count; remap disabled for this run")

        final_positive = additional_prompt_positive.strip()
        final_negative = additional_prompt_negative.strip()

        if num_loras == 0 or not lora_folder_path.strip():
            empty_preview = self._generate_preview_batch([])
            return self._finalize(model, clip, final_positive, final_negative, [], [], empty_preview)

        lora_files = self._find_lora_files(lora_folder_path, include_subfolders)
        if not lora_files:
            print(f"[AnimaFilteredRandomLoRALoader] No LoRA files found in {lora_folder_path}")
            empty_preview = self._generate_preview_batch([])
            return self._finalize(model, clip, final_positive, final_negative, [], [], empty_preview)

        filtered_files = self._filter_lora_files(lora_files, keyword_filter, filter_mode, search_in_metadata)
        if not filtered_files:
            print(f"[AnimaFilteredRandomLoRALoader] No LoRAs matched filter '{keyword_filter}'")
            empty_preview = self._generate_preview_batch([])
            return self._finalize(model, clip, final_positive, final_negative, [], [], empty_preview)

        if unique_by_filename:
            filtered_files = self._unique_by_filename(filtered_files)

        selected = self._select_random_loras(filtered_files, num_loras, seed)
        print(f"[AnimaFilteredRandomLoRALoader] Selected {len(selected)} LoRA(s)")

        all_positive_parts, all_negative_parts, preview_images = [], [], []

        for lora_path in selected:
            lora_name = os.path.splitext(os.path.basename(lora_path))[0]
            actual_model_str = self._parse_strength(model_strength)
            actual_clip_str = self._parse_strength(clip_strength)

            model, clip = self._apply_lora_with_remap(
                model, clip, lora_path, actual_model_str, actual_clip_str,
                auto_remap, manifest, model_block_count,
                extend_to_new_layers, blend_ratio, extend_strength
            )

            lora_notation = f"<lora:{lora_name}:{actual_model_str}:{actual_clip_str}>"

            if trigger_word_source == "json_combined":
                trigger_words = self._get_trigger_words_combined(lora_path)
            elif trigger_word_source == "json_random":
                trigger_words = self._get_trigger_words_random(lora_path)
            elif trigger_word_source == "json_sample_prompt":
                trigger_words, neg = self._get_sample_prompt_from_json(lora_path)
                if neg:
                    all_negative_parts.append(neg)
            else:  # metadata
                trigger_words = self._get_trigger_words_from_embedded(lora_path)

            all_positive_parts.append(f"{lora_notation}, {trigger_words}" if trigger_words else lora_notation)

            print(f"[AnimaFilteredRandomLoRALoader]   -> {lora_name} (MODEL:{actual_model_str}, CLIP:{actual_clip_str})")

            preview = self._load_preview_image_as_tensor(lora_path)
            if preview is not None:
                preview_images.append(preview)

        preview_batch = self._generate_preview_batch(preview_images)
        return self._finalize(model, clip, final_positive, final_negative,
                               all_positive_parts, all_negative_parts, preview_batch)

    def _finalize(self, model, clip, final_positive, final_negative,
                  lora_info_parts, all_negative_parts, preview_batch):
        # lora_text: additional prompt + "<lora:name:model:clip>, trigger words" per selected LoRA
        positive_parts = []
        if final_positive:
            positive_parts.append(final_positive)
        if lora_info_parts:
            positive_parts.append(", ".join(lora_info_parts))
        lora_text = ", ".join(positive_parts)

        # negative_text: additional prompt + any json_sample_prompt negatives
        negative_parts = []
        if final_negative:
            negative_parts.append(final_negative)
        sample_negative = ", ".join(n for n in all_negative_parts if n)
        if sample_negative:
            negative_parts.append(sample_negative)
        negative_text = ", ".join(negative_parts)

        # CONDITIONING: same text but with <lora:...> syntax stripped. cleaned_positive
        # is also returned as positive_text, so that pin always shows exactly what was
        # encoded into `positive`.
        cleaned_positive = self._remove_lora_syntax(lora_text)
        cleaned_negative = self._remove_lora_syntax(negative_text)
        positive_conditioning = self._encode_prompt(clip, cleaned_positive)
        negative_conditioning = self._encode_prompt(clip, cleaned_negative)

        return (model, clip, cleaned_positive, negative_text,
                positive_conditioning, negative_conditioning, preview_batch,
                lora_text)


NODE_CLASS_MAPPINGS = {
    "AnimaFilteredRandomLoRALoader": AnimaFilteredRandomLoRALoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaFilteredRandomLoRALoader": "Anima Filtered Random LoRA Loader (28/40/52 Auto)",
}
