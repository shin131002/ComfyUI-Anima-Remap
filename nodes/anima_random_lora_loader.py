"""
anima_random_lora_loader.py

Anima-only port of shin131002/RandomLoRALoader's "Random LoRA Loader"
(3-folder simultaneous selection), with automatic old-Anima(28-block) /
Anima-2.9B(40-block) detection and block-index remapping reused from
anima_common.py / lora_remap_extended_anima.py.

Differences from the SD1.5/SDXL RandomLoRALoader this was ported from:
  - No LBW (LoRA Block Weight) support. Anima's flat transformer-block
    layout doesn't map onto the SD1.5/SDXL IN/MID/OUT preset convention,
    so block-weight support is deliberately out of scope for now.
  - No save_remapped / on-disk remap caching. This node is built around
    randomly picking a different LoRA per run, so a cached "_29Bremap"
    sibling file would (a) go stale silently the moment auto_remap's
    settings change and (b) get re-discovered by this node's own folder
    scan as a second, metadata-less "duplicate" LoRA. Every run remaps
    in-memory instead; the LoRA files themselves are tiny, so the extra
    CPU cost is negligible next to generation time.
  - The folder scan explicitly excludes files ending in "_29Bremap" or
    "_29Bremap_ext" (the suffixes used by the tag-based remap loaders in
    this package), so old cached copies from those nodes never get
    treated as extra, distinct LoRA candidates here.
  - remap settings (auto_remap / manifest / extend_to_new_layers /
    blend_ratio / extend_strength) are ONE shared set applied to all 3
    groups, not per-group -- per-group settings would multiply the
    number of combinations to reason about for little practical benefit.
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
    print("[AnimaRandomLoRALoader] safetensors not available, embedded metadata reading disabled")


class AnimaRandomLoRALoader:
    """Anima 28/40-block random LoRA selector/applier (3-group)."""

    _opencv_warning_shown = False

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_manifest_choices()
        group_inputs = {}
        for i in (1, 2, 3):
            group_inputs.update({
                f"lora_folder_path_{i}": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": f"Group {i} LoRA folder path"
                }),
                f"include_subfolders_{i}": ("BOOLEAN", {"default": True}),
                f"unique_by_filename_{i}": ("BOOLEAN", {"default": True}),
                f"model_strength_{i}": ("STRING", {
                    "default": "1.0", "multiline": False,
                    "placeholder": "e.g., 1.0 or 0.4-0.8"
                }),
                f"clip_strength_{i}": ("STRING", {
                    "default": "1.0", "multiline": False,
                    "placeholder": "e.g., 1.0 or 0.4-0.8"
                }),
                f"num_loras_{i}": ("INT", {
                    "default": 1 if i == 1 else 0, "min": 0, "max": 20, "step": 1
                }),
            })

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
                **group_inputs,
                "trigger_word_source": (
                    ["json_combined", "json_random", "json_sample_prompt", "metadata"],
                    {"default": "json_combined"}
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # -- Anima block remap (shared across all 3 groups) --
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
    FUNCTION = "load_random_loras"
    CATEGORY = "loaders/anima/random"

    # ------------------------------------------------------------------
    # Folder scan / selection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_remap_cache_file(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return is_remap_cache_filename(stem)

    def _find_lora_files(self, folder_path, include_subfolders):
        if not os.path.exists(folder_path):
            print(f"[AnimaRandomLoRALoader] Folder does not exist: {folder_path}")
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
            print(f"[AnimaRandomLoRALoader] Excluded {excluded} cached remap file(s) (_animaremap*) from candidates")
        # Sorted so a given seed picks the same LoRAs on any machine (filesystem
        # enumeration order differs between OSes/filesystems). Same as
        # RandomLoRALoader v1.3.0.
        return sorted(lora_files)

    def _unique_by_filename(self, lora_files, group_name=""):
        seen, unique_files = {}, []
        for file_path in lora_files:
            filename = os.path.basename(file_path)
            if filename not in seen:
                seen[filename] = file_path
                unique_files.append(file_path)
            else:
                print(f"[AnimaRandomLoRALoader] {group_name}: duplicate filename skipped: {filename}")
        return unique_files

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
                print(f"[AnimaRandomLoRALoader] Strength parse error ({strength_str}): {e}, using 1.0")
                return 1.0
        try:
            return float(strength_str)
        except Exception:
            print(f"[AnimaRandomLoRALoader] Could not parse strength '{strength_str}', using 1.0")
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
                    print(f"[AnimaRandomLoRALoader] JSON read error ({path}): {e}")
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
            print(f"[AnimaRandomLoRALoader] Embedded metadata read error ({lora_path}): {e}")
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
        positive, negative = meta.get("prompt", ""), meta.get("negativePrompt", "")
        positive = self._remove_lora_syntax(positive)
        negative = self._remove_lora_syntax(negative)
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
            print(f"[AnimaRandomLoRALoader] Folder read error: {e}")
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
            print(f"[AnimaRandomLoRALoader] Static image load error ({path}): {e}")
            return None

    def _load_animated_first_frame(self, path):
        try:
            from PIL import Image
            img = Image.open(path)
            if hasattr(img, 'seek'):
                img.seek(0)
            return self._resize_and_convert(img.convert('RGB'))
        except Exception as e:
            print(f"[AnimaRandomLoRALoader] Animated image load error ({path}): {e}")
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
            if not AnimaRandomLoRALoader._opencv_warning_shown:
                print("[AnimaRandomLoRALoader] Video preview requires 'pip install opencv-python'; "
                      "static/animated images still work without it.")
                AnimaRandomLoRALoader._opencv_warning_shown = True
            return None
        except Exception as e:
            print(f"[AnimaRandomLoRALoader] Video load error ({path}): {e}")
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
            print(f"[AnimaRandomLoRALoader] Image resize error: {e}")
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
            print(f"[AnimaRandomLoRALoader] Failed to load LoRA {lora_path}: {e}")
            return model, clip

        lora_block_count = get_lora_block_count(lora_sd.keys())

        # A LoRA that references more blocks than the connected model has
        # cannot be remapped DOWN (there's nowhere for its high-index
        # tensors to go). This is checked regardless of auto_remap -- it's
        # about whether the LoRA can be safely applied AT ALL, not about
        # remap settings -- so this is a hard stop rather than a silent
        # partial application. See AnimaBlockMismatchError.
        if (model_block_count is not None and lora_block_count is not None
                and lora_block_count > model_block_count):
            raise AnimaBlockMismatchError(
                f"[AnimaRandomLoRALoader] '{os.path.basename(lora_path)}' references "
                f"{lora_block_count} blocks but the connected model only has "
                f"{model_block_count}. It cannot be remapped down onto a smaller model, so "
                f"this run is stopped rather than silently applying a partial/broken LoRA. "
                f"Remove this file from the folder/filter, or connect the matching model."
            )

        # Resolve the manifest for THIS SPECIFIC LoRA's (block_count -> model's
        # block_count) pair -- not once per node run -- since a mixed-generation
        # folder can hold e.g. both 28-block and 40-block LoRAs that each need
        # a different manifest against the same connected model.
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
            print(f"[AnimaRandomLoRALoader] Failed to apply LoRA {lora_path}: {e}")
            return model, clip

    def _encode_prompt(self, clip, text):
        try:
            from nodes import CLIPTextEncode
            return CLIPTextEncode().encode(clip=clip, text=text)[0]
        except Exception as e:
            print(f"[AnimaRandomLoRALoader] Prompt encode error: {e}")
            return [[clip.encode(""), {}]]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def load_random_loras(self, model, clip, additional_prompt_positive, additional_prompt_negative,
                           lora_folder_path_1, include_subfolders_1, unique_by_filename_1,
                           model_strength_1, clip_strength_1, num_loras_1,
                           lora_folder_path_2, include_subfolders_2, unique_by_filename_2,
                           model_strength_2, clip_strength_2, num_loras_2,
                           lora_folder_path_3, include_subfolders_3, unique_by_filename_3,
                           model_strength_3, clip_strength_3, num_loras_3,
                           trigger_word_source, seed,
                           auto_remap, extend_to_new_layers, blend_ratio, extend_strength, manifest):

        # --- model block count is the same for every LoRA this run; the
        # manifest itself is resolved PER LoRA inside _apply_lora_with_remap,
        # since a mixed-generation folder can contain e.g. both 28-block and
        # 40-block LoRAs that each need a DIFFERENT manifest against the same
        # (say 52-block) connected model. ---
        model_block_count = get_model_block_count(model) if auto_remap else None
        if auto_remap and model_block_count is None:
            print("[AnimaRandomLoRALoader] Could not detect connected model's block count; remap disabled for this run")

        all_text_parts, all_positive_parts, all_negative_parts, preview_images = [], [], [], []

        groups = [
            (lora_folder_path_1, include_subfolders_1, unique_by_filename_1, model_strength_1, clip_strength_1, num_loras_1, "Group 1"),
            (lora_folder_path_2, include_subfolders_2, unique_by_filename_2, model_strength_2, clip_strength_2, num_loras_2, "Group 2"),
            (lora_folder_path_3, include_subfolders_3, unique_by_filename_3, model_strength_3, clip_strength_3, num_loras_3, "Group 3"),
        ]

        for folder_path, include_subs, unique_by_name, model_str, clip_str, num, group_name in groups:
            if not folder_path.strip() or num == 0:
                continue
            lora_files = self._find_lora_files(folder_path, include_subs)
            if not lora_files:
                print(f"[AnimaRandomLoRALoader] {group_name}: no LoRA files found")
                continue
            if unique_by_name:
                lora_files = self._unique_by_filename(lora_files, group_name)
                if not lora_files:
                    continue

            selected = self._select_random_loras(lora_files, num, seed)
            print(f"[AnimaRandomLoRALoader] {group_name}: selected {len(selected)} LoRA(s)")

            for lora_path in selected:
                lora_name = os.path.splitext(os.path.basename(lora_path))[0]
                actual_model_str = self._parse_strength(model_str)
                actual_clip_str = self._parse_strength(clip_str)

                model, clip = self._apply_lora_with_remap(
                    model, clip, lora_path, actual_model_str, actual_clip_str,
                    auto_remap, manifest, model_block_count,
                    extend_to_new_layers, blend_ratio, extend_strength
                )

                if trigger_word_source == "json_combined":
                    trigger_display = self._get_trigger_words_combined(lora_path)
                    all_positive_parts.append(trigger_display)
                elif trigger_word_source == "json_random":
                    trigger_display = self._get_trigger_words_random(lora_path)
                    all_positive_parts.append(trigger_display)
                elif trigger_word_source == "json_sample_prompt":
                    pos, neg = self._get_sample_prompt_from_json(lora_path)
                    all_positive_parts.append(pos)
                    all_negative_parts.append(neg)
                    trigger_display = pos
                else:  # metadata
                    trigger_display = self._get_trigger_words_from_embedded(lora_path)
                    all_positive_parts.append(trigger_display)

                lora_notation = f"<lora:{lora_name}:{actual_model_str}:{actual_clip_str}>"
                all_text_parts.append(f"{lora_notation}, {trigger_display},")

                preview = self._load_preview_image_as_tensor(lora_path)
                if preview is not None:
                    preview_images.append(preview)

        # -- text outputs --
        if additional_prompt_positive.strip():
            head = additional_prompt_positive.strip()
            if not head.endswith(','):
                head += ','
            lora_text_output = (head + "\n" + "\n".join(all_text_parts)) if all_text_parts else additional_prompt_positive.strip()
        else:
            lora_text_output = "\n".join(all_text_parts) if all_text_parts else ""

        negative_parts = []
        if additional_prompt_negative.strip():
            negative_parts.append(additional_prompt_negative.strip())
        sample_negative = ", ".join(n for n in all_negative_parts if n)
        if sample_negative:
            negative_parts.append(sample_negative)
        negative_text_output = ", ".join(negative_parts)

        # -- conditioning --
        final_positive_parts = []
        if additional_prompt_positive.strip():
            cleaned = self._remove_lora_syntax(additional_prompt_positive.strip())
            if cleaned:
                final_positive_parts.append(cleaned)
        final_positive_parts.extend(p for p in all_positive_parts if p)
        # positive_text output = exactly what is encoded into `positive`: the additional
        # prompt with <lora:...> removed, plus the trigger words. Safe to feed straight
        # into a text encoder, unlike lora_text whose <lora:...> tags would be tokenized.
        positive_text_output = ", ".join(final_positive_parts)
        positive_conditioning = self._encode_prompt(clip, positive_text_output)

        final_negative_parts = []
        if additional_prompt_negative.strip():
            cleaned = self._remove_lora_syntax(additional_prompt_negative.strip())
            if cleaned:
                final_negative_parts.append(cleaned)
        final_negative_parts.extend(n for n in all_negative_parts if n)
        negative_conditioning = self._encode_prompt(clip, ", ".join(final_negative_parts))

        preview_batch = self._generate_preview_batch(preview_images)

        return (model, clip, positive_text_output, negative_text_output,
                positive_conditioning, negative_conditioning, preview_batch,
                lora_text_output)


NODE_CLASS_MAPPINGS = {
    "AnimaRandomLoRALoader": AnimaRandomLoRALoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaRandomLoRALoader": "Anima Random LoRA Loader (28/40/52 Auto)",
}
