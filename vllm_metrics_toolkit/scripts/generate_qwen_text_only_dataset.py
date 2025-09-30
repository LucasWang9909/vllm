#!/usr/bin/env python3
"""
Generate a text-only dataset that replaces visual tokens with an equal
number of single-token textual placeholders for Qwen2.5-Omni.

Input:
  - A JSON file created by vision_arena_processor.py (list of items with
    fields: image_path, prompt, metadata ...)

Output:
  - A new JSON file under benchmark_datasets/<output_dir>/... where each
    item has image_path set to "__TEXT_ONLY__" and prompt appended with
    N textual placeholders where N equals the number of visual tokens the
    model would produce for the original image.

Notes:
  - We try to obtain the exact visual token count using HF processor outputs
    (image_grid_thw) and the model's spatial_merge_size. If unavailable, we
    fall back to an approximate calculation based on resized H/W, patch_size
    and spatial_merge_size.
  - We build a placeholder string that encodes to exactly 1 token and repeat
    it N times to match the visual token count.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from transformers import AutoConfig, AutoTokenizer
    # Qwen2.5-Omni processor is provided via remote code
    from transformers import Qwen2_5OmniProcessor  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Please install transformers with remote code support for Qwen: pip install 'transformers>=4.41'"
    )


def load_processor_and_config(model_name: str):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_name, trust_remote_code=True)
    return processor, config


def get_visual_token_count(
    processor: Any,
    config: Any,
    image_path: str,
) -> int:
    """Return the number of visual tokens for a single image.

    Primary path: use processor to obtain image_grid_thw, then
    tokens = prod(thw) // (spatial_merge_size ** 2)
    Fallback: approximate using resized H/W, patch_size, spatial_merge_size.
    """
    img = Image.open(image_path).convert("RGB")

    # Try to get exact grid_thw from processor
    try:
        inputs = processor(
            text="", images=img, return_tensors="pt")  # type: ignore
        grid = inputs.get("image_grid_thw", None)
        if grid is not None:
            # grid shape: [1, 3] -> (T, H, W); for image T=1
            # tokens per image = (T*H*W) // (merge**2)
            import torch  # local import
            grid = grid if isinstance(grid, torch.Tensor) else torch.tensor(grid)
            spatial_merge = (
                config.thinker_config.vision_config.spatial_merge_size)
            prod = int(grid[0].prod().item())
            return max(prod // (spatial_merge ** 2), 0)
    except Exception:
        pass

    # Fallback approximate calculation
    vision_cfg = config.thinker_config.vision_config
    patch_size: int = getattr(vision_cfg, "patch_size", 14)
    spatial_merge: int = getattr(vision_cfg, "spatial_merge_size", 2)

    # Heuristic resize: keep aspect ratio so that the long edge equals
    # the processor's target longest edge if available; otherwise use
    # vision_cfg.image_size.
    size_cfg = getattr(getattr(processor, "image_processor", None), "size",
                       None)
    longest_edge = None
    if isinstance(size_cfg, dict):
        longest_edge = size_cfg.get("longest_edge") or size_cfg.get(
            "shortest_edge")
    if longest_edge is None:
        longest_edge = getattr(vision_cfg, "image_size", 1024)

    w, h = img.size
    if w >= h:
        scale = longest_edge / float(w)
    else:
        scale = longest_edge / float(h)
    rw = max(int(round(w * scale)), 1)
    rh = max(int(round(h * scale)), 1)

    grid_h = (rh + patch_size - 1) // patch_size
    grid_w = (rw + patch_size - 1) // patch_size
    prod = grid_h * grid_w  # T=1 for images
    return max(prod // (spatial_merge ** 2), 0)


def find_single_token_text(tokenizer) -> str:
    """Find a printable string that encodes to exactly 1 token."""
    # Try a few candidate token ids and pick the first that re-encodes to 1
    for token_id in range(100, min(len(tokenizer), 2000)):
        try:
            s = tokenizer.decode([token_id], skip_special_tokens=True)
            if not s or s.isspace():
                continue
            encoded = tokenizer.encode(s, add_special_tokens=False)
            if len(encoded) == 1:
                return s
        except Exception:
            continue
    # Fallback
    return "x"


def build_placeholder_string(tokenizer, count: int) -> str:
    unit = find_single_token_text(tokenizer)
    # Repeat with spaces to avoid BPE merges changing counts
    # Verify and adjust if drift occurs
    s = (unit + " ") * count
    enc = tokenizer.encode(s, add_special_tokens=False)
    if len(enc) == count:
        return s
    # If drift, try concatenation without spaces
    s = unit * count
    enc = tokenizer.encode(s, add_special_tokens=False)
    if len(enc) == count:
        return s
    # As last resort, stitch token ids and decode
    try:
        ids = [enc[0] if enc else 0] * count
        s = tokenizer.decode(ids, skip_special_tokens=True)
        # ensure count
        if len(tokenizer.encode(s, add_special_tokens=False)) == count:
            return s
    except Exception:
        pass
    return (unit + " ") * count


def main():
    parser = argparse.ArgumentParser(
        description="Generate text-only dataset replacing visual tokens with textual placeholders for Qwen2.5-Omni")
    parser.add_argument("input_json", help="Original processed dataset JSON (from vision_arena_processor.py)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B",
                        help="HF model name")
    parser.add_argument("--output_dir", default="benchmark_datasets/vision_arena_text_only",
                        help="Output directory under vllm_metrics_toolkit")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_dir = Path(__file__).resolve().parent.parent
    out_root = base_dir / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    processor, config = load_processor_and_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    output_items: List[Dict[str, Any]] = []

    for item in data:
        image_path = item.get("image_path", "")
        prompt = item.get("prompt", "")

        if not image_path or not Path(image_path).exists():
            # No image -> keep as text-only (no added placeholders)
            placeholders = ""
            new_prompt = prompt
            visual_tokens = 0
        else:
            n_visual = get_visual_token_count(processor, config, image_path)
            visual_tokens = int(n_visual)
            placeholders = build_placeholder_string(tokenizer, visual_tokens)
            sep = "\n" if prompt and not prompt.endswith("\n") else ""
            new_prompt = f"{prompt}{sep}{placeholders}" if placeholders else prompt

        new_item = {
            "image_path": "__TEXT_ONLY__",
            "prompt": new_prompt,
            "expected_response": item.get("expected_response", ""),
            "metadata": {
                **item.get("metadata", {}),
                "text_only": True,
                "visual_tokens": visual_tokens,
            },
        }
        output_items.append(new_item)

    # Save JSON next to input name
    out_file = out_root / (input_path.stem + "_text_only.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)

    print(f"✅ Text-only dataset saved: {out_file}")


if __name__ == "__main__":
    main()








