"""Run one traced generation or a CSV batch: python -m diffusion_attention.demo."""
import argparse
import csv
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import random
import warnings

import numpy as np
from .analysis import binary_mask, collect_heatmaps, pairwise_scores, validate_prompt

DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def select_device(requested, torch):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or a CUDA runtime.")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


def load_pipeline(model_id=DEFAULT_MODEL, device="auto", revision=None):
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        import daam  # noqa: F401 - fail early when the tracing dependency is absent
    except ImportError as error:
        raise RuntimeError("Install requirements-model.txt in a Python 3.10/3.11 environment first.") from error
    device = select_device(device, torch)
    if device == "cpu":
        warnings.warn("CPU inference uses float32 and can be very slow; CUDA is recommended.", stacklevel=2)
    options = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32,
               "use_safetensors": True}
    if revision:
        options["revision"] = revision
    # Uses cached Hugging Face authentication or HF_TOKEN; never logs credentials.
    # Keep the model's safety checker enabled.
    pipe = StableDiffusionPipeline.from_pretrained(model_id, **options).to(device)
    return pipe, device


def check_token_budget(tokenizer, prompt):
    ids = tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"]
    limit = tokenizer.model_max_length
    if len(ids) > limit:
        raise ValueError(f"Prompt uses {len(ids)} tokens, exceeding the model limit of {limit}; shorten it.")
    return len(ids)


def save_visuals(image, maps, directory, percentile):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    image.save(directory / "generated.png")
    fig, axes = plt.subplots(2, len(maps) + 1, figsize=(4 * (len(maps) + 1), 8), squeeze=False)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Generated image")
    axes[1, 0].text(0.05, 0.5, "Top: normalized overlays\nBottom: percentile masks\nScores use raw heatmaps", va="center", transform=axes[1, 0].transAxes)
    for column, (word, array) in enumerate(maps.items(), 1):
        normalized = (array - array.min()) / (array.max() - array.min()) if array.max() > array.min() else np.zeros_like(array)
        overlay = Image.fromarray(normalized.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR)
        axes[0, column].imshow(image)
        axes[0, column].imshow(overlay, cmap="magma", alpha=0.55, vmin=0, vmax=1)
        axes[0, column].set_title(word)
        axes[1, column].imshow(binary_mask(array, percentile), cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title(f"{word}: percentile {percentile:g}")
    for axis in axes.flat:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "attention.png", dpi=140)
    plt.close(fig)


def run_example(pipe, prompt, expression, output_dir, *, seed=42, steps=20,
                percentile=90, model_id=DEFAULT_MODEL, revision=None, device="cpu"):
    words = validate_prompt(prompt, expression)
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32 - 1.")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("Steps must be a positive integer.")
    binary_mask(np.zeros((1, 1)), percentile)  # Validate before expensive inference.
    token_count = check_token_budget(pipe.tokenizer, prompt)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing result.

    import torch
    from daam import trace
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode(), trace(pipe) as tracing:
        result = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5,
                      height=512, width=512, generator=generator)
        if result.nsfw_content_detected and any(result.nsfw_content_detected):
            raise RuntimeError("The model safety checker flagged this generation. Choose another prompt.")
        global_map = tracing.compute_global_heat_map()
        maps, missing = collect_heatmaps(global_map, words)
    if missing:
        warnings.warn("Some word heatmaps could not be extracted; see scores.json.", stacklevel=2)
    packages = {}
    for name in ("torch", "diffusers", "transformers", "daam", "numpy", "huggingface-hub", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    scores = pairwise_scores(maps)
    payload = {
        "status": "complete" if not missing else "partial",
        "prompt": prompt, "expression": expression, "words": words,
        "missing_words": missing, "pairwise_scores": scores,
        "max_iou": max((score["continuous_iou"] for score in scores), default=None),
        "seed": seed, "steps": steps, "guidance_scale": 7.5,
        "resolution": [512, 512], "percentile": percentile, "token_count": token_count,
        "model_id": model_id, "requested_revision": revision,
        "resolved_revision": getattr(pipe, "config", {}).get("_commit_hash"),
        "scheduler": type(pipe.scheduler).__name__, "scheduler_config": dict(pipe.scheduler.config),
        "device": device, "python": platform.python_version(), "packages": packages,
        "notes": "Raw-amplitude continuous IoU; repeated words aggregate across occurrences. Fixed seeds do not ensure identical output across hardware or package versions."
    }
    save_visuals(result.images[0], maps, directory, percentile)
    np.savez_compressed(directory / "heatmaps.npz", **maps)
    (directory / "scores.json").write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")
    return payload


def read_examples(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"prompt", "expression"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV needs prompt and expression columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one example.")
    for row in rows:
        validate_prompt(row["prompt"], row["expression"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--csv", type=Path)
    parser.add_argument("--expression")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Optional model commit SHA for repeatable model weights")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--percentile", type=float, default=90)
    args = parser.parse_args()
    if args.csv and args.expression:
        parser.error("Use the CSV expression column, not --expression, with --csv.")
    if args.prompt and not args.expression:
        parser.error("--expression is required with --prompt.")
    try:
        rows = read_examples(args.csv) if args.csv else [{"prompt": args.prompt, "expression": args.expression}]
        for row in rows:
            validate_prompt(row["prompt"], row["expression"])
        if not 0 <= args.seed < 2**32 or args.steps < 1:
            raise ValueError("Use seed 0..2**32-1 and a positive step count.")
        binary_mask(np.zeros((1, 1)), args.percentile)
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}. Choose a new directory.")
        pipe, device = load_pipeline(args.model, args.device, args.revision)
        for row in rows:
            check_token_budget(pipe.tokenizer, row["prompt"])
        for index, row in enumerate(rows):
            destination = args.output / f"{index:03d}" if args.csv else args.output
            run_example(pipe, row["prompt"], row["expression"], destination,
                        seed=args.seed, steps=args.steps, percentile=args.percentile,
                        model_id=args.model, revision=args.revision, device=device)
            print(f"Saved {destination}")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
