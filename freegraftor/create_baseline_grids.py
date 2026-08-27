"""Create FreeGraftor baseline contact sheets for the 10-reference experiment.

The renderer follows the visual conventions of the existing InstantX
comparison grids: one reference header, one method row, and one column for each
of the frozen 16 prompts.  Experiment images are only downsampled with their
aspect ratios preserved; they are never cropped or regenerated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from freegraftor_experiment.batch_manifest import (  # noqa: E402
    BatchManifest,
    build_batch_manifest,
)
from freegraftor_experiment.prompt_catalog import (  # noqa: E402
    PROMPT_CATALOG,
)


DEFAULT_BATCH_ROOT = (
    EXPERIMENT_ROOT / "outputs" / "freegraftor-baseline-0826-v2"
)
DEFAULT_REFERENCE_ROOT = (
    EXPERIMENT_ROOT.parent
    / "dreambooth_reference_dataset"
    / "InstantX-IP-Adapter-DreamBench_filtered"
)
DEFAULT_SOURCE_CAPTIONS = (
    EXPERIMENT_ROOT.parent
    / "Orthogonalization-InstantX-IP-Adapter"
    / "source_captions.json"
)
DEFAULT_DESTINATION = (
    EXPERIMENT_ROOT
    / "outputs"
    / "comparison_grids"
    / DEFAULT_BATCH_ROOT.name
)


@dataclass(frozen=True, slots=True)
class Layout:
    """The existing comparison-grid geometry reduced to one method row."""

    margin: int = 48
    label_width: int = 420
    label_gap: int = 24
    cell_width: int = 270
    cell_height: int = 360
    column_gap: int = 12
    header_height: int = 340
    prompt_header_height: int = 96
    reference_size: int = 300

    @property
    def grid_x(self) -> int:
        return self.margin + self.label_width + self.label_gap

    @property
    def grid_width(self) -> int:
        return (
            len(PROMPT_CATALOG) * self.cell_width
            + (len(PROMPT_CATALOG) - 1) * self.column_gap
        )

    @property
    def table_y(self) -> int:
        return self.margin + self.header_height + self.prompt_header_height

    @property
    def canvas_size(self) -> tuple[int, int]:
        return (
            self.grid_x + self.grid_width + self.margin,
            self.table_y + self.cell_height + self.margin,
        )


@dataclass(frozen=True, slots=True)
class FontSet:
    title: ImageFont.FreeTypeFont | ImageFont.ImageFont
    filename: ImageFont.FreeTypeFont | ImageFont.ImageFont
    caption: ImageFont.FreeTypeFont | ImageFont.ImageFont
    prompt: ImageFont.FreeTypeFont | ImageFont.ImageFont
    method: ImageFont.FreeTypeFont | ImageFont.ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create ten FreeGraftor baseline grids with the 16 prompts as "
            "columns."
        )
    )
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--source-captions",
        type=Path,
        default=DEFAULT_SOURCE_CAPTIONS,
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace grids and their artifact manifest if they already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_root = args.batch_root.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    manifest = build_batch_manifest(args.reference_root)
    captions = load_source_captions(args.source_captions, manifest)
    batch_metadata = validate_batch(batch_root, manifest)

    output_paths = {
        reference.stem: (
            destination / f"{reference.stem}_freegraftor_baseline_grid.png"
        )
        for reference in manifest.references
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} grid PNG(s) already exist. Use --overwrite to "
            f"replace them. First existing file: {existing[0]}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    layout = Layout()
    fonts = load_fonts()
    for index, reference in enumerate(manifest.references, start=1):
        output_path = output_paths[reference.stem]
        create_reference_grid(
            reference_path=Path(reference.reference_path),
            reference_filename=reference.filename,
            reference_stem=reference.stem,
            full_caption=captions[reference.stem],
            batch_root=batch_root,
            output_path=output_path,
            layout=layout,
            fonts=fonts,
        )
        print(
            f"[{index}/{len(manifest.references)}] saved {output_path}",
            flush=True,
        )

    artifact_manifest = {
        "schema_version": 1,
        "source_batch": batch_root.name,
        "source_batch_root": str(batch_root),
        "image_manifest_digest": manifest.semantic_digest,
        "grid_shape": {
            "reference_count": len(manifest.references),
            "method_count": 1,
            "prompt_count": len(PROMPT_CATALOG),
            "display_cell_count": len(manifest.jobs),
        },
        "canvas": {
            "width": layout.canvas_size[0],
            "height": layout.canvas_size[1],
            "cell_width": layout.cell_width,
            "cell_height": layout.cell_height,
        },
        "method": {
            "name": "FreeGraftor",
            "variant": "baseline",
            "display_label": "FreeGraftor\n(Baseline)",
        },
        "prompts": [
            {
                "suffix": prompt.suffix,
                "slug": prompt.slug,
                "category": prompt.category,
                "composition": prompt.composition,
            }
            for prompt in PROMPT_CATALOG
        ],
        "references": [
            {
                "stem": reference.stem,
                "filename": reference.filename,
                "caption": captions[reference.stem],
                "output_filename": output_paths[reference.stem].name,
            }
            for reference in manifest.references
        ],
        "source_config": batch_metadata,
    }
    _atomic_write_json(destination / "grid_manifest.json", artifact_manifest)
    print(
        "\nFreeGraftor grid generation complete\n"
        f"  grids: {len(output_paths)}\n"
        f"  cells: {len(manifest.jobs)}\n"
        f"  canvas: {layout.canvas_size[0]}x{layout.canvas_size[1]}\n"
        f"  destination: {destination}",
        flush=True,
    )
    return 0


def load_source_captions(
    path: str | Path,
    manifest: BatchManifest,
) -> dict[str, str]:
    caption_path = Path(path)
    if not caption_path.is_file():
        raise FileNotFoundError(f"Source-caption file is missing: {caption_path}")
    payload = json.loads(caption_path.read_text(encoding="utf-8"))
    records = payload.get("references")
    if not isinstance(records, dict):
        raise ValueError(f"Invalid source-caption file: {caption_path}")

    expected_stems = {reference.stem for reference in manifest.references}
    actual_stems = set(records)
    if actual_stems != expected_stems:
        raise ValueError(
            "Source-caption references differ from the image manifest: "
            f"missing={sorted(expected_stems - actual_stems)}, "
            f"extra={sorted(actual_stems - expected_stems)}"
        )

    captions: dict[str, str] = {}
    for reference in manifest.references:
        record = records[reference.stem]
        if not isinstance(record, dict):
            raise ValueError(f"Invalid caption for {reference.stem}")
        behavior = record.get("source_behavior")
        background = record.get("source_background")
        if not isinstance(behavior, str) or not behavior.strip():
            raise ValueError(f"Missing source_behavior for {reference.stem}")
        if not isinstance(background, str) or not background.strip():
            raise ValueError(f"Missing source_background for {reference.stem}")
        captions[reference.stem] = (
            f"a {reference.species} {behavior.strip()} {background.strip()}"
        )
    return captions


def validate_batch(batch_root: Path, manifest: BatchManifest) -> dict:
    """Fail before rendering if any of the 160 baseline PNGs is invalid."""

    if not batch_root.is_dir():
        raise FileNotFoundError(f"Batch folder does not exist: {batch_root}")
    config_path = batch_root / "_metadata" / "config.json"
    metadata = _read_json(config_path)
    config = metadata.get("config")
    algorithm = metadata.get("algorithm")
    if not isinstance(config, dict) or not isinstance(algorithm, dict):
        raise ValueError(f"Malformed FreeGraftor metadata: {config_path}")
    if algorithm.get("name") != "FreeGraftor":
        raise ValueError(f"Unexpected algorithm in {config_path}")
    if config.get("batch_id") != batch_root.name:
        raise ValueError(f"Batch ID does not match folder name: {config_path}")
    if config.get("expected_manifest_digest") != manifest.semantic_digest:
        raise ValueError(f"Reference/prompt manifest mismatch: {config_path}")

    width = _positive_int(config, "width", config_path)
    height = _positive_int(config, "height", config_path)
    expected_paths = {
        Path(job.output_relative_path).as_posix() for job in manifest.jobs
    }
    actual_paths = {
        path.relative_to(batch_root).as_posix()
        for reference in manifest.references
        for path in (batch_root / reference.stem).glob("*.png")
    }
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing or extra:
        raise ValueError(
            f"Incomplete FreeGraftor batch: missing={missing}, extra={extra}"
        )

    for relative_path in sorted(expected_paths):
        image_path = batch_root / relative_path
        if image_path.stat().st_size <= 0:
            raise ValueError(f"Empty image file: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"Unexpected dimensions for {image_path}: {image.size}, "
                    f"expected {(width, height)}"
                )
            image.verify()
    return metadata


def create_reference_grid(
    *,
    reference_path: Path,
    reference_filename: str,
    reference_stem: str,
    full_caption: str,
    batch_root: Path,
    output_path: Path,
    layout: Layout,
    fonts: FontSet,
) -> None:
    canvas = Image.new("RGB", layout.canvas_size, "white")
    draw = ImageDraw.Draw(canvas)
    _draw_reference_header(
        canvas=canvas,
        draw=draw,
        reference_path=reference_path,
        reference_filename=reference_filename,
        full_caption=full_caption,
        layout=layout,
        fonts=fonts,
    )
    _draw_prompt_headers(draw, layout, fonts.prompt)
    _draw_centered_multiline(
        draw=draw,
        box=(layout.margin, layout.table_y, layout.label_width, layout.cell_height),
        text="FreeGraftor\n(Baseline)",
        font=fonts.method,
        fill=(20, 20, 20),
        spacing=7,
        max_width=layout.label_width - 36,
    )

    for column_index, prompt in enumerate(PROMPT_CATALOG):
        cell_x = layout.grid_x + column_index * (
            layout.cell_width + layout.column_gap
        )
        image_path = (
            batch_root / reference_stem / f"{reference_stem}_{prompt.slug}.png"
        )
        with Image.open(image_path) as source:
            rendered = ImageOps.contain(
                source.convert("RGB"),
                (layout.cell_width, layout.cell_height),
                method=Image.Resampling.LANCZOS,
            )
        paste_x = cell_x + (layout.cell_width - rendered.width) // 2
        paste_y = layout.table_y + (layout.cell_height - rendered.height) // 2
        canvas.paste(rendered, (paste_x, paste_y))
        draw.rectangle(
            (
                cell_x,
                layout.table_y,
                cell_x + layout.cell_width - 1,
                layout.table_y + layout.cell_height - 1,
            ),
            outline=(210, 210, 210),
            width=1,
        )
    _atomic_save_png(output_path, canvas)


def _draw_reference_header(
    *,
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    reference_path: Path,
    reference_filename: str,
    full_caption: str,
    layout: Layout,
    fonts: FontSet,
) -> None:
    with Image.open(reference_path) as source:
        reference = ImageOps.contain(
            source.convert("RGB"),
            (layout.reference_size, layout.reference_size),
            method=Image.Resampling.LANCZOS,
        )
    first_two_columns_width = 2 * layout.cell_width + layout.column_gap
    reference_x = (
        layout.grid_x
        + (first_two_columns_width - layout.reference_size) // 2
    )
    reference_y = layout.margin
    canvas.paste(reference, (reference_x, reference_y))
    draw.rectangle(
        (
            reference_x,
            reference_y,
            reference_x + reference.width - 1,
            reference_y + reference.height - 1,
        ),
        outline=(180, 180, 180),
        width=2,
    )

    parts = (
        ("Reference Image", fonts.title, (15, 15, 15)),
        (reference_filename, fonts.filename, (35, 35, 35)),
        (f'"{full_caption}"', fonts.caption, (15, 15, 15)),
    )
    wrapped = [
        (_wrap_text(draw, text, font, layout.label_width - 40), font, fill)
        for text, font, fill in parts
    ]
    heights = [
        _multiline_size(draw, text, font, spacing=5)[1]
        for text, font, _ in wrapped
    ]
    gap = 12
    current_y = (
        layout.margin
        + (layout.reference_size - (sum(heights) + gap * (len(heights) - 1)))
        // 2
    )
    for (text, font, fill), height in zip(wrapped, heights):
        _draw_centered_multiline(
            draw=draw,
            box=(layout.margin, current_y, layout.label_width, height),
            text=text,
            font=font,
            fill=fill,
            spacing=5,
            max_width=layout.label_width - 40,
            wrap=False,
        )
        current_y += height + gap


def _draw_prompt_headers(
    draw: ImageDraw.ImageDraw,
    layout: Layout,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    header_y = layout.margin + layout.header_height
    for column_index, prompt in enumerate(PROMPT_CATALOG):
        cell_x = layout.grid_x + column_index * (
            layout.cell_width + layout.column_gap
        )
        _draw_centered_multiline(
            draw=draw,
            box=(
                cell_x,
                header_y,
                layout.cell_width,
                layout.prompt_header_height,
            ),
            text=prompt.suffix,
            font=font,
            fill=(10, 10, 10),
            spacing=4,
            max_width=layout.cell_width - 10,
        )


def load_fonts() -> FontSet:
    regular = _first_existing(
        (
            Path(r"C:\Windows\Fonts\times.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
        )
    )
    bold = _first_existing(
        (
            Path(r"C:\Windows\Fonts\timesbd.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        )
    )
    return FontSet(
        title=_font(bold, 31),
        filename=_font(bold, 24),
        caption=_font(regular, 25),
        prompt=_font(bold, 25),
        method=_font(bold, 27),
    )


def _font(
    path: Path | None,
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.truetype(str(path), size=size) if path else ImageFont.load_default()


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def _draw_centered_multiline(
    *,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int,
    max_width: int,
    wrap: bool = True,
) -> None:
    x, y, width, height = box
    rendered = _wrap_text(draw, text, font, max_width) if wrap else text
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), rendered, font=font, spacing=spacing, align="center"
    )
    origin_x = x + (width - (right - left)) / 2 - left
    origin_y = y + (height - (bottom - top)) / 2 - top
    draw.multiline_text(
        (origin_x, origin_y),
        rendered,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    paragraphs: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            paragraphs.append("")
            continue
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if _text_width(draw, candidate, font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        paragraphs.append("\n".join(lines))
    return "\n".join(paragraphs)


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _multiline_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    spacing: int,
) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing, align="center"
    )
    return right - left, bottom - top


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _positive_int(config: dict, key: str, path: Path) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer in {path}")
    return value


def _atomic_save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        image.save(temporary, format="PNG", compress_level=6)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
