"""Create one eight-variant by nine-prompt comparison grid per reference.

The compositor never crops or regenerates experiment images. It validates the
eight batch variants and generation settings, downsamples each 3:4 PNG into a
fixed display cell, and writes ten deterministic high-resolution contact sheets
plus a machine-readable artifact manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

from orthogonalization_instantx_ip_adapter.batch_manifest import (
    DEFAULT_REFERENCE_ROOT,
    PROMPT_SUFFIXES,
    BatchManifest,
    build_batch_manifest,
)
from orthogonalization_instantx_ip_adapter.source_captions import (
    DEFAULT_SOURCE_CAPTIONS_PATH,
    SourceCaptionManifest,
    load_source_caption_manifest,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_ROOT = EXPERIMENT_ROOT / "outputs"
DEFAULT_DESTINATION_ROOT = DEFAULT_OUTPUTS_ROOT / "comparison_grids"

COMMON_GENERATION_SETTING_KEYS = (
    "ip_adapter_scale",
    "guidance_scale",
    "steps",
    "seed",
    "width",
    "height",
)


@dataclass(frozen=True, slots=True)
class MethodDefinition:
    variant_id: str
    condition: str
    orthogonalization_strength: float
    default_batch: str
    display_label: str


METHODS: tuple[MethodDefinition, ...] = (
    MethodDefinition(
        variant_id="baseline",
        condition="baseline",
        orthogonalization_strength=0.5,
        default_batch="dreambench-baseline-0723",
        display_label="Baseline\n(InstantX-IP-Adapter)",
    ),
    MethodDefinition(
        variant_id="projection",
        condition="projection",
        orthogonalization_strength=0.5,
        default_batch="dreambench-projection-0723",
        display_label="+ Target Projection\n(λ = 0.5)",
    ),
    MethodDefinition(
        variant_id="gate",
        condition="gate",
        orthogonalization_strength=0.5,
        default_batch="dreambench-gate-0723",
        display_label="+ Single-Stream Gate",
    ),
    MethodDefinition(
        variant_id="combined",
        condition="combined",
        orthogonalization_strength=0.5,
        default_batch="dreambench-combined-0723",
        display_label=(
            "+ Target Projection\n"
            "(λ = 0.5)\n"
            "+ Single-Stream Gate"
        ),
    ),
    MethodDefinition(
        variant_id="sa_projection",
        condition="sa_projection",
        orthogonalization_strength=0.5,
        default_batch="dreambench-sa-projection-0724",
        display_label="+ Source-Aware Projection\n(λ = 0.5)",
    ),
    MethodDefinition(
        variant_id="sa_combined",
        condition="sa_combined",
        orthogonalization_strength=0.5,
        default_batch="dreambench-sa-combined-0724",
        display_label=(
            "+ Source-Aware Projection\n"
            "(λ = 0.5)\n"
            "+ Single-Stream Gate"
        ),
    ),
    MethodDefinition(
        variant_id="combined_lambda1",
        condition="combined",
        orthogonalization_strength=1.0,
        default_batch="dreambench-combined-lambda1-0724",
        display_label=(
            "+ Target Projection\n"
            "(λ = 1.0)\n"
            "+ Single-Stream Gate"
        ),
    ),
    MethodDefinition(
        variant_id="sa_combined_lambda1",
        condition="sa_combined",
        orthogonalization_strength=1.0,
        default_batch="dreambench-sa-combined-lambda1-0724",
        display_label=(
            "+ Source-Aware Projection\n"
            "(λ = 1.0)\n"
            "+ Single-Stream Gate"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Layout:
    margin: int = 48
    label_width: int = 420
    label_gap: int = 24
    cell_width: int = 270
    cell_height: int = 360
    column_gap: int = 12
    row_gap: int = 14
    header_height: int = 340
    prompt_header_height: int = 80
    reference_size: int = 300

    @property
    def grid_x(self) -> int:
        return self.margin + self.label_width + self.label_gap

    @property
    def grid_width(self) -> int:
        return (
            len(PROMPT_SUFFIXES) * self.cell_width
            + (len(PROMPT_SUFFIXES) - 1) * self.column_gap
        )

    @property
    def table_y(self) -> int:
        return self.margin + self.header_height + self.prompt_header_height

    @property
    def table_height(self) -> int:
        return (
            len(METHODS) * self.cell_height
            + (len(METHODS) - 1) * self.row_gap
        )

    @property
    def canvas_size(self) -> tuple[int, int]:
        return (
            self.grid_x + self.grid_width + self.margin,
            self.table_y + self.table_height + self.margin,
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
            "Create ten reference-specific 8-variant x 9-prompt comparison "
            "PNGs from the downloaded InstantX experiment batches."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--source-captions",
        type=Path,
        default=DEFAULT_SOURCE_CAPTIONS_PATH,
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION_ROOT,
    )
    for method in METHODS:
        parser.add_argument(
            f"--{method.variant_id.replace('_', '-')}-batch",
            dest=f"{method.variant_id}_batch",
            default=method.default_batch,
            help=(
                f"Batch folder for {method.variant_id} "
                f"(default: {method.default_batch})"
            ),
        )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace comparison PNGs that already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs_root = args.outputs_root.expanduser().resolve()
    destination_root = args.destination.expanduser().resolve()

    image_manifest = build_batch_manifest(args.reference_root)
    source_captions = load_source_caption_manifest(
        args.source_captions,
        image_manifest,
    )
    batch_roots = {
        method.variant_id: _safe_child(
            outputs_root,
            getattr(args, f"{method.variant_id}_batch"),
        )
        for method in METHODS
    }

    batch_configs = validate_inputs(
        image_manifest=image_manifest,
        source_captions=source_captions,
        batch_roots=batch_roots,
    )
    output_paths = {
        reference.reference_stem: (
            destination_root
            / f"{reference.reference_stem}_six_method_comparison.png"
        )
        for reference in image_manifest.references
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} comparison PNG(s) already exist. "
            "Use --overwrite to replace them. "
            f"First existing file: {existing[0]}"
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    layout = Layout()
    fonts = load_fonts()

    for index, reference in enumerate(image_manifest.references, start=1):
        caption = source_captions.for_reference(reference.reference_stem)
        output_path = output_paths[reference.reference_stem]
        create_reference_grid(
            reference_path=Path(reference.reference_path),
            reference_filename=Path(reference.reference_path).name,
            reference_stem=reference.reference_stem,
            full_caption=caption.object_behavior_background,
            batch_roots=batch_roots,
            output_path=output_path,
            layout=layout,
            fonts=fonts,
        )
        print(
            f"[{index}/{len(image_manifest.references)}] "
            f"saved {output_path}",
            flush=True,
        )

    artifact_manifest = {
        "schema_version": 1,
        "image_manifest_digest": image_manifest.semantic_digest,
        "source_caption_digest": source_captions.semantic_digest,
        "canvas": {
            "width": layout.canvas_size[0],
            "height": layout.canvas_size[1],
            "cell_width": layout.cell_width,
            "cell_height": layout.cell_height,
        },
        "prompt_suffixes": list(PROMPT_SUFFIXES),
        "methods": [
            {
                "variant_id": method.variant_id,
                "condition": method.condition,
                "orthogonalization_strength": (
                    method.orthogonalization_strength
                ),
                "display_label": method.display_label,
                "batch_folder": batch_roots[method.variant_id].name,
                "config": batch_configs[method.variant_id],
            }
            for method in METHODS
        ],
        "references": [
            {
                "reference_stem": reference.reference_stem,
                "reference_filename": Path(reference.reference_path).name,
                "caption": source_captions.for_reference(
                    reference.reference_stem
                ).object_behavior_background,
                "output_filename": output_paths[
                    reference.reference_stem
                ].name,
            }
            for reference in image_manifest.references
        ],
    }
    _atomic_write_json(
        destination_root / "comparison_manifest.json",
        artifact_manifest,
    )

    print(
        "\nComparison grid generation complete\n"
        f"  references: {len(output_paths)}\n"
        f"  source images: "
        f"{len(METHODS) * len(PROMPT_SUFFIXES) * len(output_paths)}\n"
        f"  canvas: {layout.canvas_size[0]}x{layout.canvas_size[1]}\n"
        f"  destination: {destination_root}",
        flush=True,
    )
    return 0


def validate_inputs(
    *,
    image_manifest: BatchManifest,
    source_captions: SourceCaptionManifest,
    batch_roots: dict[str, Path],
) -> dict[str, dict]:
    """Fail before rendering if any batch cannot support a fair comparison."""

    expected_relative_paths = {
        job.output_relative_path for job in image_manifest.jobs
    }
    expected_image_size: tuple[int, int] | None = None
    reference_settings: dict[str, object] | None = None
    configs: dict[str, dict] = {}

    for method in METHODS:
        batch_root = batch_roots[method.variant_id]
        if not batch_root.is_dir():
            raise FileNotFoundError(
                f"Batch folder does not exist: {batch_root}"
            )

        config_path = batch_root / "_metadata" / "config.json"
        config = _read_json(config_path)
        configs[method.variant_id] = config
        if config.get("condition") != method.condition:
            raise ValueError(
                f"Condition mismatch in {config_path}: "
                f"expected {method.condition!r}, "
                f"got {config.get('condition')!r}"
            )
        if (
            config.get("orthogonalization_strength")
            != method.orthogonalization_strength
        ):
            raise ValueError(
                f"Orthogonalization strength mismatch in {config_path}: "
                f"expected {method.orthogonalization_strength}, "
                f"got {config.get('orthogonalization_strength')!r}"
            )
        if (
            config.get("expected_manifest_digest")
            != image_manifest.semantic_digest
        ):
            raise ValueError(
                f"Reference manifest mismatch in {config_path}"
            )
        if method.condition.startswith("sa_"):
            if (
                config.get("source_caption_digest")
                != source_captions.semantic_digest
            ):
                raise ValueError(
                    f"Source caption digest mismatch in {config_path}"
                )

        current_settings = {
            key: config.get(key)
            for key in COMMON_GENERATION_SETTING_KEYS
        }
        if reference_settings is None:
            reference_settings = current_settings
        elif current_settings != reference_settings:
            raise ValueError(
                "Generation settings differ across comparison batches: "
                f"{batch_root.name} has {current_settings}, "
                f"expected {reference_settings}"
            )

        current_size = (
            _required_positive_int(config, "width", config_path),
            _required_positive_int(config, "height", config_path),
        )
        if expected_image_size is None:
            expected_image_size = current_size
        elif current_size != expected_image_size:
            raise ValueError(
                "Configured image dimensions differ across batches"
            )

        actual_relative_paths = {
            path.relative_to(batch_root).as_posix()
            for path in batch_root.rglob("*.png")
        }
        missing = sorted(expected_relative_paths - actual_relative_paths)
        extra = sorted(actual_relative_paths - expected_relative_paths)
        if missing or extra:
            raise ValueError(
                f"Incomplete batch {batch_root.name}: "
                f"missing={missing}, extra={extra}"
            )

        assert expected_image_size is not None
        for relative_path in sorted(expected_relative_paths):
            image_path = batch_root / relative_path
            if image_path.stat().st_size <= 0:
                raise ValueError(f"Empty image file: {image_path}")
            with Image.open(image_path) as image:
                if image.size != expected_image_size:
                    raise ValueError(
                        f"Unexpected image dimensions for {image_path}: "
                        f"{image.size}, expected {expected_image_size}"
                    )
                image.verify()

    return configs


def create_reference_grid(
    *,
    reference_path: Path,
    reference_filename: str,
    reference_stem: str,
    full_caption: str,
    batch_roots: dict[str, Path],
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
    _draw_prompt_headers(draw=draw, layout=layout, font=fonts.prompt)

    for row_index, method in enumerate(METHODS):
        row_y = layout.table_y + row_index * (
            layout.cell_height + layout.row_gap
        )
        _draw_centered_multiline(
            draw=draw,
            box=(
                layout.margin,
                row_y,
                layout.label_width,
                layout.cell_height,
            ),
            text=method.display_label,
            font=fonts.method,
            fill=(20, 20, 20),
            spacing=7,
            max_width=layout.label_width - 36,
        )

        for column_index, prompt_suffix in enumerate(PROMPT_SUFFIXES):
            cell_x = layout.grid_x + column_index * (
                layout.cell_width + layout.column_gap
            )
            image_path = (
                batch_roots[method.variant_id]
                / reference_stem
                / (
                    f"{reference_stem}_"
                    f"{'_'.join(prompt_suffix.split())}.png"
                )
            )
            with Image.open(image_path) as source:
                rendered = ImageOps.contain(
                    source.convert("RGB"),
                    (layout.cell_width, layout.cell_height),
                    method=Image.Resampling.LANCZOS,
                )
            paste_x = cell_x + (layout.cell_width - rendered.width) // 2
            paste_y = row_y + (layout.cell_height - rendered.height) // 2
            canvas.paste(rendered, (paste_x, paste_y))
            draw.rectangle(
                (
                    cell_x,
                    row_y,
                    cell_x + layout.cell_width - 1,
                    row_y + layout.cell_height - 1,
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

    header_parts = (
        ("Reference Image", fonts.title, (15, 15, 15)),
        (reference_filename, fonts.filename, (35, 35, 35)),
        (f'"{full_caption}"', fonts.caption, (15, 15, 15)),
    )
    wrapped_parts = [
        (
            _wrap_text(
                draw,
                text,
                font,
                layout.label_width - 40,
            ),
            font,
            fill,
        )
        for text, font, fill in header_parts
    ]
    spacing_between_parts = 12
    part_heights = [
        _multiline_size(draw, text, font, spacing=5)[1]
        for text, font, _ in wrapped_parts
    ]
    total_height = (
        sum(part_heights)
        + spacing_between_parts * (len(wrapped_parts) - 1)
    )
    current_y = (
        layout.margin + (layout.reference_size - total_height) // 2
    )
    for (text, font, fill), part_height in zip(
        wrapped_parts,
        part_heights,
    ):
        _draw_centered_multiline(
            draw=draw,
            box=(
                layout.margin,
                current_y,
                layout.label_width,
                part_height,
            ),
            text=text,
            font=font,
            fill=fill,
            spacing=5,
            max_width=layout.label_width - 40,
            wrap=False,
        )
        current_y += part_height + spacing_between_parts


def _draw_prompt_headers(
    *,
    draw: ImageDraw.ImageDraw,
    layout: Layout,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    header_y = layout.margin + layout.header_height
    for column_index, prompt_suffix in enumerate(PROMPT_SUFFIXES):
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
            text=prompt_suffix,
            font=font,
            fill=(10, 10, 10),
            spacing=4,
            max_width=layout.cell_width - 10,
        )


def load_fonts() -> FontSet:
    regular_candidates = (
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    )
    bold_candidates = (
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
    )
    regular_path = _first_existing(regular_candidates)
    bold_path = _first_existing(bold_candidates)
    return FontSet(
        title=_font(bold_path, 31),
        filename=_font(bold_path, 24),
        caption=_font(regular_path, 25),
        prompt=_font(bold_path, 25),
        method=_font(bold_path, 27),
    )


def _font(
    path: Path | None,
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(str(path), size=size)


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
    rendered_text = (
        _wrap_text(draw, text, font, max_width)
        if wrap
        else text
    )
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0),
        rendered_text,
        font=font,
        spacing=spacing,
        align="center",
    )
    text_width = right - left
    text_height = bottom - top
    origin_x = x + (width - text_width) / 2 - left
    origin_y = y + (height - text_height) / 2 - top
    draw.multiline_text(
        (origin_x, origin_y),
        rendered_text,
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
    wrapped_paragraphs: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            wrapped_paragraphs.append("")
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
        wrapped_paragraphs.append("\n".join(lines))
    return "\n".join(wrapped_paragraphs)


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
        (0, 0),
        text,
        font=font,
        spacing=spacing,
        align="center",
    )
    return right - left, bottom - top


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _required_positive_int(config: dict, key: str, path: Path) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer in {path}")
    return value


def _safe_child(root: Path, relative_value: str) -> Path:
    relative_path = Path(relative_value)
    if relative_path.is_absolute():
        raise ValueError(
            f"Expected a relative batch folder, got {relative_value!r}"
        )
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(
            f"Batch folder escapes outputs root: {relative_value!r}"
        ) from error
    return candidate


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
