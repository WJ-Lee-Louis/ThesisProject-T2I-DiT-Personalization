"""Create focused 0903 generic-removal comparison grids.

Two grid families are rendered for every reference image:

* target projection + SSG;
* target + all-source projection + SSG.

Each grid has one projection-only control row followed by four generic-class
strategies at class strengths 0.5 and 1.0 (nine rows total).  The compositor
only reads downloaded experiment artifacts; it never regenerates or crops an
experiment image.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from create_comparison_grids import (
    COMMON_GENERATION_SETTING_KEYS,
    FontSet,
    _atomic_save_png,
    _atomic_write_json,
    _draw_centered_multiline,
    _draw_prompt_headers,
    _draw_reference_header,
    _read_json,
    _required_positive_int,
    _safe_child,
    load_fonts,
)
from orthogonalization_instantx_ip_adapter.batch_manifest import (
    DEFAULT_REFERENCE_ROOT,
    PROMPT_SUFFIXES,
    BatchManifest,
    build_batch_manifest,
)
from orthogonalization_instantx_ip_adapter.prompt_catalog import (
    PROMPT_CATALOG,
    get_prompt_spec,
)
from orthogonalization_instantx_ip_adapter.source_captions import (
    DEFAULT_SOURCE_CAPTIONS_PATH,
    SourceCaptionManifest,
    load_source_caption_manifest,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_ROOT = EXPERIMENT_ROOT / "outputs"
DEFAULT_DESTINATION_ROOT = DEFAULT_OUTPUTS_ROOT / "generic_comparison_grids_0903"


@dataclass(frozen=True, slots=True)
class MethodDefinition:
    variant_id: str
    batch_folder: str
    expected_conditions: tuple[str, ...]
    display_label: str
    class_strength: float | None


@dataclass(frozen=True, slots=True)
class FamilyDefinition:
    family_id: str
    title: str
    methods: tuple[MethodDefinition, ...]
    control_factorial_fallback_batch: str | None = None
    control_factorial_fallback_conditions: tuple[str, ...] = ()


def _generic_methods(prefix: str, batch_prefix: str) -> tuple[MethodDefinition, ...]:
    condition_prefix = "class_target" if prefix == "target" else "class_st_all"
    return (
        MethodDefinition(
            variant_id="absolute_direct_c05",
            batch_folder=f"exp0903-{batch_prefix}-abs-direct-c05",
            expected_conditions=(f"{condition_prefix}_absolute_direct_ssg",),
            display_label="Absolute Direct\nclass lambda = 0.5",
            class_strength=0.5,
        ),
        MethodDefinition(
            variant_id="absolute_direct_c10",
            batch_folder=f"exp0903-{batch_prefix}-abs-direct-c10",
            expected_conditions=(f"{condition_prefix}_absolute_direct_ssg",),
            display_label="Absolute Direct\nclass lambda = 1.0",
            class_strength=1.0,
        ),
        MethodDefinition(
            variant_id="absolute_residual_c05",
            batch_folder=f"exp0903-{batch_prefix}-abs-residual-c05",
            expected_conditions=(f"{condition_prefix}_absolute_residual_ssg",),
            display_label="Absolute Residual\nclass lambda = 0.5",
            class_strength=0.5,
        ),
        MethodDefinition(
            variant_id="absolute_residual_c10",
            batch_folder=f"exp0903-{batch_prefix}-abs-residual-c10",
            expected_conditions=(f"{condition_prefix}_absolute_residual_ssg",),
            display_label="Absolute Residual\nclass lambda = 1.0",
            class_strength=1.0,
        ),
        MethodDefinition(
            variant_id="no_object_c05",
            batch_folder=f"exp0903-{batch_prefix}-no-object-c05",
            expected_conditions=(f"{condition_prefix}_no_object_ssg",),
            display_label="No-Object Contextual\nclass lambda = 0.5",
            class_strength=0.5,
        ),
        MethodDefinition(
            variant_id="no_object_c10",
            batch_folder=f"exp0903-{batch_prefix}-no-object-c10",
            expected_conditions=(f"{condition_prefix}_no_object_ssg",),
            display_label="No-Object Contextual\nclass lambda = 1.0",
            class_strength=1.0,
        ),
        MethodDefinition(
            variant_id="superclass_c05",
            batch_folder=f"exp0903-{batch_prefix}-superclass-c05",
            expected_conditions=(f"{condition_prefix}_superclass_ssg",),
            display_label="Superclass Contextual\nclass lambda = 0.5",
            class_strength=0.5,
        ),
        MethodDefinition(
            variant_id="superclass_c10",
            batch_folder=f"exp0903-{batch_prefix}-superclass-c10",
            expected_conditions=(f"{condition_prefix}_superclass_ssg",),
            display_label="Superclass Contextual\nclass lambda = 1.0",
            class_strength=1.0,
        ),
    )


FAMILIES: tuple[FamilyDefinition, ...] = (
    FamilyDefinition(
        family_id="target",
        title="Target Projection + SSG: Generic-Removal Comparison",
        methods=(
            MethodDefinition(
                variant_id="control",
                batch_folder="dreambench-v2-target-gate-l1-0825",
                expected_conditions=("combined", "target_projection_ssg"),
                display_label=(
                    "Target Projection + SSG\n(No Generic Removal)"
                ),
                class_strength=None,
            ),
            *_generic_methods("target", "target"),
        ),
    ),
    FamilyDefinition(
        family_id="st_all",
        title="Target + All Source Projection + SSG: Generic-Removal Comparison",
        methods=(
            MethodDefinition(
                variant_id="control",
                batch_folder="dreambench-v2-all-source-union-gate-l1-0825",
                expected_conditions=("st_all_combined", "st_all_projection_ssg"),
                display_label=(
                    "Target + All Source Projection + SSG\n"
                    "(No Generic Removal)"
                ),
                class_strength=None,
            ),
            *_generic_methods("st_all", "sta"),
        ),
        # Historical ST-all omitted O+A+B files because ST-matched and ST-all
        # are definitionally identical when the target already contains both
        # factors.  Reuse is restricted to the control row.
        control_factorial_fallback_batch=(
            "dreambench-v2-matched-union-gate-l1-0825"
        ),
        control_factorial_fallback_conditions=(
            "st_combined",
            "st_matched_projection_ssg",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Layout:
    method_count: int = 9
    margin: int = 48
    label_width: int = 420
    label_gap: int = 24
    cell_width: int = 270
    cell_height: int = 360
    column_gap: int = 12
    row_gap: int = 14
    header_height: int = 340
    prompt_header_height: int = 96
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
            self.method_count * self.cell_height
            + (self.method_count - 1) * self.row_gap
        )

    @property
    def canvas_size(self) -> tuple[int, int]:
        return (
            self.grid_x + self.grid_width + self.margin,
            self.table_y + self.table_height + self.margin,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create twenty focused 0903 generic-removal comparison grids."
    )
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--source-captions", type=Path, default=DEFAULT_SOURCE_CAPTIONS_PATH
    )
    parser.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION_ROOT
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace grids and manifest already present at the destination.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs_root = args.outputs_root.expanduser().resolve()
    destination_root = args.destination.expanduser().resolve()
    image_manifest = build_batch_manifest(args.reference_root)
    source_captions = load_source_caption_manifest(
        args.source_captions, image_manifest
    )

    output_paths = {
        (family.family_id, reference.reference_stem): (
            destination_root
            / family.family_id
            / f"{reference.reference_stem}_{family.family_id}_generic_comparison.png"
        )
        for family in FAMILIES
        for reference in image_manifest.references
    }
    manifest_path = destination_root / "comparison_manifest.json"
    existing = [path for path in output_paths.values() if path.exists()]
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} output artifact(s) already exist; use --overwrite. "
            f"First: {existing[0]}"
        )

    resolved = _resolve_batches(outputs_root)
    batch_configs, common_settings = validate_inputs(
        image_manifest=image_manifest,
        source_captions=source_captions,
        resolved=resolved,
    )

    layout = Layout()
    fonts = load_fonts()
    generated = 0
    for family in FAMILIES:
        for reference in image_manifest.references:
            caption = source_captions.for_reference(reference.reference_stem)
            output_path = output_paths[(family.family_id, reference.reference_stem)]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            create_reference_grid(
                family=family,
                reference_path=Path(reference.reference_path),
                reference_filename=Path(reference.reference_path).name,
                reference_stem=reference.reference_stem,
                full_caption=caption.object_behavior_background,
                resolved=resolved,
                output_path=output_path,
                layout=layout,
                fonts=fonts,
            )
            generated += 1
            print(f"[{generated}/20] saved {output_path}", flush=True)

    artifact_manifest = {
        "schema_version": 1,
        "purpose": "focused_generic_removal_comparison",
        "image_manifest_digest": image_manifest.semantic_digest,
        "source_caption_digest": source_captions.semantic_digest,
        "common_generation_settings": common_settings,
        "grid_shape": {
            "family_count": len(FAMILIES),
            "reference_count": len(image_manifest.references),
            "method_count_per_grid": layout.method_count,
            "prompt_count": len(PROMPT_SUFFIXES),
            "grid_count": len(output_paths),
            "display_cell_count": (
                len(output_paths) * layout.method_count * len(PROMPT_SUFFIXES)
            ),
        },
        "canvas": {
            "width": layout.canvas_size[0],
            "height": layout.canvas_size[1],
            "cell_width": layout.cell_width,
            "cell_height": layout.cell_height,
        },
        "prompt_suffixes": list(PROMPT_SUFFIXES),
        "prompt_specs": [spec.to_dict() for spec in PROMPT_CATALOG],
        "families": [
            {
                "family_id": family.family_id,
                "title": family.title,
                "control_factorial_fallback_batch": (
                    family.control_factorial_fallback_batch
                ),
                "methods": [
                    {
                        "variant_id": method.variant_id,
                        "display_label": method.display_label,
                        "batch_folder": method.batch_folder,
                        "condition": batch_configs[method.batch_folder]["condition"],
                        "class_projection_strength": method.class_strength,
                    }
                    for method in family.methods
                ],
            }
            for family in FAMILIES
        ],
        "references": [
            {
                "reference_stem": reference.reference_stem,
                "reference_filename": Path(reference.reference_path).name,
                "caption": source_captions.for_reference(
                    reference.reference_stem
                ).object_behavior_background,
                "target_grid": output_paths[
                    ("target", reference.reference_stem)
                ].relative_to(destination_root).as_posix(),
                "st_all_grid": output_paths[
                    ("st_all", reference.reference_stem)
                ].relative_to(destination_root).as_posix(),
            }
            for reference in image_manifest.references
        ],
    }
    _atomic_write_json(manifest_path, artifact_manifest)
    print(
        "\nGeneric comparison grids complete\n"
        f"  grids: {len(output_paths)}\n"
        f"  shape: 9 methods x {len(PROMPT_SUFFIXES)} prompts\n"
        f"  canvas: {layout.canvas_size[0]}x{layout.canvas_size[1]}\n"
        f"  destination: {destination_root}",
        flush=True,
    )
    return 0


def _resolve_batches(outputs_root: Path) -> dict[str, Path]:
    names = {
        method.batch_folder
        for family in FAMILIES
        for method in family.methods
    }
    names.update(
        family.control_factorial_fallback_batch
        for family in FAMILIES
        if family.control_factorial_fallback_batch is not None
    )
    return {name: _safe_child(outputs_root, name) for name in sorted(names)}


def validate_inputs(
    *,
    image_manifest: BatchManifest,
    source_captions: SourceCaptionManifest,
    resolved: dict[str, Path],
) -> tuple[dict[str, dict], dict[str, object]]:
    """Validate all displayed cells and fair-comparison settings up front."""

    expected_conditions: dict[str, tuple[str, ...]] = {}
    expected_class_strengths: dict[str, float | None] = {}
    for family in FAMILIES:
        for method in family.methods:
            expected_conditions[method.batch_folder] = method.expected_conditions
            expected_class_strengths[method.batch_folder] = method.class_strength
        if family.control_factorial_fallback_batch is not None:
            expected_conditions[family.control_factorial_fallback_batch] = (
                family.control_factorial_fallback_conditions
            )
            expected_class_strengths[family.control_factorial_fallback_batch] = None

    configs: dict[str, dict] = {}
    common_settings: dict[str, object] | None = None
    expected_size: tuple[int, int] | None = None
    for batch_name, root in resolved.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Batch folder does not exist: {root}")
        config_path = root / "_metadata" / "config.json"
        config = _read_json(config_path)
        configs[batch_name] = config
        if config.get("condition") not in expected_conditions[batch_name]:
            raise ValueError(
                f"Unexpected condition in {config_path}: {config.get('condition')!r}; "
                f"expected one of {expected_conditions[batch_name]!r}"
            )
        if config.get("orthogonalization_strength") != 1.0:
            raise ValueError(f"Expected nuisance strength 1.0 in {config_path}")
        expected_class = expected_class_strengths[batch_name]
        if expected_class is not None and config.get(
            "class_projection_strength"
        ) != expected_class:
            raise ValueError(
                f"Class strength mismatch in {config_path}: "
                f"expected {expected_class}, got "
                f"{config.get('class_projection_strength')!r}"
            )
        if "sta-" in batch_name or "source-union" in batch_name or "matched-union" in batch_name:
            if config.get("source_caption_digest") != source_captions.semantic_digest:
                raise ValueError(f"Source-caption digest mismatch in {config_path}")

        current_settings = {
            key: config.get(key) for key in COMMON_GENERATION_SETTING_KEYS
        }
        if common_settings is None:
            common_settings = current_settings
        elif current_settings != common_settings:
            raise ValueError(
                f"Generation settings differ in {config_path}: "
                f"{current_settings} != {common_settings}"
            )
        current_size = (
            _required_positive_int(config, "width", config_path),
            _required_positive_int(config, "height", config_path),
        )
        if expected_size is None:
            expected_size = current_size
        elif current_size != expected_size:
            raise ValueError("Configured image dimensions differ across batches")

    assert common_settings is not None and expected_size is not None
    for family in FAMILIES:
        for method in family.methods:
            for job in image_manifest.jobs:
                image_path = _resolve_cell_path(
                    family=family,
                    method=method,
                    prompt_suffix=job.prompt_suffix,
                    reference_stem=job.reference_stem,
                    resolved=resolved,
                )
                if not image_path.is_file() or image_path.stat().st_size <= 0:
                    raise FileNotFoundError(f"Missing or empty image: {image_path}")
                with Image.open(image_path) as image:
                    if image.size != expected_size:
                        raise ValueError(
                            f"Unexpected dimensions for {image_path}: "
                            f"{image.size}, expected {expected_size}"
                        )
                    image.verify()
    return configs, common_settings


def _resolve_cell_path(
    *,
    family: FamilyDefinition,
    method: MethodDefinition,
    prompt_suffix: str,
    reference_stem: str,
    resolved: dict[str, Path],
) -> Path:
    batch_name = method.batch_folder
    if (
        method.variant_id == "control"
        and family.control_factorial_fallback_batch is not None
        and get_prompt_spec(prompt_suffix).is_factorial
    ):
        batch_name = family.control_factorial_fallback_batch
    filename = f"{reference_stem}_{'_'.join(prompt_suffix.split())}.png"
    return resolved[batch_name] / reference_stem / filename


def create_reference_grid(
    *,
    family: FamilyDefinition,
    reference_path: Path,
    reference_filename: str,
    reference_stem: str,
    full_caption: str,
    resolved: dict[str, Path],
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
    title_x = layout.grid_x + 2 * (layout.cell_width + layout.column_gap)
    _draw_centered_multiline(
        draw=draw,
        box=(
            title_x,
            layout.margin,
            layout.grid_x + layout.grid_width - title_x,
            layout.reference_size,
        ),
        text=family.title,
        font=fonts.title,
        fill=(15, 15, 15),
        spacing=8,
        max_width=layout.grid_x + layout.grid_width - title_x - 40,
    )
    _draw_prompt_headers(draw=draw, layout=layout, font=fonts.prompt)

    for row_index, method in enumerate(family.methods):
        row_y = layout.table_y + row_index * (
            layout.cell_height + layout.row_gap
        )
        _draw_centered_multiline(
            draw=draw,
            box=(layout.margin, row_y, layout.label_width, layout.cell_height),
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
            image_path = _resolve_cell_path(
                family=family,
                method=method,
                prompt_suffix=prompt_suffix,
                reference_stem=reference_stem,
                resolved=resolved,
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


if __name__ == "__main__":
    raise SystemExit(main())

