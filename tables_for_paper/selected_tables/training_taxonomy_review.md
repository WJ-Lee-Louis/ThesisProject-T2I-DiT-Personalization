# Training dependency review for Tables 01 and 02

Reviewed: 2026-11-16. Scope: the actual versions used in the repository's selected
tables, their pinned inference code and model locks, and the original papers.
This is a terminology and dependency audit, not a rerun of training or generation.
No table labels or PNGs were changed during this review.

## Decision and operational definitions

The intended distinction is defensible as **personalization-specific training
dependency**, rather than per-subject tuning. An operational definition is needed
because training-free/tuning-free/zero-shot are used at different scopes across
papers; they are not a single universal taxonomy.

1. **Subject-specific tuning**: optimizing model parameters, LoRA weights or text
   embeddings for each new reference subject before generating its images.
   The evaluated pipelines do not perform this optimization.
2. **Personalization-specific training dependency**: the deployed generation
   pipeline requires weights learned to acquire reference-conditioned
   personalization capabilities. This includes a pretrained personalization
   adapter or personalization LoRA, even when the current authors did not train it.
   It excludes generic foundation-model, text/vision encoder, segmentation/LLM,
   acceleration and generic inpainting pretraining by themselves.
3. **New training introduced by VIP-Adapter**: whether our projection, semantic
   analysis or gate requires new learned weights or finetuning. The reviewed
   implementation uses frozen pretrained models, LLM inference, SVD/projection and
   a deterministic row mask; it introduces no such training.

The binary table terms **Training-based / Training-free** are usable only when
explicitly defined using criterion 2. More transparent terms are
**Personalization-trained / Training-free** or a column titled
**Personalization training** with Required / Not required. These labels describe
the pipeline's dependencies, not whether the benchmark runner called an optimizer.

Suggested definition:

“We categorize methods according to their reliance on weights learned through
personalization-specific training. Training-based methods rely on such weights,
whereas training-free methods reuse pretrained models without
personalization-specific training. Generic foundation-model and inpainting
pretraining are excluded from this distinction.”

## Method-by-method findings

| Evaluated method | Personalization-trained dependency | New-subject tuning | Code and paper evidence |
|---|---|---|---|
| IP-Adapter (InstantX FLUX) | Yes | No | Actual FLUX adapter has a trained SigLIP-to-image-token MLP and IP-attention layers. The official model card documents 10M training images and 80K steps. The loader instantiates the published IPAdapter with `ip-adapter.bin`; original SD IP-Adapter is conceptual context, not proof of the FLUX implementation. |
| UNO, 512 comparison | Yes | No | UNO paper describes progressive training of an in-context customization model and LoRA. `uno/runtime.py` loads official `dit_lora.safetensors`, constructs UNOPipeline, freezes the modules and runs inference. The 512 configuration uses rank 512. No subject-specific LoRA training occurs. |
| DreamO v1.1 + Turbo | Yes | No | Paper describes customization training, learnable embeddings and LoRA. The actual v1.1 loader reads DreamO embeddings and fuses pretrained DreamO, CFG, SFT/DPO and optional Turbo LoRAs. Fusing/unloading LoRA does not erase the learned dependency. Turbo alone is not the reason for classification. |
| XVerse | Yes | No | Paper trains both modulation adapters and LoRA. The actual wrapper calls `load_modulation_adapter(..., is_training=False)` and `load_dit_lora(..., is_training=False)` and conditions on each supplied image. `is_training=False` indicates inference, not absence of earlier training. |
| USO | Yes | No | Paper describes style alignment, content/style disentanglement and reward learning. Runtime loads `uso_flux_v1.0/dit_lora.safetensors` and `projector.safetensors`, then freezes modules. Both content and style routes use the pretrained customization model. |
| MOSAIC, subject_512 | Yes | No | Paper describes personalization LoRA trained with correspondence/disentanglement losses. Runtime loads `subject_512.safetensors` as the `subject` adapter and performs the official generation call. |
| Diptych Prompting | No, under the task-specific definition above | No | Paper explicitly avoids specialized subject-generation training. Actual code loads FLUX plus `FLUX.1-dev-Controlnet-Inpainting-Beta` and changes diptych attention. Thus it DOES depend on an additional trained ControlNet, but for generic inpainting rather than learned subject personalization. It must not be described as requiring no trained auxiliary modules whatsoever. |
| FreeCus, paper/512 runtime | No, under the task-specific definition above | No | Paper positions the method as training-free. Runtime uses vanilla FLUX, frozen Qwen caption models and BiRefNet segmentation, reference inversion/trajectory capture and pivotal attention sharing. The reviewed generation path does not load a subject-personalization LoRA/adapter. Learned generic VLM/segmentation dependencies still exist. |
| FreeGraftor, freegrafor2 archived run | No, under the task-specific definition above | No | Paper uses pretrained FLUX feature grafting and structure-consistent initialization. Archived actual source loads FLUX/text encoders/VAE and generic grounding/segmentation models. Reference-dependent trajectories/features are inference caches, not newly optimized model weights. The archived FLUX format conversion is tensor mapping, not training. |
| VIP-Adapter, both scales | **Yes, inherited from IP-Adapter** | No | `OrthogonalizedIPAdapter` composes our mixin with the published InstantX IPAdapter and loads the same trained adapter weight. The proposed module adds LLM-generated contrasts, analytical projection and deterministic single-stream gating without training new weights. Thus it is training-free as a refinement, while the end-to-end personalization pipeline remains dependent on a trained adapter. |

Changing the scalar IP injection strength (0.7/0.8) does not change the training
dependency classification. Scale selection is hyperparameter selection and should
not be confused with per-subject parameter learning.

## Primary paper/model references

- [InstantX FLUX IP-Adapter model card](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter#model-card)
  — actual evaluated adapter architecture and training.
- [Original IP-Adapter, §3.2.3](https://arxiv.org/html/2308.06721v1)
  — adapter training with frozen diffusion backbone; distinct from the FLUX port.
- [UNO, §3.3, §4.1 and Appendix G](https://arxiv.org/html/2504.02160v1)
  — progressive customization training and LoRA.
- [DreamO, §3.2, §3.5 and §4.1](https://arxiv.org/html/2504.16915v1)
  — learnable embeddings, customization LoRA and staged training.
- [DreamO v1.1 release](https://github.com/bytedance/DreamO/blob/461e35ba6b068f42ca53b43f289b3d91b7f911cd/dreamo_v1.1.md)
  — evaluated release differs from the original paper version.
- [XVerse, §4.1](https://arxiv.org/html/2506.21416v1)
  — jointly trained modulation adapters and LoRA.
- [USO, §3 and implementation appendix](https://arxiv.org/html/2508.18966v1)
  — personalization training stages; also uses the expression training-based in
  its related-work discussion of learned style models.
- [MOSAIC, §4.1 and §5.1.1](https://arxiv.org/html/2509.01977v1)
  — LoRA reference branch and correspondence/disentanglement training.
- [Diptych Prompting, §3.1 and §4.1](https://arxiv.org/html/2411.15466v2)
  — generic trained ControlNet-Inpainting dependency and no specialized
  subject-generation training.
- [FreeCus, §1–2 and §3](https://arxiv.org/html/2507.15249v1)
  — distinguishes per-subject training and trained image-conditioning models
  from the proposed training-free pipeline.
- [FreeGraftor, §2.2, §3 and §4.1.2](https://arxiv.org/html/2504.15958v1)
  — distinguishes subject-specific tuning, pretrained zero-shot personalization
  modules and training-free feature-grafting methods.

## Code evidence in this repository

The upstream snapshots were verified against their stored SHA-256 manifests:
UNO 15 files, DreamO 11, XVerse 18, USO 14, MOSAIC 7, Diptych 6 and FreeCus 8.
All 79 files matched. FreeGraftor uses its separate archived run snapshot and
PROVENANCE.json; its code was reviewed and fingerprinted separately, not included
in the 79-file claim. The companion JSON records file hashes and evidence lines.

- [UNO inference wrapper](../../uno/runtime.py) and
  [original checkpoint loader](../../uno/upstream/uno/flux/util.py).
- [DreamO wrapper](../../dreamo/runtime.py) and
  [original LoRA/embedding loader](../../dreamo/upstream/dreamo/dreamo_pipeline.py).
- [XVerse and Diptych wrappers](../../comparison_candidates/runtime.py).
- [USO wrapper](../../uso/runtime.py) and
  [original projector/LoRA loader](../../uso/upstream/uso/flux/util.py).
- [MOSAIC wrapper](../../mosaic/runtime.py).
- [FreeCus wrapper](../../freecus/runtime.py) and its
  [paper implementation attention handler](../../freecus/runtime_src/sa_handler_for_flux_clean.py).
- [FreeGraftor archived pipeline](../../freegrafor2/runpod/src_snapshot/src/pipeline.py)
  and [archived provenance](../../freegrafor2/runpod/PROVENANCE.json).
- [VIP-Adapter model composition](../../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/model_loader.py),
  [SVD/projection](../../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/orthogonalization.py),
  [gate construction](../../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/attention_processor.py),
  [semantic LLM calls](../../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/semantic_contrasts.py).

Note: code links above are relative to this file and traverse two levels back to
the repository root. The evidence JSON uses repository-relative paths.

## VIP-Adapter wording and table placement

Recommended method description:

“VIP-Adapter is a training-free inference-time refinement of a pretrained
IP-Adapter. It introduces no additional parameter learning or subject-specific
fine-tuning, while retaining the pretrained adapter as its personalization
backbone.”

For a strict binary **end-to-end dependency** table, put VIP-Adapter in the
personalization-trained-dependency group, annotated to clarify that the dependency
is inherited and the proposed refinement requires no new training. Alternatively,
keep our rows in a separately delineated Ours block after the two baseline groups
and explain the inherited dependency in the caption. Do not put VIP-Adapter in the
training-free end-to-end group while classifying the same underlying IP-Adapter
as training-based, unless the column explicitly uses a different, disclosed scope.

The present code does not establish that VIP-Adapter works on vanilla FLUX
without IP-Adapter or on arbitrary DiT backbones. It acts on IP-conditioned image
features and IP-attention residuals. Broader architecture portability requires
additional implementation and experiments; it does not follow from absence of
training in the proposed interventions.

## Review limits

Original authors' training claims were checked against papers and model cards;
their training jobs were not reproduced. Current local wrappers, vendored
inference implementations, model locks and the archived FreeGraftor code were
inspected. Optimizer/backpropagation searches are supporting checks, not a proof
of training history. Pretrained weights, including generic auxiliary models, were
not re-downloaded or independently retrained. This review does not assert equal
pretraining data or compute between methods.
