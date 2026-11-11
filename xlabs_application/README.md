# XLabs projection application

Completed locally on 2026-11-06: **4,200 original PNGs and 4,200 metric rows**,
verified against 8 Git LFS ZIP parts (6,440,625,039 bytes). Completion evidence:
[`verification/collection.json`](verification/collection.json). Metric tables:
[`results/reports/README.md`](results/reports/README.md).

Isolated implementation: the existing InstantX, XLabs scaffold and other comparison
methods are unchanged. `vendor/SOURCE.json` records the byte hashes of the original
semantic contracts, projection mathematics, evaluator and archive verifier.

The final protocol is 1024x1024 PNG, seed 42, 24 steps, guidance 3.5, XLabs v1
scale 1.0, BF16 FLUX. Conditions are `baseline` and `source_target`, projection
strength 1.0 with norm restoration, no single-stream gate. The source/target
directions use projected CLIP-L/14 text features and edit projected CLIP image
features before the learned XLabs projection. Models and dependencies are pinned.

## Local benchmark bundles

The completed bundle is `598a7ef43356a8eb05e1314b`. The local layout follows the
FreeCus/DreamO/UNO convention:

```text
artifacts/<benchmark>/<bundle>/<shard>/images-*.zip     # Git LFS, <= 1 GiB each
results/<benchmark>/<bundle>/metrics.json              # all image scores + summaries
results/<benchmark>/<bundle>/per_image.csv
results/<benchmark>/<bundle>/manifest.jsonl
results/<benchmark>/<bundle>/run.json
results/<benchmark>/<bundle>/provenance.json
results/<benchmark>/<bundle>/verification.json
results/<benchmark>/<bundle>/generation_source.zip
results/<benchmark>/<bundle>/lanes/<shard>/             # original account/shard records
results/reports/<benchmark>_scores.json
results/reports/<benchmark>_summary.json
verification/<benchmark>.json
outputs/restored/<benchmark>/<bundle>/<shard>/results/<condition>/<benchmark>/*.png
```

`baseline` and `source_target` each contain 750 DreamBench and 1,350 DreamBench++
images. The ZIPs preserve original PNGs, metadata, selected references and LLM
conditioning caches. Root bundle indexes use `xlabs-lane-archives-v1`: they point
to each lane's verified archive index, retaining the original per-lane ZIP bytes.
Raw restored images and client environments remain ignored by Git. Partial
`.download` files are ignored too; completed ZIPs are covered by `.gitattributes`.

After a Git clone and Git LFS download, verify or restore **without Modal, GPU,
model downloads or LLM calls** using Python 3.11 from the project root:

```powershell
git lfs pull --include="xlabs_application/artifacts/**"
python -m xlabs_application.collect verify --benchmark all --bundle-id 598a7ef43356a8eb05e1314b
python -m xlabs_application.collect restore --benchmark all --bundle-id 598a7ef43356a8eb05e1314b
```

`cloud_control download` fetches complete cloud deliverables; `collect assemble`
validates complete case coverage, reference/cache/PNG metric binding and equality
of local versus cloud metric merges before publishing the canonical local bundle.
No other method's metrics or paper tables are updated. No commit or push is made.

DreamBench is 750 cases / 1,500 generated images. DreamBench++ is 1,350 cases /
2,700 images, including style. DreamBench++ completed in wj-lee-louis using two
additional H100 workers, following the user's subsequent authorization.
No LLM request is made. Both authorized source-only fallbacks are retained.

Use the project root and Python 3.11 client:

```powershell
py -3.11 -m venv xlabs_application/.venv-client
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m pip install -r xlabs_application/requirements-client.txt
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client prepare
```

`prepare` requires the original, locally present saved semantic run and original
references. It validates all 2,100 inputs and creates portable input/source ZIPs
under ignored `outputs/`. It writes the canonical benchmark manifests under
`manifests/`. Each DreamBench reference belongs to one account, and both conditions
for a reference stay in that account (375 cases / 750 images per account).

| Explicit profile | Workspace | User-reported remaining credit | Generation admission |
|---|---|---:|---:|
| wj-lee-louis | wj-lee-louis | $7.52 | $12.00 |
| method-comparison | louislee-85468 | $7.83 | $12.00 |

The user subsequently authorized additional DreamBench spending. The generation
admission is now $12/account, including pilot history across older source bundles;
this is an upper admission, not a target charge. Allow a separate ~$1/account
planning reserve for CPU preparation, evaluation/export and uncertainty. These
are estimates, **not a reading of live billing or a Modal hard spending cap**. Credits were supplied on 2026-11-06.
Existing Modal defaults and account spending limits are never changed. Existing
model cache Volumes are reused without deleting their contents; results have a
new `xlabs-application-runs-v1` Volume in each workspace.

For each explicit profile:

```powershell
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client bootstrap --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client deploy --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client stage --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client status --profile wj-lee-louis
```

Bootstrap reads the existing local HF login and registers a dedicated Secret
without printing its value. Wait for model staging to be ready. Run bounded L40S and H100 pilots before full submission and select by measured
cost per image rather than GPU hourly price. Pilot checks baseline/strength-zero image
parity and records actual per-image time plus peak allocated/reserved VRAM.

```powershell
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client pilot --profile wj-lee-louis --budget-usd 0.40
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client status --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client estimate --profile wj-lee-louis
```

Submission uses a **remaining** generation admission: subtract the pilot estimate
from the account generation cap. Each GPU worker saves and commits each PNG/JSON
pair. It stops before its conservative time budget expires. A credit-limited partial
run is never presented as a complete benchmark. Source/config changes require a
new bundle; do not mix runs or delete receipts to bypass duplication checks.

After complete generation, standalone XLabs evaluation uses the unchanged
`uno-selected-reference-v1` CLIP-B/32 and DINOv1 vits16 implementation in a separate
T4 environment. It retains raw scores, reference macro/case micro and class/style
breakdowns. Existing methods' metric tables are outside the scope of this folder.

```powershell
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client evaluate --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client export --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client collect --profile wj-lee-louis
```

ZIP parts preserve original PNG/reference/conditioning bytes, are individually
bounded to 1 GiB and checked by member/ZIP SHA256 and CRC. Large image ZIPs are
under `artifacts/` (Git LFS); standalone scores, source snapshot and provenance are
under `results/` (normal Git). `outputs/` contains temporary local receipts/caches.
No commit, push, merge or changes to `tables_for_paper` are performed automatically.

CPU geometry and contract checks (requires a CPU Torch environment):

```powershell
dreamo/.venv/Scripts/python.exe -X utf8 -m unittest discover -s xlabs_application/tests -v
```

Pricing basis: https://modal.com/pricing (checked 2026-11-06). L40S GPU $0.000542/s,
one physical CPU $0.0000131/s, RAM $0.00000222/GiB/s. Generation estimates charge
the full 64 GiB host RAM limit even though the reservation is 48 GiB. H100 GPU is $0.001097/s. Actual
workspace usage, other jobs and storage/egress can differ; no live balance is claimed.

## Active DreamBench run (2026-11-06)

Frozen bundle: `598a7ef43356a8eb05e1314b`. Both workspaces use one H100 each,
375 cases / 750 images per workspace. Generation admission is $8.50 per invocation
(plus recorded pilot history). DreamBench++ uses a separate additional app. L40S
measured 14.708 seconds/image and H100 6.999 seconds/image, making H100 about 15%
cheaper per image at the conservative CPU/RAM rate. Both pilots produced exact
baseline/strength-zero PNG parity; peak reserved VRAM was about 37.9 GiB.
See `results/reports/preflight.json` for the measurements and call IDs.

A deployed CPU workflow performs generation -> T4 evaluation -> ZIP export in
each workspace. Cloud finalizers verify the archives and metric/PNG bindings and
save source snapshots, per-image CSVs and provenance. Cloud consolidation combines
XLabs account shards and publishes each benchmark's complete summary in wj-lee-louis.
The local PC is not required for any of these steps. Download to the physical PC
can be resumed when it is turned on again. Do not prepare/redeploy a new source
bundle while this run is active. The actual generation source is the frozen
snapshot/source ZIP; the later local Windows atomic-write fix affects status
publication only. No running cloud app was redeployed for that local fix.

Status (safe alongside the monitor):

```powershell
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client status --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.client status --profile method-comparison
```

Cloud status and resumable downloads (these never submit GPU work):

```powershell
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.cloud_control status --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.cloud_control status --profile method-comparison
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.cloud_control download --profile wj-lee-louis
xlabs_application/.venv-client/Scripts/python.exe -X utf8 -m xlabs_application.cloud_control download --profile method-comparison
```

Cloud receipts: `outputs/<bundle_id>/cloud-control/`. Persistent state is in
`offline_state/<bundle_id>/` on each workspace's `xlabs-application-runs-v1` Volume.
Completed shards are under `deliverables/<bundle_id>/<benchmark>/<shard>/`;
full benchmark reports are under `deliverables/<bundle_id>/<benchmark>/combined/`
in wj-lee-louis. ZIPs remain under the `exports/` paths in the verified indexes.
No public download endpoint is created. Cross-account metric collection uses the
existing other-workspace credentials stored in a dedicated Modal Secret in
wj-lee-louis; credentials are never saved to source snapshots or printed.
Other methods' scores and paper tables are not modified.

Vendored files are marked `-text` in `.gitattributes` to preserve original CRLF/LF
byte hashes across Windows/Linux checkouts. Requirement files use LF.

The active CPU continuation supervisors allow up to $12/account generation
compute including pilot history, after the user's additional-spend authorization.
The initial $8.50 GPU invocations are preserved; if one checkpoints on its time
budget, at most one continuation reuses the exact frozen GPU function/source and
complete images before evaluation/export. This changes orchestration only.
Pure inference projected ~$13.15, but actual end-to-end GPU time also includes
PNG encoding and durable Volume commits: early-run total is closer to ~$20.
Workspace usage/spend limits are separate dashboard settings, not changed here.
Controller sources, receipts and hashes are preserved in result execution_control/.

PC-independent execution adds one bounded DreamBench recovery if the old $12
controller stops at its budget checkpoint: the total generation admission becomes
$16/account including pilots, without repeating saved PNGs. Actual expected
DreamBench compute remains around $20--25 across the two accounts.
DreamBench++ is split by reference into 675 cases / 1,350 PNGs per H100. Each
worker allows up to two invocations with a $30 total generation admission; the
combined $60 admission is a guard, while expected compute is approximately $40--50
plus CPU/evaluation/storage charges. None of these are Modal billing hard limits.
The new `xlabs-application-plus-v1` app uses the exact frozen renderer and model
settings. `xlabs-application-offline-v1` finalizes shards, and
`xlabs-application-consolidation-v1` publishes complete reports. The existing GPU
app was not redeployed. Orchestration sources are archived separately from the
frozen generation source. Partial reports are never published as final results.

Validation evidence: eleven CPU contract tests passed. The four-image H100 pilot
also passed actual T4 CLIP/DINO evaluation (`results/reports/metric_smoke.json`)
and ZIP download, CRC/SHA checks, extraction and restored-member SHA checks
(`results/reports/artifact_smoke.json`). These are pipeline checks, not final
DreamBench quality results. Collection can resume after interrupted extraction.
Both workspace usage/spend limits were confirmed enabled by the user.

## Billing interruption recovery (2026-11-06)

The persisted inventory after the interruption was 1,495/1,500 DreamBench PNGs
(750 already evaluated in louislee-85468) and 1,368/2,700 DreamBench++ PNGs.
After the user increased the wj-lee-louis usage limit, stale controller/GPU queues
were cancelled and replaced by checkpoint-aware calls using the identical frozen
renderer. H100 workers skip existing PNGs only after checking their signatures and
hashes. The original calls/receipts are retained under `outputs/<bundle>/recovery/`;
new receipts and recovery source are also persisted under `recovery/<bundle>/`
in the wj-lee-louis results Volume. No extra work is submitted in louislee-85468;
its already completed metric shard is reused. Remaining generation and evaluation
run in wj-lee-louis, followed by the same cloud finalization and consolidation.
`generation_audit.py` provides a separate CPU SHA/PNG/provenance audit; its report
is a timestamped snapshot while generation continues, not a final completeness claim.
The resumed saved-image rates were approximately 7--8 seconds per image per H100.
The conservative 2026-11-06 forecast is generation by 15:30--16:00 KST and metric
evaluation by 16:00--16:30 KST; see `results/reports/recovery_eta.json` for the
observations and assumptions. These are estimates, not completion receipts.
