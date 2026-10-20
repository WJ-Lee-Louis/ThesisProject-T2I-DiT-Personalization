"""Prepare UNO references and persist first-response Module A caption monitoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.contrast_prompt import CAPTION_SCHEMA, CAPTION_SYSTEM_PROMPT
from orthogonalization_instantx_ip_adapter.semantic_contrasts import identity_occurrences, parse_json_object

UNO_COMMIT = "a563432dcfb4caf98df4368facb8102bf97d376c"
DB_COMMIT = "4f887af7970a06fc0cd3adaa1d0b368547d6a1d0"
SELECTED = ROOT / "datasets/dreambench_uno"
OUT = ROOT / "datasets/caption_monitoring/module_a_v2"
DBPLUS = ROOT / "datasets/XVerseBench/dreambench_plus_raw/extracted/data"
PROMPT_DOC = ROOT / "docs/prompts/module_a_caption_system_prompt.md"
REPORT = ROOT / "docs/module_a_caption_monitoring_v2.md"


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def fetch(url, destination):
    if destination.exists():
        return destination.read_bytes()
    with urlopen(url, timeout=60) as response:
        data = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return data


def image_info(path):
    data = path.read_bytes()
    with Image.open(path) as im:
        im.load()
        width, height = im.size
    return dict(image_path=rel(path), image_sha256=sha(data), width=width, height=height)


def prepare():
    provenance = SELECTED / "provenance"
    uno_url = f"https://raw.githubusercontent.com/bytedance/UNO/{UNO_COMMIT}/datasets/dreambench_singleip.json"
    uno_bytes = fetch(uno_url, provenance / "dreambench_singleip.original.json")
    uno = json.loads(uno_bytes)
    classes_url = f"https://raw.githubusercontent.com/google/dreambooth/{DB_COMMIT}/dataset/prompts_and_classes.txt"
    class_bytes = fetch(classes_url, provenance / "prompts_and_classes.txt")
    classes = dict(re.findall(r"^([a-z0-9_]+),([^\r\n,]+)$", class_bytes.decode(), re.M))
    classes.pop("subject_name", None)
    tree_url = f"https://api.github.com/repos/google/dreambooth/git/trees/{DB_COMMIT}?recursive=1"
    tree = json.loads(fetch(tree_url, provenance / "dreambooth_tree.json"))
    if tree.get("truncated"):
        raise ValueError("Incomplete DreamBooth tree")
    blobs = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
    grouped = defaultdict(list)
    for row in uno:
        if len(row["image_paths"]) != 1:
            raise ValueError("Expected single reference per UNO evaluation row")
        source = Path(row["image_paths"][0])
        grouped[source.parent.name].append(row)
    assert len(uno) == 750 and len(grouped) == 30 and set(grouped) == set(classes)
    cases, evaluation = [], []
    for subject, rows in grouped.items():
        paths = {r["image_paths"][0] for r in rows}
        assert len(rows) == 25 and len(paths) == 1
        filename = Path(next(iter(paths))).name
        official_path = f"dataset/{subject}/{filename}"
        expected_blob = blobs[official_path]
        local = ROOT.parent / "dreambooth_reference_dataset" / subject / filename
        dest = SELECTED / "images" / subject / filename
        raw = local.read_bytes() if local.exists() else b""
        blob = lambda b: hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
        if blob(raw) != expected_blob:
            raw = fetch(f"https://raw.githubusercontent.com/google/dreambooth/{DB_COMMIT}/{official_path}", dest)
        assert blob(raw) == expected_blob, official_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != raw:
            raise ValueError(f"Existing reference differs: {dest}")
        dest.write_bytes(raw)
        record = dict(id=f"dreambench/{subject}", benchmark="dreambench", category="subject",
                      subject_id=subject, identity_term=classes[subject],
                      uno_image_path=next(iter(paths)), official_git_blob=expected_blob,
                      target_prompts=[r["prompt"] for r in rows], **image_info(dest))
        cases.append(record)
        evaluation.extend(dict(id=f"{record['id']}/{i:02d}", subject_id=subject,
                               identity_term=classes[subject], prompt=row["prompt"],
                               image_paths=[record["image_path"]]) for i, row in enumerate(rows, 1))
    write_json(SELECTED / "manifest.json", dict(schema_version=1, uno_commit=UNO_COMMIT,
               dreambooth_commit=DB_COMMIT, selection_source=uno_url,
               selection_sha256=sha(uno_bytes), class_source=classes_url, references=cases))
    write_json(SELECTED / "evaluation_750.json", evaluation)
    (SELECTED / "README.md").write_text(
        "# DreamBench — UNO single-subject references\n\n"
        f"Selection: [{UNO_COMMIT}]({uno_url}). DreamBooth submodule: `{DB_COMMIT}`.\n\n"
        "30 subjects × one original image, verified against the pinned DreamBooth Git blob. "
        "Original filenames and subject folders are preserved under `images/`.\n\n"
        "`manifest.json` contains labels, image hashes and 25 UNO prompts per subject. "
        "`evaluation_750.json` contains 750 ready-to-use rows; image paths are relative to the project root. "
        "This reproduces UNO image selection and prompt text, not UNO model preprocessing.\n",
        encoding="utf-8")
    plus = []
    for caption_file in sorted((DBPLUS / "captions").rglob("*.txt")):
        relative = caption_file.relative_to(DBPLUS / "captions")
        lines = caption_file.read_text(encoding="utf-8").splitlines()
        label = lines[0].strip()
        prompts = [s.strip() for s in lines[1:] if s.strip()]
        assert label and len(prompts) == 9, relative
        matches = list((DBPLUS / "images" / relative.parent).glob(relative.stem + ".*"))
        assert len(matches) == 1, relative
        plus.append(dict(id="dreambench_plus/" + relative.with_suffix("").as_posix(),
                         benchmark="dreambench_plus", category=relative.parent.as_posix(),
                         subject_id=relative.with_suffix("").as_posix(), identity_term=label,
                         label_source=rel(caption_file), label_source_sha256=sha(caption_file.read_bytes()),
                         target_prompts=prompts, **image_info(matches[0])))
    assert len(plus) == 150
    assert len(list((DBPLUS / "images").rglob("*.jpg"))) == 150
    cases += plus
    assert len({c["id"] for c in cases}) == 180
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "references.json", cases)
    return cases


def report(cases, metadata, results):
    ordered = [results[c["id"]] for c in cases if c["id"] in results]
    statuses = Counter(r["status"] for r in ordered)
    lines = ["# Module A reference caption 전수 모니터링", "",
             f"- 시작: {metadata['started_at']} (Asia/Seoul)",
             f"- 요청 모델: `{metadata['model']}` / 이미지 + identity_term만 API에 전달",
             "- 대상: DreamBench 30장 + DreamBench++ 150장; target prompt는 전달하지 않음",
             "- 각 이미지의 첫 응답을 보존하며 자동 수정·검증 재시도 없음",
             f"- 진행: {len(ordered)}/180; 구조 검증 결과: `{dict(statuses)}`",
             f"- System prompt SHA-256: `{metadata['prompt_sha256']}`",
             "- [프롬프트 원본](prompts/module_a_caption_system_prompt.md) / "
             "[실행 프롬프트 snapshot](../datasets/caption_monitoring/module_a_v2/system_prompt.txt)",
             "- [전체 JSON](../datasets/caption_monitoring/module_a_v2/captions.json) / "
             "[JSONL](../datasets/caption_monitoring/module_a_v2/captions.jsonl) / "
             "[UNO reference manifest](../datasets/dreambench_uno/manifest.json)",
             "", "PASS는 JSON 키·문자열·identity exact-match 검증 통과입니다. "
             "시각적 정확성, 누락, 상세도, 후치 수식, 불필요한 style 표현은 아래 이미지와 비교하여 검토해야 합니다. "
             "Caption은 원본 응답 그대로이며 각 항목의 검토 칸은 수동 검토 전 상태입니다.", "",
             "| 구분 | 항목 수 | 완료 |", "|---|---:|---:|"]
    groups = sorted({(c['benchmark'], c['category']) for c in cases})
    for benchmark, category in groups:
        subset = [c for c in cases if (c['benchmark'], c['category']) == (benchmark, category)]
        lines.append(f"| {benchmark} / {category} | {len(subset)} | {sum(c['id'] in results for c in subset)} |")
    failures = [r for r in ordered if r['status'] != 'PASS']
    if failures:
        lines += ["", "## 구조 검증 실패 항목", "",
                  "Caption은 보정하거나 재생성하지 않았습니다. 대소문자 차이도 현재 exact-match 계약상 실패로 기록합니다.", ""]
        for result in failures:
            lines.append(f"- `{result['id']}` / `{result['identity_term']}`: {result['error']}")
    for i, case in enumerate(cases, 1):
        result = results.get(case['id'])
        image_link = quote(os.path.relpath(ROOT / case['image_path'], REPORT.parent).replace('\\', '/'), safe='/')
        lines += ["", f"## {i:03d}. {case['id']}", "",
                  f"- Identity: `{case['identity_term']}`",
                  f"- 이미지: [{Path(case['image_path']).name}]({image_link}) ({case['width']} × {case['height']})",
                  f"- 상태: {result['status'] if result else 'PENDING'}", "",
                  f"![{case['id']}]({image_link})", ""]
        if result:
            lines += ["```text", result.get('source_caption') or result.get('raw_response') or result.get('error', ''), "```", ""]
            if result.get('error') and result.get('raw_response'):
                lines += [f"검증 오류: {html.escape(result['error'])}", ""]
        lines += ["검토: □ 시각적 일치 □ 외형·동작·배경 범위 □ 관사·후치 수식 □ 불필요한 style 묘사 없음", "", "메모:"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "captions.json", dict(metadata=metadata, counts=dict(statuses), records=ordered))
    (OUT / "captions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")


def run_case(case, provider, prompt, metadata):
    started = time.monotonic()
    result = dict(case, requested_model=provider.model, prompt_sha256=metadata['prompt_sha256'],
                  started_at=now(), attempt=1)
    try:
        data = (ROOT / case['image_path']).read_bytes()
        assert sha(data) == case['image_sha256']
        raw = provider.complete(system_prompt=prompt, payload={"identity_term": case['identity_term']},
                                schema=CAPTION_SCHEMA, image_bytes=data)
        result['raw_response'] = raw
        parsed = parse_json_object(raw)
        if set(parsed) != {'source_caption', 'preserve'}:
            raise ValueError('Expected source_caption and preserve')
        if isinstance(parsed['source_caption'], str):
            result['source_caption'] = parsed['source_caption']
        identity_occurrences(parsed['source_caption'], case['identity_term'])
        from orthogonalization_instantx_ip_adapter.semantic_contrasts import preservation_phrases
        if not isinstance(parsed['preserve'], list):
            raise ValueError('preserve must be a JSON array')
        preservation_phrases(parsed['preserve'], parsed['source_caption'], case['identity_term'])
        result.update(status='PASS', source_caption=parsed['source_caption'], preserve=parsed['preserve'])
    except Exception as error:
        result.update(status='INVALID' if 'raw_response' in result else 'ERROR',
                      error=f'{type(error).__name__}: {error}')
    result.update(completed_at=now(), seconds=round(time.monotonic() - started, 2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Make paid vision requests; otherwise prepare only')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=180, help='Maximum total cases to complete, including saved results')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.limit <= 180:
        parser.error('workers must be 1..8, limit 1..180')
    # Use the existing reference ledger; a new caption run must not regenerate
    # benchmark manifests or overwrite the original 180-reference audit.
    ledger = ROOT / 'datasets/caption_monitoring/module_a_v1/references.json'
    if not ledger.exists():
        raise ValueError('Existing reference ledger required; prepare benchmark assets separately')
    cases = json.loads(ledger.read_text(encoding='utf-8'))
    reference_path = OUT / 'references.json'
    if reference_path.exists():
        if json.loads(reference_path.read_text(encoding='utf-8')) != cases:
            raise ValueError('Cannot mix reference ledgers; use a new run directory')
    else:
        write_json(reference_path, cases)
    prompt = CAPTION_SYSTEM_PROMPT
    metadata_path = OUT / 'run.json'
    metadata = dict(schema_version=1, started_at=now(), model='gpt-6-luna',
                    prompt_sha256=sha(prompt.encode()), references_sha256=sha((OUT/'references.json').read_bytes()),
                    validation_retries=0, automatic_transport_retries=0,
                    prompt_source='orthogonalization_instantx_ip_adapter/contrast_prompt.py', image_transport='Existing provider RGB PNG conversion; no resize')
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding='utf-8'))
        for key in ['model', 'prompt_sha256', 'references_sha256']:
            if old[key] != metadata[key]:
                raise ValueError(f'Cannot mix monitoring runs: {key} changed')
        metadata = old
    else:
        write_json(metadata_path, metadata)
        (OUT / 'system_prompt.txt').write_text(prompt, encoding='utf-8')
    results = {}
    records_dir = OUT / 'records'
    records_dir.mkdir(exist_ok=True)
    for file in records_dir.glob('*.json'):
        result = json.loads(file.read_text(encoding='utf-8'))
        assert result['prompt_sha256'] == metadata['prompt_sha256']
        results[result['id']] = result
    report(cases, metadata, results)
    print(json.dumps(dict(references=Counter(c['benchmark'] for c in cases), saved=len(results),
                          prompt_sha256=metadata['prompt_sha256']), ensure_ascii=False), flush=True)
    if not args.run:
        return
    provider = ResponsesJSONProvider.from_env(model=metadata['model'])
    provider.timeout = 180
    pending = [c for c in cases[:args.limit] if c['id'] not in results]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, c, provider, prompt, metadata): c for c in pending}
        for future in as_completed(futures):
            result = future.result()
            results[result['id']] = result
            write_json(records_dir / (result['id'].replace('/', '__') + '.json'), result)
            report(cases, metadata, results)
            print(f"{len(results):03d}/180 {result['id']} {result['status']} {result['seconds']}s", flush=True)
    if len(results) == len(cases):
        metadata['completed_at'] = max(r['completed_at'] for r in results.values())
        metadata['completed_count'] = len(results)
        metadata['status_counts'] = dict(Counter(r['status'] for r in results.values()))
        write_json(metadata_path, metadata)
        report(cases, metadata, results)
    print(json.dumps(dict(completed=len(results), statuses=Counter(r['status'] for r in results.values()))), flush=True)


if __name__ == '__main__':
    main()
