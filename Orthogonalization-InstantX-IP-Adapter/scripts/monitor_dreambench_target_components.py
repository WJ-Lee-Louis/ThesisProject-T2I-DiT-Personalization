"""Analyze all 25 DreamBench templates for two subjects; write Markdown only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.semantic_components import (
    COMPONENT_SYSTEM_PROMPT, SemanticLLM, identity_spans,
)

SOURCE_URL = "https://raw.githubusercontent.com/google/dreambooth/main/dataset/prompts_and_classes.txt"
SUBJECTS = (("backpack", "object"), ("dog", "live_subject"))


def load_cases():
    records = [json.loads(line) for line in
               (ROOT / "datasets/prompt_component_analysis_v1/dreambench_components.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    with urlopen(SOURCE_URL, timeout=30) as response:
        official = response.read().decode("utf-8")
    object_section, live_section = official.split("Object Prompts", 1)[1].split("Live Subject Prompts", 1)
    pattern = r"^'([^']+)'\.format\(unique_token, class_token\),?$"
    templates = {"object": re.findall(pattern, object_section, re.M),
                 "live_subject": re.findall(pattern, live_section, re.M)}
    cases = []
    for subject, category in SUBJECTS:
        selected = sorted((r for r in records if r["subject_id"] == subject),
                          key=lambda r: r["prompt_index"])
        assert len(selected) == len(templates[category]) == 25
        assert [r["prompt_index"] for r in selected] == list(range(25))
        for record, template in zip(selected, templates[category]):
            # Read inputs only. Existing components/annotations are never consumed.
            text = template.format(f"<{subject}>", record["class_name"])
            assert text == record["prompt"]
            identity = f"<{subject}> {record['class_name']}"
            identity_spans(text, identity)
            cases.append(dict(subject=subject, category=category,
                              index=record["prompt_index"] + 1,
                              text=text, identity_term=identity))
    return cases, hashlib.sha256(official.encode()).hexdigest()


class RecordingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.attempts = []

    def complete(self, **kwargs):
        attempt = dict(feedback=kwargs.get("feedback"))
        self.attempts.append(attempt)
        try:
            response = self.provider.complete(**kwargs)
            attempt["response"] = response
            return response
        except Exception as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
            raise


def run_case(case, provider, max_retries=2):
    recording = RecordingProvider(provider)
    result = dict(case, attempts=recording.attempts)
    start = time.monotonic()
    try:
        analysis = SemanticLLM(recording, max_retries=max_retries).analyze(case["text"], case["identity_term"])
        assert analysis.preserved_spans is not None and analysis.neutral_spans is not None
        n = len(analysis.components)
        assert analysis.prompt_for_subset(range(n)) == case["text"]
        fixed = (*analysis.preserved_spans, *analysis.neutral_spans)
        for size in range(n + 1):
            for subset in combinations(range(n), size):
                context = analysis.prompt_for_subset(subset)
                assert all(span in context for span in fixed)
        result.update(status="PASS", output=analysis.to_dict(),
                      all_removed=analysis.prompt_for_subset([]))
    except Exception as error:
        result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    result["seconds"] = round(time.monotonic() - start, 2)
    return result


def cell(value):
    return json.dumps(value, ensure_ascii=False).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def write_report(path, cases, results, model, source_hash, started, max_retries=2):
    success = sum(r["status"] == "PASS" for r in results.values())
    failed = sum(r["status"] == "FAIL" for r in results.values())
    calls = sum(len(r["attempts"]) for r in results.values())
    retry_cases = [r for r in results.values() if len(r["attempts"]) > 1]
    prompt_hash = hashlib.sha256(COMPONENT_SYSTEM_PROMPT.encode()).hexdigest()
    lines = ["# DreamBench target components 전수 모니터링 (50개)", "",
             f"- 실행 시작: {started} (Asia/Seoul)",
             f"- 실제 API 모델: `{model}`; 현재 Module B system prompt와 strict JSON schema 사용",
             f"- 허용 재시도: {max_retries}회 (0이면 각 prompt의 첫 응답만 평가)",
             "- 대상: object `backpack` 25개 + live subject `dog` 25개, 공식 순서 유지",
             "- Caption/Module A, source 분석, embedding 계산, 이미지 생성은 실행하지 않음",
             "- 기존 annotation/cache는 사용하지 않고 각각 새 API 요청; 재사용용 JSON 파일은 생성하지 않음",
             f"- 완료: {len(results)}/50; 구조 검증 PASS {success}, FAIL {failed}; API 시도 {calls}회; 재시도 사례 {len(retry_cases)}개",
             f"- [공식 prompt 원본]({SOURCE_URL})과 로컬 입력 50개가 모두 일치함",
             f"- 공식 원본 SHA-256: `{source_hash}`",
             f"- 실행 system prompt SHA-256: `{prompt_hash}` ([공유 프롬프트](prompts/llm_modules_share.md))", "",
             "## 읽는 방법", "",
             "`<backpack>`과 `<dog>`는 benchmark의 unique-token 자리를 나타내는 로컬 문자열이다. Identity는 각각 `<backpack> backpack`, `<dog> dog`로 지정했다. 마지막 5개 prompt는 `a red …`처럼 관사가 identity와 떨어지므로 `a`를 identity에 포함하지 않는다. 원문은 수정하지 않는다.", "",
             "보존 구간과 중립 구간은 제거하지 않는다. Components의 바깥 배열은 그룹 목록이며, 각 안쪽 배열은 함께 제거되는 span들이다. 예: `[[\"on a\",\"cloud\"]]`는 component 하나다. 빈 배열 `[]`도 실제 LLM 결과 그대로 표시한다.", "",
             "PASS는 JSON/span 타입·원문 일치·identity 보호·비중첩·순서·단어 배정 검증 통과를 뜻한다. 추가로 전체 component를 유지했을 때의 원문 복원과 모든 부분집합에서 고정 span 유지를 검사했다. 의미적 분해의 적절성까지 자동 판정한 점수는 아니다. 단어 분리·관사의 배정·같은 구문의 object/live 차이를 직접 검토할 수 있도록 결과를 수정하지 않고 기록한다.", ""]
    for subject, category in SUBJECTS:
        lines += [f"## {'Object' if category == 'object' else 'Live subject'}: {subject}", "",
                  "| 번호 | Target prompt | 보존 구간 | 중립 구간 | 제거 components (그룹별 spans) | 검증 / API 시도 |",
                  "|---|---|---|---|---|---|"]
        for case in (c for c in cases if c["subject"] == subject):
            result = results.get((subject, case["index"]))
            if result is None:
                fields = ["실행 대기"] * 4
            elif result["status"] == "FAIL":
                fields = ["—", "—", cell(result["error"]), f"FAIL / {len(result['attempts'])}"]
            else:
                output = result["output"]
                groups = [c["spans"] if isinstance(c, dict) else [c] for c in output["components"]]
                fields = [cell(output["preserved_spans"]), cell(output["neutral_spans"]),
                          cell(groups), f"PASS / {len(result['attempts'])}"]
            lines.append(f"| {case['index']:02d} | {cell(case['text'])} | " + " | ".join(fields) + " |")
        lines.append("")
    if retry_cases or failed:
        lines += ["## 재시도 및 실패 기록", ""]
        for result in sorted(results.values(), key=lambda r: (r["subject"], r["index"])):
            if len(result["attempts"]) == 1 and result["status"] == "PASS":
                continue
            lines += [f"### {result['subject']} {result['index']:02d}", ""]
            for index, attempt in enumerate(result["attempts"], 1):
                lines += [f"시도 {index}: 피드백 `{attempt['feedback'] or '없음'}`", "",
                          "```text", attempt.get("response", attempt.get("error", "")), "```", ""]
    lines += ["## 검토 메모", "", "실행 종료 후 50개 전체 결과의 의미 단위와 일관성을 검토하여 기록한다.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Validation retries; use 0 to evaluate first responses only")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/dreambench_target_component_monitoring.md")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    if not 0 <= args.max_retries <= 5:
        parser.error("max-retries must be between 0 and 5")
    if args.output.exists():
        parser.error("output exists; choose a new Markdown path to keep prior observations")
    cases, source_hash = load_cases()
    provider = ResponsesJSONProvider.from_env(args.model)
    started = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    results = {}
    write_report(args.output, cases, results, provider.model, source_hash, started, args.max_retries)

    def record(result):
        results[(result["subject"], result["index"])] = result
        write_report(args.output, cases, results, provider.model, source_hash, started, args.max_retries)
        print(f"{len(results):02d}/50 {result['subject']} {result['index']:02d} "
              f"{result['status']} attempts={len(result['attempts'])}", flush=True)

    # Check one real request before submitting the rest, to avoid repeated auth failures.
    first = run_case(cases[0], provider, args.max_retries)
    record(first)
    if first["status"] == "FAIL" and any("error" in a for a in first["attempts"]):
        print("Transport failure on initial request; stopped before further API calls.", flush=True)
        return 1
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, case, provider, args.max_retries) for case in cases[1:]]
        for future in as_completed(futures):
            record(future.result())
    print(f"Markdown report: {args.output}", flush=True)
    return int(any(r["status"] == "FAIL" for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
