"""Render every saved final contrast and retry without making API calls."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def render(run, destination, *, preserve_review=True):
    marker=read(run/'run.json'); rows=marker['selection']
    lines=['# 40개 공동 contrast — 전체 원문과 차분 문장','',
           '각 방향은 normalize(e(아래 positive) − e(각 negative))이다. 각 negative는 독립된 원문 기준이며 누적 삭제가 아니다.',
           'PASS는 구조 검증 통과만 뜻한다. 아래 응답과 오류는 수정하지 않은 실제 API 결과다.','',
           '| 번호 | ID | identity | 결과 |','|---|---|---|---|']
    for n,row in enumerate(rows,1):
        v=read(run/'cases'/(row['case_id']+'.json'))
        lines.append(f"| {n} | [{row['case_id']}](#case-{row['case_id']}) | {row['identity_term']} | {v['status']} |")
    negatives=0
    for n,row in enumerate(rows,1):
        v=read(run/'cases'/(row['case_id']+'.json')); cid=row['case_id']
        lines += ['',f'<a id="case-{cid}"></a>',f'## {n:02d}. {cid} — {row["identity_term"]}','',f'- Reference: `{row["reference_id"]}`',f'- 결과: **{v["status"]}**',f'- 알려진 사례: {row.get("known_case") or "추가 benchmark 사례"}']
        cond=v.get('conditioning'); trace=cond['llm']['contrast'] if cond else v.get('trace')
        source=cond['inputs']['source_caption'] if cond else v.get('source_caption')
        if source is None:
            prior=read(ROOT/'datasets/caption_monitoring/module_a_v2/captions.json')['records']
            source=next(x['source_caption'] for x in prior if x['id']==row['reference_id'])
        lines += ['', '**Source positive — 원문 caption**','', '> '+source,'','**Target positive — 생성에도 그대로 전달하는 원문**','', '> '+row['target_prompt'],'']
        if cond:
            if 'preserve' in cond['inputs']:
                lines += ['**A가 전달한 기본 preserve (identity 별도 보호)**', '', json.dumps(cond['inputs']['preserve'], ensure_ascii=False), '']
            lines += ['**Source 보존 정보**','', ', '.join('`'+x+'`' for x in cond['analysis']['source_preserved']),'']
            for side in ('source','target'):
                removals=cond['analysis'][side]; negatives+=len(removals)
                lines += [f'### {side.capitalize()} 차분 ({len(removals)}개)','']
                if not removals: lines+=['빈 목록: 이쪽 제거 방향 없음.','']
                for i,r in enumerate(removals):
                    lines += [f'**{side}[{i}] — {r["unit"]}**','',r['negative'],'']
        else:
            lines += ['**중단 이유**','',v.get('error','Unknown error'),'']
        if trace:
            lines += [f'모델: `{trace["model"]}` · prompt: `{trace["prompt_version"]}` · API attempts: {len(trace["attempts"])}','']
            for i,a in enumerate(trace['attempts']):
                if a['validation_error'] or not cond:
                    lines += [f'<details><summary>응답 {i+1}: 오류/중단 원문</summary>','',a.get('validation_error') or '구조적으로 유효한 중단 응답','', '```json', a.get('raw_response') or 'null','```','','</details>','']
    rendered='\n'.join(lines)+'\n'
    if preserve_review and destination.exists():
        prior=destination.read_text(encoding='utf-8')
        if '<!-- REVIEW_START -->' in prior and '<!-- REVIEW_END -->' in prior:
            review=prior.split('<!-- REVIEW_START -->',1)[1].split('<!-- REVIEW_END -->',1)[0]
            rendered=rendered.replace('| 번호 | ID | identity | 결과 |', '<!-- REVIEW_START -->'+review+'<!-- REVIEW_END -->\n\n| 번호 | ID | identity | 결과 |',1)
    destination.write_text(rendered,encoding='utf-8')
    print(destination.name, 'cases',len(rows),'negatives',negatives)
if __name__=='__main__':
    render(ROOT/'outputs/joint_pilot_gpt56_sol_concept_v6',ROOT/'docs/joint_pilot_gpt56_sol_compact_40.md', preserve_review=False)
