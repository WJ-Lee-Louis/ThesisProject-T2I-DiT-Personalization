"""Read-only live cost estimates from generated DB++ metadata and lane journals."""
import argparse
import json
import statistics
from datetime import datetime,timedelta,timezone
from pathlib import Path
from freecus.manifest import PROJECT
from freecus.run import write_json
from .split_dbpp import STATE,SHARES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',choices=list(SHARES),required=True)
    a=p.parse_args()
    prepared=json.loads((STATE/'plan.json').read_text());part=prepared['plans'][a.workspace]
    from freecus.modal_client import connect
    modal=connect(part['profile']);v=modal.Volume.from_name('xverse-runs')
    name='split_workflows/'+part['bundle_id']+'-'+a.workspace+'.json'
    state=json.loads(b''.join(v.read_file(name)))
    rate_min=.000694+2*.0000131+32*.00000222
    rate_max=.000694+2*.0000131+64*.00000222
    samples=[];lanes=[]
    for call in state.get('calls',[]):
        lane=call['lane'];folder=f"invocations/{state['invocation']}/lane-{lane}"
        try:
            raw=b''.join(v.read_file(folder+'/state.json'))
            try:journal=json.loads(raw)
            except (ValueError,UnicodeError):journal=dict(status='checkpoint_publication')
            text=b''.join(v.read_file(folder+'/worker.log')).decode(errors='replace')
        except FileNotFoundError:continue
        generated=[]
        for line in text.splitlines():
            if line.startswith('{"saved"'):
                try:generated.append(json.loads(line))
                except ValueError:pass
        times=[r['seconds'] for r in generated]
        samples.extend(times)
        lanes.append(dict(lane=lane,status=journal['status'],elapsed_seconds=journal.get('elapsed_seconds'),
                          observed_generated=len(generated),mean_seconds=statistics.mean(times) if times else None,
                          recent_mean_seconds=statistics.mean(times[-10:]) if times else None,
                          assigned=len(part['lanes'][lane]),error_tail=journal.get('error_tail')))
    report=dict(workspace=a.workspace,status=state['status'],assigned=len(part['assigned_case_ids']),
                budget_usd=part['budget'],observed_images=len(samples),lanes=lanes)
    now=datetime.now(timezone(timedelta(hours=9)))
    report['observed_at_kst']=now.isoformat()
    if lanes and all(l['recent_mean_seconds'] is not None for l in lanes):
        remaining=max(max(0,l['assigned']-l['observed_generated'])*max(l['mean_seconds'],l['recent_mean_seconds']) for l in lanes)
        report.update(generation_remaining_seconds=remaining,
            generation_eta_kst=(now+timedelta(seconds=remaining)).isoformat(),
            generation_eta_with_15pct_margin_kst=(now+timedelta(seconds=remaining*1.15)).isoformat(),
            metric_completion_range_kst=[(now+timedelta(seconds=remaining+300)).isoformat(),
                                         (now+timedelta(seconds=remaining*1.15+900)).isoformat()])
    if samples:
        seconds=statistics.mean(samples)
        # Startup and scoring/CPU margin explicit; early category mix is limited.
        report.update(mean_seconds_per_image=seconds,
            generation_compute_usd_range=[len(part['assigned_case_ids'])*seconds*r for r in [rate_min,rate_max]],
            total_with_setup_scoring_allowance_usd=[len(part['assigned_case_ids'])*seconds*r+1.2 for r in [rate_min,rate_max]],
            total_with_15pct_generation_margin_usd=len(part['assigned_case_ids'])*seconds*rate_max*1.15+1.2,
            note='Early observed cases; not a billing balance. Human/style may run at different speeds. 32-64 GiB host RAM range.')
    path=STATE/a.workspace/'cost_estimate.json';write_json(path,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
