"""STOP-APR Phase 08: frozen-protocol final evidence audit.

This is a diagnostic audit and validation-only sensitivity analysis, not a
search for a new test-optimised final policy. It never modifies earlier files.

The core model/split/policy code is imported from Phase 02, 06 and 07 and
verified using SHA-256 signatures from the Phase 07 release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SEEDS = [2026, 2027, 2028]
TARGETS = [0.90, 0.95, 0.98]
MARGINS = [0.00, 0.01, 0.03]  # validation requirements only
PAIRED_POLICIES = ('STOP_APR_PC', 'BurdenOnlyHGB', 'SequentialProbHGB')
FINGERPRINTS = {
    '02_STOP_APR_FullProtocol.py': '580ccb209404449527f5c8d9d07e1ffe23423ceb2b20892bfd0dc038529810b6',
    '06_STOP_APR_EmpiricalFinal.py': 'cbd95c8c632b2c3a4f471d113f9e90feb69c2ca48f92056b302f560a441c4ba3',
    '07_STOP_APR_LeakageFreeAblation.py': 'fc1c2918050aea1fab1a1165fbb44f89350a0324aa9a5c1efa6577816b903712',
    'aidev_pull_request.parquet': '08f520b5ef36b6281f82090db09ac14aefc3534ee113886e7bc85131fd8d214c',
    'aidev_pr_reviews.parquet': '94051ff602c0ac560dbbc6b34826c1ec1353277d33744718443bc9f3b0532afa',
    'aidev_pr_comments.parquet': 'fa208bafadd5196c6ff414270da4d33a9b13fb65fea47207920ee6df070bf316',
    'aidev_pr_timeline.parquet': '9f45c52c0a7ffa4395ea6c7ce5beb993b3dfb2425ef3848c6d28e2e96471a6f8',
}


def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def json_clean(x):
    """Keep valid numeric diagnostics and serialize non-finite values as null."""
    if x is None or x is pd.NA:
        return None
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return json_clean(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def json_save(path: Path, x):
    path.write_text(json.dumps(json_clean(x), indent=2, default=str, ensure_ascii=True, allow_nan=False), encoding='utf-8')


def integrity_gate(root: Path):
    results = {}
    for fn, expected in FINGERPRINTS.items():
        directory = 'data' if fn.endswith('.parquet') else 'scripts'
        path = root / directory / fn
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen input: {path}')
        actual = file_sha256(path)
        results[fn] = {'matched': actual == expected, 'sha256': actual}
        if actual != expected:
            raise RuntimeError(f'FROZEN_INPUT_MISMATCH: {fn}. Do not overwrite the old project. '
                               'Send the error to the research assistant.')
    return results


def first_state(base, states):
    return base.pr_first_table(states).copy()


def cohort_counts(prs, rc):
    counts = rc.groupby('pr_id').size().rename('observed_interactions')
    tab = prs[['id', 'agent', 'repo_id', 'merged', 'created_at_dt']].drop_duplicates('id').copy()
    tab['observed_interactions'] = tab['id'].map(counts).fillna(0).astype(int)
    tab['cohort'] = np.select(
        [tab['observed_interactions'] == 0, tab['observed_interactions'] == 1],
        ['0 interactions', '1 interaction'], default='2+ interactions'
    )
    grp = tab.groupby(['cohort', 'agent'], as_index=False).agg(
        prs=('id', 'size'), merged_prs=('merged', 'sum'), repositories=('repo_id', 'nunique'),
        historical_merge_rate=('merged', 'mean'))
    overall = tab.groupby('cohort', as_index=False).agg(
        prs=('id', 'size'), merged_prs=('merged', 'sum'), repositories=('repo_id', 'nunique'),
        historical_merge_rate=('merged', 'mean'))
    return tab, grp, overall


def build_single_states(prs, rc, commit):
    """Reconstruct the one-interaction PR states using Phase02 feature semantics.

    Only creates first-observed states with zero recorded downstream events;
    this zero is an evaluation label and is never consulted by simulate().
    """
    counts = rc.groupby('pr_id').size()
    ids = set(counts[counts == 1].index.astype(int))
    one = rc[rc['pr_id'].isin(ids)].copy()
    one = one.drop_duplicates('pr_id')
    pl = prs.set_index('id', drop=False)
    commit_counts = commit[commit['pr_id'].isin(ids)] if len(commit) else commit
    commit_map = {int(pid): g['ts'].astype('int64').to_numpy()
                  for pid, g in commit_counts.groupby('pr_id')} if len(commit_counts) else {}
    rows = []
    for e in one.itertuples(index=False):
        pid = int(e.pr_id)
        if pid not in pl.index:
            continue
        p = pl.loc[pid]
        ts = pd.Timestamp(e.ts)
        st = str(e.review_state or '').upper()
        kind = str(e.kind)
        bl = int(e.body_len_event) if pd.notna(e.body_len_event) else 0
        actor = str(e.actor or '')
        cts = commit_map.get(pid, np.array([], dtype='datetime64[ns]'))
        rows.append({
            'id': pid, 'repo_id': int(p['repo_id']), 'agent': str(p['agent']),
            'created_at_dt': p['created_at_dt'], 'merged': int(p['merged']),
            'title_len': int(p['title_len']), 'body_len': int(p['body_len']),
            'created_month': int(p['created_month']), 'created_weekday': int(p['created_weekday']),
            'decision_index': 1,
            'elapsed_hours': max(0.0, (ts - p['created_at_dt']).total_seconds() / 3600.0),
            'gap_hours': max(0.0, (ts - p['created_at_dt']).total_seconds() / 3600.0),
            'current_kind': kind, 'last_review_state': st if kind == 'review' else '',
            'cum_reviews': int(kind == 'review'), 'cum_comments': int(kind == 'comment'),
            'cum_approved': int(kind == 'review' and st == 'APPROVED'),
            'cum_changes_requested': int(kind == 'review' and st == 'CHANGES_REQUESTED'),
            'cum_review_commented': int(kind == 'review' and st == 'COMMENTED'),
            'cum_commit_events': int(np.searchsorted(cts, int(ts.value), side='right')) if len(cts) else 0,
            'unique_actors': int(bool(actor)), 'current_body_len': bl,
            'cumulative_body_len': bl, 'remaining_events': 0,
            'total_interactions': 1, 'state_rows_per_pr': 1, 'sample_weight': 1.0,
        })
    if not rows:
        return pd.DataFrame(columns=['id', 'repo_id', 'agent', 'created_at_dt',
                                     'merged', 'remaining_events', 'decision_index'])
    return pd.DataFrame(rows).sort_values(['id', 'decision_index']).reset_index(drop=True)


def cohort_test_singles(one, test, split, strict_meta=None):
    if split == 'TEMPORAL_STRICT':
        cut2 = pd.Timestamp(strict_meta['cut_validation_test'])
        return one[one['created_at_dt'] >= cut2].copy()
    repos = set(test['repo_id'].astype(int))
    return one[one['repo_id'].isin(repos)].copy()


def joint_metrics(base, test, extra, original_stops, extra_stops):
    if len(extra) == 0:
        return dict(base.policy_metrics(test, original_stops), one_event_prs=0,
                    extra_merged_stop_count=0, one_event_stop_count=0)
    joined_states = pd.concat([test, extra], ignore_index=True)
    stops = pd.concat([original_stops, extra_stops], ignore_index=True)
    out = base.policy_metrics(joined_states, stops)
    out['one_event_prs'] = int(extra['id'].nunique())
    out['one_event_stop_count'] = int(len(extra_stops))
    out['extra_merged_stop_count'] = int((extra_stops['merged'] == 1).sum())
    return out


def distributions_by_agent(base, tr, va, te, split, seed):
    rows = []
    for section, dat in [('train', tr), ('validation', va), ('test', te)]:
        p = first_state(base, dat)
        count = len(p)
        for agent, g in p.groupby('agent'):
            rows.append({'split': split, 'seed': seed, 'partition': section, 'agent': agent,
                         'n_prs': len(g), 'share': len(g)/max(1,count),
                         'historical_merge_rate': g['merged'].mean(),
                         'mean_total_recorded_interactions': g['total_interactions'].mean()})
    return rows


def drift_numeric(base, trained, va, te, split, seed):
    v = first_state(base, va)
    t = first_state(base, te)
    fields = ['merged', 'decision_index', 'elapsed_hours', 'gap_hours',
              'title_len', 'body_len', 'current_body_len', 'cum_commit_events',
              'p_seq', 'pred_remaining', 'pc_stop_score']
    out = []
    for name in fields:
        v0, t0 = v[name].to_numpy(float), t[name].to_numpy(float)
        mv, mt = float(np.nanmean(v0)), float(np.nanmean(t0))
        spread = float(np.sqrt((np.nanvar(v0) + np.nanvar(t0))/2.0))
        smd = (mt-mv)/spread if spread > 0 else 0.0
        out.append({'split': split, 'seed': seed, 'measure': name,
                    'validation_mean': mv, 'test_mean': mt,
                    'test_minus_validation': mt-mv, 'standardised_mean_difference': smd,
                    'metric_is_historical_outcome_label': bool(name == 'merged')})
    # Agent Jensen-Shannon divergence measures covariate distribution shift.
    av = v['agent'].value_counts(normalize=True)
    at = t['agent'].value_counts(normalize=True)
    names = sorted(set(av.index).union(at.index))
    p = np.array([float(av.get(name,0.0)) for name in names])
    q = np.array([float(at.get(name,0.0)) for name in names])
    m = (p+q)/2
    nzp, nzq = p>0, q>0
    js = 0.5 * (np.sum(p[nzp]*np.log2(p[nzp]/m[nzp])) +
                np.sum(q[nzq]*np.log2(q[nzq]/m[nzq])))
    out.append({'split': split, 'seed': seed, 'measure': 'agent_JS_divergence_bits',
                'validation_mean': None, 'test_mean': float(js),
                'test_minus_validation': None, 'standardised_mean_difference': None,
                'metric_is_historical_outcome_label': False})
    return out


def per_pr_stop_map(base, test, stopped, suffix):
    x = base.pr_first_table(test)[['id','repo_id','agent','merged','remaining_events','current_kind']].copy()
    x = x.rename(columns={'remaining_events':'total_downstream_events','current_kind':'first_event_kind'})
    sm = stopped[['id','remaining_events']].rename(columns={'remaining_events':f'saved_{suffix}'})
    x = x.merge(sm,on='id',how='left')
    x[f'stopped_{suffix}'] = x[f'saved_{suffix}'].notna().astype(int)
    x[f'saved_{suffix}'] = x[f'saved_{suffix}'].fillna(0).astype(float)
    x[f'merged_lost_{suffix}'] = x[f'stopped_{suffix}'] * x['merged']
    return x


def paired_subgroups(base, te, stopped, split, seed, target):
    full = per_pr_stop_map(base,te,stopped['STOP_APR_PC'],'full')
    for competitor, label in [('BurdenOnlyHGB','burden'),('SequentialProbHGB','outcome')]:
        b = per_pr_stop_map(base,te,stopped[competitor],label)
        keep = ['id',f'saved_{label}',f'stopped_{label}',f'merged_lost_{label}']
        full = full.merge(b[keep],on='id',validate='one_to_one')
    results=[]
    for category in ('agent','first_event_kind'):
        for cat, d in full.groupby(category):
            nmerged = int(d['merged'].sum())
            nden = int(d['total_downstream_events'].sum())
            if len(d)<20 or nmerged < 5 or nden<=0:
                continue
            r = {'split':split,'seed':seed,'target_preservation':target,
                 'group_type':category,'group_value':cat,
                 'prs':int(len(d)),'historically_merged_prs':nmerged,
                 'downstream_event_denominator':nden}
            for name in ('full','burden','outcome'):
                r[f'{name}_preservation'] = 1.0-float(d[f'merged_lost_{name}'].sum())/nmerged
                r[f'{name}_avoidance'] = float(d[f'saved_{name}'].sum())/nden
                r[f'{name}_stopped_prs']=int(d[f'stopped_{name}'].sum())
            r['full_minus_burden_avoidance_pp'] = 100*(r['full_avoidance']-r['burden_avoidance'])
            r['full_minus_burden_preservation_pp'] = 100*(r['full_preservation']-r['burden_preservation'])
            results.append(r)
    return results


def evaluate_split(base, p07, tr, va, te, name, seed, one, counts, temporal_meta=None):
    print(f'RUN {name} {seed}: train={tr.id.nunique()} val={va.id.nunique()} test={te.id.nunique()}',flush=True)
    models = base.fit_models(tr,seed)
    scored_va=p07.add_scores(base,va,models)
    scored_te=p07.add_scores(base,te,models)
    saved = []
    pairs=[]
    extent=[]
    threshold=[]
    subgroups=[]
    baseline=[]
    drift=drift_numeric(base,models,scored_va,scored_te,name,seed)
    agents=distributions_by_agent(base,tr,va,te,name,seed)
    single_test=cohort_test_singles(one,te,name,temporal_meta)
    scored_single=p07.add_scores(base,single_test,models) if len(single_test) else single_test
    cache=p07.calibration_cache(base,scored_va)

    for rho in TARGETS:
        cands=p07.choose_candidates(cache,rho)
        chosen=max(p07.BASELINES,key=lambda n:(
            cands[n].validation_metrics['avoided_event_rate'],
            cands[n].validation_metrics['stop_rate']))
        stops={}
        measures={}
        for key in p07.SPECS:
            cand=cands[key]
            s=p07.simulate(scored_te,cand.kind,cand.field,cand.parameter)
            stops[key]=s
            measures[key]=base.policy_metrics(scored_te,s)
        own, bb=measures['STOP_APR_PC'],measures[chosen]
        baseline.append({'split':name,'seed':seed,'target_preservation':rho,
                         'chosen_baseline':chosen,'test_prs':own['n_prs'],
                         'full_test_preservation':own['merged_preservation'],
                         'full_test_avoidance':own['avoided_event_rate'],
                         'baseline_test_preservation':bb['merged_preservation'],
                         'baseline_test_avoidance':bb['avoided_event_rate'],
                         'delta':own['avoided_event_rate']-bb['avoided_event_rate']})
        b=measures['BurdenOnlyHGB']; o=measures['SequentialProbHGB']
        pairs.append({'split':name,'seed':seed,'target_preservation':rho,
                      'full_avoidance':own['avoided_event_rate'],
                      'burden_avoidance':b['avoided_event_rate'],
                      'outcome_avoidance':o['avoided_event_rate'],
                      'full_preservation':own['merged_preservation'],
                      'burden_preservation':b['merged_preservation'],
                      'outcome_preservation':o['merged_preservation'],
                      'full_minus_burden_gain_pp':100*(own['avoided_event_rate']-b['avoided_event_rate']),
                      'full_minus_burden_preservation_pp':100*(own['merged_preservation']-b['merged_preservation']),
                      'full_meets_target':own['merged_preservation']>=rho-1e-12,
                      'burden_meets_target':b['merged_preservation']>=rho-1e-12})
        subgroups.extend(paired_subgroups(base,scored_te,stops,name,seed,rho))
        for key in PAIRED_POLICIES:
            c=cands[key]
            sextra = p07.simulate(scored_single,c.kind,c.field,c.parameter) if len(scored_single) else scored_single
            jm=joint_metrics(base,scored_te,scored_single,stops[key],sextra)
            extent.append({'split':name,'seed':seed,'target_preservation':rho,'policy':key,
                           'original_test_prs':int(te.id.nunique()),
                           'one_interaction_test_prs':jm['one_event_prs'],
                           'extra_one_event_stop_count':jm['one_event_stop_count'],
                           'extra_one_event_merged_stops':jm['extra_merged_stop_count'],
                           'extended_test_prs':jm['n_prs'],
                           'extended_merged_prs':jm['n_merged'],
                           'extended_preservation':jm['merged_preservation'],
                           'extended_avoided_event_rate':jm['avoided_event_rate'],
                           'original_preservation':measures[key]['merged_preservation'],
                           'original_avoided_event_rate':measures[key]['avoided_event_rate'],
                           'extended_meets_target':jm['merged_preservation']>=rho-1e-12})
    # Sensitivity: margins pre-specified, selected ONLY on validation data.
    # Recent-half calibration is evaluated ONLY for strict temporal transfer.
    caches={'all_validation':cache}
    if name=='TEMPORAL_STRICT':
        vp=first_state(base,scored_va).sort_values(['created_at_dt','id'])
        keep=set(vp.iloc[len(vp)//2:]['id'].astype(int))
        recent=scored_va[scored_va['id'].isin(keep)].copy()
        assert recent.id.nunique() >= 100
        caches['most_recent_validation_half']=p07.calibration_cache(base,recent)
    for source,cached in caches.items():
        for rho in TARGETS:
            for margin in MARGINS:
                effective=min(1.0,round(rho+margin,5))
                calibr=p07.choose_candidates(cached,effective)
                for key in PAIRED_POLICIES:
                    cand=calibr[key]
                    s=p07.simulate(scored_te,cand.kind,cand.field,cand.parameter)
                    met=base.policy_metrics(scored_te,s)
                    threshold.append({'split':name,'seed':seed,'calibration_window':source,
                        'policy':key,'target_preservation':rho,
                        'validation_margin':margin,'effective_validation_target':effective,
                        'calibration_preservation':cand.validation_metrics['merged_preservation'],
                        'calibration_avoided_event_rate':cand.validation_metrics['avoided_event_rate'],
                        'test_preservation':met['merged_preservation'],
                        'test_avoided_event_rate':met['avoided_event_rate'],
                        'test_meets_requested_target':met['merged_preservation']>=rho-1e-12,
                        'n_test_prs':met['n_prs'],'stopped_prs':met['stopped_prs'],
                        'parameter':cand.parameter})
    return {'baseline':baseline,'paired':pairs,'extent':extent,'threshold':threshold,
            'subgroups':subgroups,'drift':drift,'agents':agents}


def verify_frozen_phase07(comparisons:pd.DataFrame,ref_path:Path):
    if not ref_path.is_file():
        raise FileNotFoundError('08_reference_phase07.csv is missing from scripts.')
    ref=pd.read_csv(ref_path)
    # Phase08 repeats the three repository seeds and the strict temporal split;
    # the existing five LOAO evaluations remain frozen Phase07 evidence.
    ref=ref[ref['split'].isin(['REPO_DISJOINT','TEMPORAL_STRICT'])].copy()
    key=['split','seed','target_preservation']
    got=comparisons.merge(ref,on=key,validate='one_to_one',suffixes=('','_phase07'),indicator=True)
    if len(got)!=len(ref) or (got['_merge']!='both').any():
        raise RuntimeError('Phase08 core split/target set does not match frozen Phase07.')
    tests={'full_test_preservation':'stopapr_preservation',
           'full_test_avoidance':'stopapr_avoided_event_rate',
           'baseline_test_preservation':'baseline_preservation',
           'baseline_test_avoidance':'baseline_avoided_event_rate',
           'delta':'delta_vs_validation_selected_baseline'}
    max_err=0.0
    for now,old in tests.items():
        e=float(np.max(np.abs(got[now].to_numpy(float)-got[old].to_numpy(float))))
        max_err=max(max_err,e)
    match=(got['chosen_baseline']==got['validation_selected_baseline']).all()
    if max_err>1e-9 or not match:
        raise RuntimeError('Phase08 reproducibility gate failed: Phase07 policy outcomes do not match. '
                           f'Max error={max_err:.4g}, baseline match={match}.')
    return {'phase07_reproduced':True,'comparisons':int(len(got)),
            'max_abs_metric_error':max_err,'baselines_identical':bool(match)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    root=args.root
    logs=root/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    fingerprints=integrity_gate(root)
    print('GATE: frozen source/data SHA-256 PASS',flush=True)
    p07mod=Path(root/'scripts'/'07_STOP_APR_LeakageFreeAblation.py')
    # Module loader retains exact Phase07 code, not older simulation functions.
    import importlib.util
    def load_module(path,name):
        spec=importlib.util.spec_from_file_location(name,path)
        m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m
        spec.loader.exec_module(m)
        return m
    base=load_module(root/'scripts'/'02_STOP_APR_FullProtocol.py','stopapr_base08')
    phase06=load_module(root/'scripts'/'06_STOP_APR_EmpiricalFinal.py','stopapr_phase0608')
    p07=load_module(p07mod,'stopapr_phase0708')
    assert p07.check_replay_behavior(base)=='PASS'
    print('GATE: future-independent replay PASS',flush=True)
    dfs=base.load_data(root)
    prs=base.prep_prs(dfs['prs'])
    rc,commit=base.interaction_tables(prs,dfs['reviews'],dfs['comments'],dfs['timeline'])
    states=base.build_states(prs,rc,commit)
    dataset=p07.audit_dataset(states)
    cohort,by_agent,overall=cohort_counts(prs,rc)
    one=build_single_states(prs,rc,commit)
    assert int((cohort['observed_interactions']==1).sum())==one.id.nunique()
    assert len(set(states.id).intersection(set(one.id)))==0
    print(f'GATE: cohort PASS; {len(states)} states, {states.id.nunique()} original PRs, '
          f'{one.id.nunique()} one-event PRs',flush=True)
    outputs={n:[] for n in ('baseline','paired','extent','threshold','subgroups','drift','agents')}
    for seed in SEEDS:
        tr,va,te=base.split_repo_disjoint(states,seed)
        assert base.valid_split(tr,va,te)
        bundle=evaluate_split(base,p07,tr,va,te,'REPO_DISJOINT',seed,one,cohort)
        for name,rows in bundle.items(): outputs[name].extend(rows)
    tr,va,te,audit=phase06.strict_temporal_split(base,states,prs)
    bundle=evaluate_split(base,p07,tr,va,te,'TEMPORAL_STRICT',2026,one,cohort,audit)
    for name,rows in bundle.items(): outputs[name].extend(rows)
    base_comparison=pd.DataFrame(outputs['baseline'])
    gate=verify_frozen_phase07(base_comparison,root/'scripts'/'08_reference_phase07.csv')
    print('GATE: 12 core Phase07 comparisons exact reproduction PASS',flush=True)
    titles={'baseline':'08_frozen_reproduction.csv',
            'paired':'08_paired_burden_comparison.csv',
            'extent':'08_single_event_cohort_sensitivity.csv',
            'threshold':'08_calibration_sensitivity.csv',
            'subgroups':'08_component_subgroup_diagnostics.csv',
            'drift':'08_temporal_covariate_drift.csv',
            'agents':'08_agent_composition.csv'}
    for key,filename in titles.items():
        pd.DataFrame(outputs[key]).to_csv(logs/filename,index=False)
    by_agent.to_csv(logs/'08_cohort_by_agent.csv',index=False)
    overall.to_csv(logs/'08_cohort_overall.csv',index=False)
    rep=base_comparison[base_comparison['split']=='REPO_DISJOINT']
    with_target=rep[rep.full_test_preservation >= rep.target_preservation-1e-12]
    pair=pd.DataFrame(outputs['paired'])
    strict=base_comparison[base_comparison['split']=='TEMPORAL_STRICT']
    sensitivity=pd.DataFrame(outputs['threshold'])
    temporal_s=sensitivity[(sensitivity['split']=='TEMPORAL_STRICT') &
                           (sensitivity['policy']=='STOP_APR_PC')]
    summary={'status':'COMPLETE','stage':'08_final_evidence_without_test_reselection',
             'integrity':{'source_and_data':fingerprints, 'phase07':gate,
                          'future_information_gate':'PASS'},
             'dataset':dataset,
             'cohort_overall':overall.to_dict(orient='records'),
             'primary_frozen':{'cases':int(len(rep)),
                 'preservation_successes':int((rep.full_test_preservation>=rep.target_preservation-1e-12).sum()),
                 'positive_deltas':int((rep.delta>0).sum()),
                 'all_mean_gain_pp':float(100*rep.delta.mean()),
                 'feasible_cases':int(len(with_target)),
                 'feasible_mean_gain_pp':float(100*with_target.delta.mean())},
             'primary_burden_only':{'mean_full_minus_burden_pp':
                 float(pair[pair['split']=='REPO_DISJOINT'].full_minus_burden_gain_pp.mean()),
                 'mean_full_minus_burden_pp_feasible_cases':float(pair[
                    (pair['split']=='REPO_DISJOINT') & pair['full_meets_target']].full_minus_burden_gain_pp.mean())},
             'strict_temporal':{'preservation_successes':int((strict.full_test_preservation>=strict.target_preservation-1e-12).sum()),
                                'cases':int(len(strict))},
             'calibration_sensitivity_temporal':temporal_s.to_dict(orient='records'),
             'claims_guard':[
                 'Frozen Phase07 results are not replaced by selecting favourable test outcomes.',
                 'Calibration margins and window diagnostics are descriptive sensitivity analyses.',
                 'One-event extension is an exposure/scope audit, not deployment validation.',
                 'Observational replay is not a causal intervention or reviewer-time saving.',
                 'Subgroup diagnostics are descriptive; no inferential subgroup claims.'
             ]}
    json_save(logs/'08_final_evidence_summary.json',summary)
    print('STATUS: COMPLETE',flush=True)
    print('REPO',json.dumps(summary['primary_frozen']),flush=True)
    print('BURDEN',json.dumps(summary['primary_burden_only']),flush=True)
    print('TEMPORAL',json.dumps(summary['strict_temporal']),flush=True)
    print('SAVE all 08_* files in logs',flush=True)


if __name__=='__main__':
    main()
