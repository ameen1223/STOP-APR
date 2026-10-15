"""Phase 07: leakage-free historical replay and paired component ablations.

Never uses observed remaining_events, merge outcome, or total_interactions to
select an action or establish decision eligibility. Those fields are allowed
only for supervised fitting and retrospective evaluation after action selection.

Requires untouched Phase 02 and 06 sources in D:\\IJIES_STOP_APR\\scripts.
Writes Phase 07 outputs only; original experiments remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = (0.90, 0.95, 0.98)
SEEDS = (2026, 2027, 2028)
BOOT = 2000
EXPECTED_PRS = 9217
EXPECTED_STATES = 53776
FIELDS = ['id', 'merged', 'remaining_events', 'decision_index']
SPECS = {
    'AlwaysContinue': ('none', None),
    'FixedRoundCap': ('round', 'decision_index'),
    'CreationOnlyHGB': ('low', 'p_creation'),
    'SequentialProbHGB': ('low', 'p_seq'),
    'BurdenOnlyHGB': ('high', 'pred_remaining'),
    'STOP_APR_PC': ('high', 'pc_stop_score'),
}
BASELINES = ('AlwaysContinue', 'FixedRoundCap', 'CreationOnlyHGB', 'SequentialProbHGB')


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Candidate:
    policy: str
    kind: str
    field: str | None
    parameter: float | int | None
    validation_metrics: dict


def first_stop(states: pd.DataFrame, hit: np.ndarray | pd.Series) -> pd.DataFrame:
    """Select first STOP from observable score/index; do NOT filter on future r.

    states must be pre-sorted by PR id and decision_index, as in Phase 02.
    remaining_events is copied into the stopped record only AFTER the action.
    """
    mask = np.asarray(hit, dtype=bool)
    if mask.size != len(states):
        raise ValueError('Decision mask must be one bool per observed state.')
    return states.loc[mask, FIELDS].drop_duplicates('id', keep='first').copy()


def simulate(states: pd.DataFrame, kind: str, field: str | None, param):
    if kind == 'none' or param is None:
        return states.iloc[0:0][FIELDS].copy()
    if kind == 'round':
        return first_stop(states, states['decision_index'].to_numpy() >= int(param))
    score = states[field].to_numpy(dtype=float)
    if kind == 'low':
        return first_stop(states, score < float(param))
    if kind == 'high':
        return first_stop(states, score > float(param))
    raise ValueError(f'Unknown decision kind: {kind}')


def grid(x: pd.Series):
    values = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if values.size == 0:
        raise RuntimeError('Cannot calibrate a score that has no finite values.')
    return np.unique(np.quantile(values, np.linspace(0.005, 0.995, 81))).tolist()


def calibration_cache(base, validation: pd.DataFrame):
    """Measure validation candidates once per split, reusable for all 3 targets."""
    result = {}
    always = base.policy_metrics(validation, [])
    result['AlwaysContinue'] = [(None, always)]
    for policy, (kind, field) in SPECS.items():
        if kind == 'none':
            continue
        if kind == 'round':
            max_k = int(min(30, max(1, validation['decision_index'].max())))
            params = list(range(1, max_k + 1))
        else:
            params = grid(validation[field])
        records = []
        for p in params:
            st = simulate(validation, kind, field, p)
            records.append((p, base.policy_metrics(validation, st)))
        records.append((None, always))  # true no-stop, including for long PRs
        result[policy] = records
    return result


def choose_candidates(cache, target: float):
    """Match Phase 06 objective/tie-break: maximum avoided rate, then stop rate."""
    out = {}
    for policy, records in cache.items():
        eligible = [(p, m) for p, m in records if m['merged_preservation'] >= target - 1e-12]
        if not eligible:
            raise RuntimeError(f'No feasible {policy} setting at rho={target}.')
        p, m = max(eligible, key=lambda z: (z[1]['avoided_event_rate'], z[1]['stop_rate']))
        kind, field = SPECS[policy]
        out[policy] = Candidate(policy, kind, field, p, m)
    return out


def add_scores(base, states: pd.DataFrame, fitted) -> pd.DataFrame:
    scored = base.add_scores(states, fitted).copy()
    scored['pc_stop_score'] = (
        np.maximum(0.0, scored['pred_remaining'].to_numpy(dtype=float)) *
        (1.0 - np.clip(scored['p_seq'].to_numpy(dtype=float), 0.0, 1.0))
    )
    return scored.sort_values(['id', 'decision_index']).reset_index(drop=True)


def action_metrics(base, states: pd.DataFrame, stopped: pd.DataFrame) -> dict:
    m = base.policy_metrics(states, stopped)
    # r=0 is legitimately observable only as a retrospective outcome, never
    # used to decide whether to STOP. Count it to audit formerly suppressed stops.
    m['stops_at_last_recorded_event'] = int((stopped['remaining_events'] == 0).sum())
    m['merged_stops_at_last_recorded_event'] = int(
        ((stopped['remaining_events'] == 0) & (stopped['merged'] == 1)).sum()
    )
    return m


def check_replay_behavior(base):
    """Regression tests: a terminal-in-the-record state remains STOP-eligible."""
    toy = pd.DataFrame([
        dict(id=1, merged=1, remaining_events=1, decision_index=1, toy=0.1),
        dict(id=1, merged=1, remaining_events=0, decision_index=2, toy=0.9),
        dict(id=2, merged=0, remaining_events=1, decision_index=1, toy=0.1),
        dict(id=2, merged=0, remaining_events=0, decision_index=2, toy=0.0),
    ])
    st = simulate(toy, 'high', 'toy', 0.5)
    assert len(st) == 1 and int(st.iloc[0]['id']) == 1
    assert int(st.iloc[0]['remaining_events']) == 0
    m = base.policy_metrics(toy, st)
    assert m['saved_downstream_events'] == 0, m
    assert m['merged_preservation'] == 0.0, m
    fixed = simulate(toy, 'round', 'decision_index', 2)
    assert len(fixed) == 2 and (fixed['remaining_events'] == 0).all()
    assert simulate(toy, 'none', None, None).empty
    old = base.simulate_threshold(toy, 'toy', 0.5, False)
    assert old.empty, 'Legacy Phase02 future-mask behavior unexpectedly changed.'
    return 'PASS'


def audit_dataset(states):
    pr_n = states['id'].nunique()
    st_n = len(states)
    last = states[states['remaining_events'] == 0]
    obs_per_pr = states.groupby('id').size()
    if not (last['id'].nunique() == pr_n and len(last) == pr_n):
        raise RuntimeError('State history integrity error: expected one final state per PR.')
    if not (obs_per_pr.min() >= 2 and (states['remaining_events'] >= 0).all()):
        raise RuntimeError('History integrity error: invalid sequential state count.')
    if pr_n != EXPECTED_PRS or st_n != EXPECTED_STATES:
        raise RuntimeError(f'Dataset no longer matches frozen cohort: PRs={pr_n}, states={st_n}; '
                           f'expected {EXPECTED_PRS}, {EXPECTED_STATES}.')
    return {
        'decision_ready_prs': int(pr_n),
        'states': int(st_n),
        'repositories': int(states['repo_id'].nunique()),
        'zero_future_event_states_included_for_decision': int(len(last)),
        'one_final_state_per_pr': True,
        'decision_requires_true_future_event_count': False,
        'decision_uses_only': ['predicted scores or observed decision index'],
        'label_remaining_events_used_only_after_action': True,
    }


def evaluate_split(base, old, train, valid, test, name: str, seed: int):
    print(f'RUN: {name}, seed={seed}, train={train.id.nunique()}, '
          f'val={valid.id.nunique()}, test={test.id.nunique()}', flush=True)
    models = base.fit_models(train, seed)
    va, te = add_scores(base, valid, models), add_scores(base, test, models)
    cached = calibration_cache(base, va)
    policy_rows, comparison_rows, ablation_rows = [], [], []
    for target in TARGETS:
        policies = choose_candidates(cached, target)
        best_baseline = max(
            BASELINES,
            key=lambda p: (policies[p].validation_metrics['avoided_event_rate'],
                           policies[p].validation_metrics['stop_rate'])
        )
        evaluation = {}
        for p, cand in policies.items():
            st = simulate(te, cand.kind, cand.field, cand.parameter)
            tm = action_metrics(base, te, st)
            evaluation[p] = (cand, st, tm)
            policy_rows.append({
                'split': name, 'seed': int(seed), 'target_preservation': target,
                'policy': p, 'tuned_param': cand.parameter,
                'validation_preservation': cand.validation_metrics['merged_preservation'],
                'validation_avoided_event_rate': cand.validation_metrics['avoided_event_rate'],
                **tm,
            })
        own = evaluation['STOP_APR_PC']; b = evaluation[best_baseline]
        om, bm = own[2], b[2]
        boot = old.cluster_bootstrap_delta(base, te, own[1], b[1],
                                           seed=int(seed + round(target * 1000) + 9000), n_boot=BOOT)
        comparison_rows.append({
            'split': name, 'seed': seed, 'target_preservation': target,
            'stopapr_validation_preservation': own[0].validation_metrics['merged_preservation'],
            'stopapr_validation_avoided_event_rate': own[0].validation_metrics['avoided_event_rate'],
            'stopapr_preservation': om['merged_preservation'],
            'stopapr_avoided_event_rate': om['avoided_event_rate'],
            'stopapr_meets_target': bool(om['merged_preservation'] >= target - 1e-12),
            'stopapr_stops_at_last_recorded_event': om['stops_at_last_recorded_event'],
            'stopapr_merged_stops_at_last_recorded_event': om['merged_stops_at_last_recorded_event'],
            'validation_selected_baseline': best_baseline,
            'baseline_preservation': bm['merged_preservation'],
            'baseline_avoided_event_rate': bm['avoided_event_rate'],
            'baseline_meets_target': bool(bm['merged_preservation'] >= target - 1e-12),
            'delta_vs_validation_selected_baseline': float(om['avoided_event_rate'] - bm['avoided_event_rate']),
            'test_n_prs': int(om['n_prs']), 'test_n_merged': int(om['n_merged']),
            **boot,
        })
        # Controlled ablation: same model outputs, same validation-based tuning,
        # same eligibility, same held-out PRs. Only the scoring term changes.
        for competitor, desc, offset in [
            ('SequentialProbHGB', 'outcome_only', 11000),
            ('BurdenOnlyHGB', 'burden_only', 12000),
        ]:
            ab = evaluation[competitor]
            cb = old.cluster_bootstrap_delta(base, te, own[1], ab[1],
                                              seed=int(seed + round(target * 1000) + offset), n_boot=BOOT)
            ablation_rows.append({
                'split': name, 'seed': seed, 'target_preservation': target,
                'comparison': desc, 'full_preservation': om['merged_preservation'],
                'ablation_preservation': ab[2]['merged_preservation'],
                'full_avoided_event_rate': om['avoided_event_rate'],
                'ablation_avoided_event_rate': ab[2]['avoided_event_rate'],
                'gain_full_minus_ablation': float(om['avoided_event_rate'] - ab[2]['avoided_event_rate']),
                'full_meets_target': bool(om['merged_preservation'] >= target - 1e-12),
                'ablation_meets_target': bool(ab[2]['merged_preservation'] >= target - 1e-12),
                **cb,
            })
        print(f'  rho={target:.2f}: preserve={om["merged_preservation"]:.4f}, '
              f'avoid={om["avoided_event_rate"]:.4f}, '
              f'gain={om["avoided_event_rate"]-bm["avoided_event_rate"]:+.4f}, '
              f'last-event STOPs={om["stops_at_last_recorded_event"]}', flush=True)
    diag = {
        'split': name, 'seed': seed,
        'train_prs': int(train['id'].nunique()),
        'validation_prs': int(valid['id'].nunique()),
        'test_prs': int(test['id'].nunique()),
        'train_repositories': int(train['repo_id'].nunique()),
        'validation_repositories': int(valid['repo_id'].nunique()),
        'test_repositories': int(test['repo_id'].nunique()),
        'test_agents': int(test['agent'].nunique()),
    }
    return pd.DataFrame(policy_rows), pd.DataFrame(comparison_rows), pd.DataFrame(ablation_rows), diag


def simple_summary(cmp: pd.DataFrame):
    if cmp.empty:
        return {}
    d = cmp['delta_vs_validation_selected_baseline'].to_numpy(float)
    return {
        'comparisons': int(len(cmp)),
        'target_compliant': int(cmp['stopapr_meets_target'].sum()),
        'wins': int((d > 1e-12).sum()),
        'ties': int((np.abs(d) <= 1e-12).sum()),
        'losses': int((d < -1e-12).sum()),
        'mean_gain_pp': float(d.mean() * 100),
        'median_gain_pp': float(np.median(d) * 100),
        'positive_cluster_ci': int((cmp['delta_ci_low'] > 0).sum()),
        'mean_avoided_event_rate': float(cmp['stopapr_avoided_event_rate'].mean()),
        'mean_preservation': float(cmp['stopapr_preservation'].mean()),
        'last_recorded_event_stops': int(cmp['stopapr_stops_at_last_recorded_event'].sum()),
        'merged_last_recorded_event_stops': int(cmp['stopapr_merged_stops_at_last_recorded_event'].sum()),
    }


def compare_previous(logs: Path, now: pd.DataFrame) -> dict:
    source = logs / '06_empirical_comparison.csv'
    if not source.exists():
        return {'available': False, 'note': 'Phase 06 comparison CSV not found; no old/new comparison claimed.'}
    old = pd.read_csv(source)
    keys = ['split', 'seed', 'target_preservation']
    cols = keys + [
        'stopapr_preservation', 'stopapr_avoided_event_rate',
        'delta_vs_validation_selected_baseline', 'validation_selected_baseline',
        'test_n_prs', 'test_n_merged',
    ]
    if any(c not in old for c in cols):
        return {'available': False, 'note': 'Phase 06 comparison file missing required fields.'}
    previous = old[cols].copy().rename(columns={c: f'old_{c}' for c in cols if c not in keys})
    newer = now[cols].copy().rename(columns={c: f'new_{c}' for c in cols if c not in keys})
    merged = previous.merge(newer, on=keys, how='outer', validate='one_to_one', indicator=True)
    if (merged['_merge'] != 'both').any():
        raise RuntimeError('Old and new Phase 06/07 evaluation splits do not match.')
    for c in ['test_n_prs', 'test_n_merged']:
        if not (merged[f'old_{c}'] == merged[f'new_{c}']).all():
            raise RuntimeError('Old and new test cohorts differ; direct comparison invalid.')
    merged['preservation_change_pp'] = 100 * (
        merged['new_stopapr_preservation'] - merged['old_stopapr_preservation'])
    merged['avoided_event_change_pp'] = 100 * (
        merged['new_stopapr_avoided_event_rate'] - merged['old_stopapr_avoided_event_rate'])
    merged['delta_vs_selected_baseline_change_pp'] = 100 * (
        merged['new_delta_vs_validation_selected_baseline'] - merged['old_delta_vs_validation_selected_baseline'])
    merged.drop(columns=['_merge']).to_csv(logs / '07_old_vs_corrected.csv', index=False)
    return {
        'available': True,
        'old_new_matched_comparisons': int(len(merged)),
        'mean_preservation_change_pp': float(merged['preservation_change_pp'].mean()),
        'mean_avoided_event_change_pp': float(merged['avoided_event_change_pp'].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    logs = root / 'logs'
    logs.mkdir(exist_ok=True, parents=True)
    p02 = root / 'scripts' / '02_STOP_APR_FullProtocol.py'
    p06 = root / 'scripts' / '06_STOP_APR_EmpiricalFinal.py'
    if not p02.exists() or not p06.exists():
        raise FileNotFoundError('Requires untouched scripts 02 and 06 in root/scripts.')
    base = load_module(p02, 'stopapr_phase02_07')
    phase06 = load_module(p06, 'stopapr_phase06_07')
    assert check_replay_behavior(base) == 'PASS'
    print('GATE: replay future-information test PASS', flush=True)

    input_paths = [root / 'data' / f for f in (
        'aidev_pull_request.parquet', 'aidev_pr_reviews.parquet',
        'aidev_pr_comments.parquet', 'aidev_pr_timeline.parquet')]
    dfs = base.load_data(root)
    prs = base.prep_prs(dfs['prs'])
    review, commit = base.interaction_tables(prs, dfs['reviews'], dfs['comments'], dfs['timeline'])
    states = base.build_states(prs, review, commit)
    aud = audit_dataset(states)
    print(f'GATE: data audit PASS ({aud["decision_ready_prs"]} PRs / {aud["states"]} states)', flush=True)

    rows, comparisons, ablations, diagnostics = [], [], [], []
    for seed in SEEDS:
        tr, va, te = base.split_repo_disjoint(states, seed)
        if not base.valid_split(tr, va, te):
            raise RuntimeError(f'Invalid original repository-disjoint split, seed={seed}.')
        r, c, a, d = evaluate_split(base, phase06, tr, va, te, 'REPO_DISJOINT', seed)
        rows.append(r); comparisons.append(c); ablations.append(a); diagnostics.append(d)

    tr, va, te, temporal = phase06.strict_temporal_split(base, states, prs)
    r, c, a, d = evaluate_split(base, phase06, tr, va, te, 'TEMPORAL_STRICT', 2026)
    rows.append(r); comparisons.append(c); ablations.append(a); diagnostics.append(d)

    held_out_agents = []
    for agent, tr, va, te in phase06.loao_splits(base, states, 2026):
        r, c, a, d = evaluate_split(base, phase06, tr, va, te, f'LOAO:{agent}', 2026)
        rows.append(r); comparisons.append(c); ablations.append(a); diagnostics.append(d)
        held_out_agents.append(agent)

    allpol = pd.concat(rows, ignore_index=True)
    cmp = pd.concat(comparisons, ignore_index=True)
    abl = pd.concat(ablations, ignore_index=True)
    pd.DataFrame(diagnostics).to_csv(logs / '07_split_diagnostics.csv', index=False)
    allpol.to_csv(logs / '07_all_policy_results.csv', index=False)
    cmp.to_csv(logs / '07_corrected_comparison.csv', index=False)
    abl.to_csv(logs / '07_component_ablation.csv', index=False)
    dump(temporal, logs / '07_strict_temporal_audit.json')

    olddiff = compare_previous(logs, cmp)
    core_repo = cmp[cmp['split'] == 'REPO_DISJOINT']
    temp = cmp[cmp['split'] == 'TEMPORAL_STRICT']
    unseen = cmp[cmp['split'].str.startswith('LOAO:')]
    summary = {
        'phase': '07_future_information_free_replay_v2',
        'status': 'COMPLETED',
        'decision_guard': 'PASS: STOP evaluated at all observed states; no use of true future event count in decision eligibility.',
        'model_unchanged_from_06': True,
        'retrained_per_split': True,
        'validation_tuned': True,
        'baselines_use_identical_corrected_replay': True,
        'ablation_scores': {
            'outcome_only': '1 - p_seq (thresholded; equivalent to SequentialProbHGB)',
            'burden_only': 'pred_remaining (thresholded)',
            'combined': 'pred_remaining * (1 - p_seq)',
        },
        'bootstrap_replicates': BOOT,
        'dataset': aud,
        'repo_disjoint': simple_summary(core_repo),
        'strict_temporal': simple_summary(temp),
        'leave_one_agent_out': simple_summary(unseen),
        'held_out_agent_families': held_out_agents,
        'ablation_repo_disjoint': {
            desc: {
                'comparisons': int(len(g)),
                'full_wins': int((g['gain_full_minus_ablation'] > 1e-12).sum()),
                'mean_gain_pp': float(g['gain_full_minus_ablation'].mean() * 100),
                'positive_cluster_ci': int((g['delta_ci_low'] > 0).sum()),
            }
            for desc, g in abl[abl['split'] == 'REPO_DISJOINT'].groupby('comparison')
        },
        'phase06_comparison': olddiff,
        'temporal_audit': temporal,
        'script_sha256': {
            'phase02': sha256(p02), 'phase06': sha256(p06),
            'phase07': sha256(Path(__file__)),
        },
        'input_sha256': {p.name: sha256(p) for p in input_paths},
        'claim_guard': 'Observational replay of recorded future events; empirical calibration, not a deployment guarantee or measured reviewer-time savings.',
    }
    dump(summary, logs / '07_leakage_free_summary.json')
    print('STATUS: COMPLETE', flush=True)
    print('REPO_DISJOINT:', json.dumps(summary['repo_disjoint']), flush=True)
    print('TEMPORAL:', json.dumps(summary['strict_temporal']), flush=True)
    print('LOAO:', json.dumps(summary['leave_one_agent_out']), flush=True)
    print('ABLATION:', json.dumps(summary['ablation_repo_disjoint']), flush=True)
    print('PHASE06_COMPARE:', json.dumps(olddiff), flush=True)


if __name__ == '__main__':
    main()
