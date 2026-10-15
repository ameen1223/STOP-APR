from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TARGETS = [0.90, 0.95, 0.98]
SEEDS = [2026, 2027, 2028]
N_BOOT = 2000


def load_base(root: Path):
    p = root / 'scripts' / '02_STOP_APR_FullProtocol.py'
    if not p.exists():
        raise FileNotFoundError(f'Missing Phase02 script: {p}')
    spec = importlib.util.spec_from_file_location('stopapr_p2', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def threshold_grid(series: pd.Series):
    vals = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(vals) == 0:
        return [0.0]
    return np.unique(np.quantile(vals, np.linspace(0.005, 0.995, 81))).tolist()


def add_scores(base, df: pd.DataFrame, models):
    x = base.add_scores(df, models).copy()
    x['pc_stop_score'] = np.maximum(0.0, x['pred_remaining'].to_numpy(float)) * (
        1.0 - np.clip(x['p_seq'].to_numpy(float), 0.0, 1.0)
    )
    return x


def sim_hi(states: pd.DataFrame, score_col: str, threshold: float):
    d = states[states['remaining_events'] > 0].sort_values(['id', 'decision_index'])
    c = d[d[score_col] > threshold]
    if len(c) == 0:
        return c[['id', 'merged', 'remaining_events', 'decision_index']].copy()
    return c.groupby('id', as_index=False).first()[['id', 'merged', 'remaining_events', 'decision_index']]


def tune_low(base, val: pd.DataFrame, col: str, target: float):
    best = None
    for t in threshold_grid(val[col]):
        st = base.simulate_threshold(val, col, float(t), True)
        m = base.policy_metrics(val, st)
        if m['merged_preservation'] + 1e-12 < target:
            continue
        cand = (m['avoided_event_rate'], m['stop_rate'], float(t), m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        t = float(val[col].min()) - 1e-12
        m = base.policy_metrics(val, base.simulate_threshold(val, col, t, True))
        best = (m['avoided_event_rate'], m['stop_rate'], t, m)
    return best[2], best[3]


def tune_hi(base, val: pd.DataFrame, col: str, target: float):
    best = None
    for t in threshold_grid(val[col]):
        st = sim_hi(val, col, float(t))
        m = base.policy_metrics(val, st)
        if m['merged_preservation'] + 1e-12 < target:
            continue
        cand = (m['avoided_event_rate'], m['stop_rate'], float(t), m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        t = float(val[col].max()) + 1e-12
        m = base.policy_metrics(val, sim_hi(val, col, t))
        best = (m['avoided_event_rate'], m['stop_rate'], t, m)
    return best[2], best[3]


def tune_round(base, val: pd.DataFrame, target: float):
    mx = int(min(30, max(1, val['decision_index'].max())))
    best = None
    for k in range(1, mx + 1):
        st = base.simulate_round(val, k)
        m = base.policy_metrics(val, st)
        if m['merged_preservation'] + 1e-12 < target:
            continue
        cand = (m['avoided_event_rate'], m['stop_rate'], int(k), m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        m = base.policy_metrics(val, [])
        best = (0.0, 0.0, mx + 1, m)
    return best[2], best[3]


@dataclass
class Candidate:
    policy: str
    kind: str
    param: float | int | None
    validation_metrics: dict


def make_candidates(base, val: pd.DataFrame, target: float):
    none_m = base.policy_metrics(val, [])
    k, km = tune_round(base, val, target)
    t0, m0 = tune_low(base, val, 'p_creation', target)
    t1, m1 = tune_low(base, val, 'p_seq', target)
    ts, ms = tune_hi(base, val, 'pc_stop_score', target)
    return {
        'AlwaysContinue': Candidate('AlwaysContinue', 'none', None, none_m),
        'FixedRoundCap': Candidate('FixedRoundCap', 'round', k, km),
        'CreationOnlyHGB': Candidate('CreationOnlyHGB', 'low_creation', t0, m0),
        'SequentialProbHGB': Candidate('SequentialProbHGB', 'low_seq', t1, m1),
        'STOP_APR_PC': Candidate('STOP_APR_PC', 'hi_pc', ts, ms),
    }


def apply_candidate(base, states: pd.DataFrame, cand: Candidate):
    if cand.kind == 'none':
        return states.iloc[0:0][['id', 'merged', 'remaining_events', 'decision_index']].copy()
    if cand.kind == 'round':
        return base.simulate_round(states, int(cand.param))
    if cand.kind == 'low_creation':
        return base.simulate_threshold(states, 'p_creation', float(cand.param), True)
    if cand.kind == 'low_seq':
        return base.simulate_threshold(states, 'p_seq', float(cand.param), True)
    if cand.kind == 'hi_pc':
        return sim_hi(states, 'pc_stop_score', float(cand.param))
    raise ValueError(cand.kind)


def valid_subset(base, x: pd.DataFrame, min_prs: int = 100, min_merged: int = 20):
    if x['id'].nunique() < min_prs:
        return False
    pr = base.pr_first_table(x)
    return pr['merged'].nunique() == 2 and int(pr['merged'].sum()) >= min_merged


def strict_temporal_split(base, states: pd.DataFrame, prs: pd.DataFrame):
    meta = prs[['id', 'terminal_at']].copy()
    pr = base.pr_first_table(states).merge(meta, on='id', how='left')
    pr = pr.dropna(subset=['created_at_dt', 'terminal_at']).sort_values(['created_at_dt', 'id']).reset_index(drop=True)
    n = len(pr)
    if n < 500:
        raise RuntimeError('Too few PRs for strict temporal split.')
    i1 = min(n - 2, max(1, int(n * 0.60)))
    i2 = min(n - 1, max(i1 + 1, int(n * 0.80)))
    cut1 = pr.iloc[i1]['created_at_dt']
    cut2 = pr.iloc[i2]['created_at_dt']

    orig_tr = pr.iloc[:i1].copy()
    orig_va = pr.iloc[i1:i2].copy()
    orig_train_cross = int((orig_tr['terminal_at'] >= cut1).sum())
    orig_val_cross = int((orig_va['terminal_at'] >= cut2).sum())

    trp = pr[(pr['created_at_dt'] < cut1) & (pr['terminal_at'] < cut1)].copy()
    vap = pr[(pr['created_at_dt'] >= cut1) & (pr['created_at_dt'] < cut2) & (pr['terminal_at'] < cut2)].copy()
    tep = pr[pr['created_at_dt'] >= cut2].copy()

    def take(ids):
        return states[states['id'].isin(set(ids.astype(int)))].copy()

    tr, va, te = take(trp['id']), take(vap['id']), take(tep['id'])
    for name, x, mp, mm in [('train', tr, 100, 30), ('validation', va, 100, 30), ('test', te, 100, 30)]:
        if not valid_subset(base, x, mp, mm):
            raise RuntimeError(f'Strict temporal {name} subset is invalid: prs={x.id.nunique()}')

    audit = {
        'cut_train_validation': str(cut1),
        'cut_validation_test': str(cut2),
        'original_creation_split_train_prs_crossing_validation_start': orig_train_cross,
        'original_creation_split_validation_prs_crossing_test_start': orig_val_cross,
        'strict_train_prs': int(tr['id'].nunique()),
        'strict_validation_prs': int(va['id'].nunique()),
        'strict_test_prs': int(te['id'].nunique()),
        'excluded_train_boundary_crossers': int(len(orig_tr) - len(trp)),
        'excluded_validation_boundary_crossers': int(len(orig_va) - len(vap)),
        'strict_train_terminal_before_validation_start': bool((trp['terminal_at'] < cut1).all()),
        'strict_validation_terminal_before_test_start': bool((vap['terminal_at'] < cut2).all()),
    }
    return tr, va, te, audit


def loao_splits(base, states: pd.DataFrame, seed: int = 2026):
    pr = base.pr_first_table(states)
    counts = pr.groupby('agent').agg(n=('id', 'size'), merge_rate=('merged', 'mean')).reset_index()
    eligible = counts[(counts['n'] >= 150) & (counts['merge_rate'].between(0.10, 0.90))]['agent'].tolist()
    for agent in eligible:
        test_ids = set(pr.loc[pr['agent'] == agent, 'id'].astype(int))
        tv = pr[pr['agent'] != agent].copy()
        ok = False
        for off in range(30):
            g = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=seed + off)
            ti, vi = next(g.split(tv, groups=tv['repo_id']))
            trp = tv.iloc[ti].copy(); vap = tv.iloc[vi].copy()
            tr = states[states['id'].isin(set(trp['id'].astype(int)))].copy()
            va = states[states['id'].isin(set(vap['id'].astype(int)))].copy()
            te = states[states['id'].isin(test_ids)].copy()
            if valid_subset(base, tr, 100, 30) and valid_subset(base, va, 100, 20) and valid_subset(base, te, 100, 20):
                ok = True
                break
        if ok:
            yield agent, tr, va, te


def stopped_to_pr(base, test: pd.DataFrame, stopped: pd.DataFrame, suffix: str):
    pr = base.pr_first_table(test)[['id', 'repo_id', 'merged', 'remaining_events']].copy()
    pr = pr.rename(columns={'remaining_events': 'total_downstream_events'})
    sm = stopped[['id', 'remaining_events']].copy() if len(stopped) else pd.DataFrame(columns=['id', 'remaining_events'])
    sm = sm.rename(columns={'remaining_events': f'saved_{suffix}'})
    pr = pr.merge(sm, on='id', how='left')
    pr[f'saved_{suffix}'] = pr[f'saved_{suffix}'].fillna(0).astype(float)
    stop_ids = set(stopped['id'].astype(int)) if len(stopped) else set()
    pr[f'lost_{suffix}'] = ((pr['merged'] == 1) & (pr['id'].astype(int).isin(stop_ids))).astype(int)
    return pr


def cluster_bootstrap_delta(base, test: pd.DataFrame, stopped_ours: pd.DataFrame, stopped_base: pd.DataFrame,
                            seed: int, n_boot: int = N_BOOT):
    a = stopped_to_pr(base, test, stopped_ours, 'ours')
    b = stopped_to_pr(base, test, stopped_base, 'base')[['id', 'saved_base', 'lost_base']]
    pr = a.merge(b, on='id', how='left')
    agg = pr.groupby('repo_id', as_index=False).agg(
        total=('total_downstream_events', 'sum'), saved_ours=('saved_ours', 'sum'), saved_base=('saved_base', 'sum'),
        merged=('merged', 'sum'), lost_ours=('lost_ours', 'sum'), lost_base=('lost_base', 'sum'))
    if len(agg) < 2:
        return {'n_repositories': int(len(agg)), 'delta_ci_low': None, 'delta_ci_high': None,
                'prob_delta_gt_zero': None, 'ours_preservation_ci_low': None, 'ours_preservation_ci_high': None}
    arr = agg[['total', 'saved_ours', 'saved_base', 'merged', 'lost_ours', 'lost_base']].to_numpy(float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    op = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        z = arr[idx].sum(axis=0)
        total = max(1.0, z[0]); merged = max(1.0, z[3])
        deltas[i] = z[1] / total - z[2] / total
        op[i] = 1.0 - z[4] / merged
    return {
        'n_repositories': int(len(agg)),
        'delta_ci_low': float(np.quantile(deltas, 0.025)),
        'delta_ci_high': float(np.quantile(deltas, 0.975)),
        'prob_delta_gt_zero': float(np.mean(deltas > 0)),
        'ours_preservation_ci_low': float(np.quantile(op, 0.025)),
        'ours_preservation_ci_high': float(np.quantile(op, 0.975)),
    }


def evaluate_split(base, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, split: str, seed: int):
    models = base.fit_models(tr, seed)
    va = add_scores(base, va, models)
    te = add_scores(base, te, models)
    all_rows, cmp_rows = [], []

    for target in TARGETS:
        cands = make_candidates(base, va, target)
        baseline_names = ['AlwaysContinue', 'FixedRoundCap', 'CreationOnlyHGB', 'SequentialProbHGB']
        selected_name = max(
            baseline_names,
            key=lambda n: (cands[n].validation_metrics['avoided_event_rate'], cands[n].validation_metrics['stop_rate'])
        )

        evaluated = {}
        for name, cand in cands.items():
            st = apply_candidate(base, te, cand)
            tm = base.policy_metrics(te, st)
            evaluated[name] = (cand, st, tm)
            all_rows.append({
                'split': split, 'seed': int(seed), 'target_preservation': float(target), 'policy': name,
                'tuned_param': cand.param,
                'validation_preservation': float(cand.validation_metrics['merged_preservation']),
                'validation_avoided_event_rate': float(cand.validation_metrics['avoided_event_rate']),
                **tm,
            })

        ours = evaluated['STOP_APR_PC']
        base_sel = evaluated[selected_name]
        o_tm = ours[2]; b_tm = base_sel[2]
        delta = float(o_tm['avoided_event_rate'] - b_tm['avoided_event_rate'])

        diag_pool = []
        for n0 in baseline_names:
            m = evaluated[n0][2]
            diag_pool.append((n0, m['merged_preservation'] >= target - 1e-12,
                              m['avoided_event_rate'], m['merged_preservation']))
        feasible = [x for x in diag_pool if x[1]]
        use = feasible if feasible else diag_pool
        diag_best = max(use, key=lambda x: (x[2], x[3]))

        bs = cluster_bootstrap_delta(base, te, ours[1], base_sel[1],
                                     seed=int(seed + round(target * 1000) + 9000))
        cmp_rows.append({
            'split': split, 'seed': int(seed), 'target_preservation': float(target),
            'stopapr_validation_preservation': float(ours[0].validation_metrics['merged_preservation']),
            'stopapr_validation_avoided_event_rate': float(ours[0].validation_metrics['avoided_event_rate']),
            'stopapr_preservation': float(o_tm['merged_preservation']),
            'stopapr_avoided_event_rate': float(o_tm['avoided_event_rate']),
            'stopapr_meets_target': bool(o_tm['merged_preservation'] >= target - 1e-12),
            'validation_selected_baseline': selected_name,
            'baseline_preservation': float(b_tm['merged_preservation']),
            'baseline_avoided_event_rate': float(b_tm['avoided_event_rate']),
            'baseline_meets_target': bool(b_tm['merged_preservation'] >= target - 1e-12),
            'delta_vs_validation_selected_baseline': delta,
            'diagnostic_best_test_baseline': diag_best[0],
            'diagnostic_best_test_baseline_preservation': float(diag_best[3]),
            'diagnostic_best_test_baseline_avoided_event_rate': float(diag_best[2]),
            'delta_vs_diagnostic_test_oracle_baseline': float(o_tm['avoided_event_rate'] - diag_best[2]),
            'test_n_prs': int(o_tm['n_prs']), 'test_n_merged': int(o_tm['n_merged']),
            **bs,
        })

    diag = {
        'split': split, 'seed': int(seed),
        'train_prs': int(tr['id'].nunique()), 'validation_prs': int(va['id'].nunique()), 'test_prs': int(te['id'].nunique()),
        'train_repositories': int(tr['repo_id'].nunique()), 'validation_repositories': int(va['repo_id'].nunique()),
        'test_repositories': int(te['repo_id'].nunique()), 'test_agents': int(te['agent'].nunique()),
    }
    return pd.DataFrame(all_rows), pd.DataFrame(cmp_rows), diag


def summarize(g: pd.DataFrame):
    if len(g) == 0:
        return {}
    d = g['delta_vs_validation_selected_baseline']
    return {
        'comparisons': int(len(g)),
        'stopapr_target_compliance_count': int(g['stopapr_meets_target'].sum()),
        'stopapr_target_compliance_rate': float(g['stopapr_meets_target'].mean()),
        'wins_vs_validation_selected_baseline': int((d > 1e-12).sum()),
        'ties_vs_validation_selected_baseline': int((d.abs() <= 1e-12).sum()),
        'losses_vs_validation_selected_baseline': int((d < -1e-12).sum()),
        'mean_delta_vs_validation_selected_baseline': float(d.mean()),
        'median_delta_vs_validation_selected_baseline': float(d.median()),
        'bootstrap_ci_excludes_zero_positive_count': int((g['delta_ci_low'] > 0).sum()),
        'bootstrap_positive_probability_ge_0_95_count': int((g['prob_delta_gt_zero'] >= 0.95).sum()),
        'mean_delta_vs_diagnostic_test_oracle_baseline': float(g['delta_vs_diagnostic_test_oracle_baseline'].mean()),
        'median_delta_vs_diagnostic_test_oracle_baseline': float(g['delta_vs_diagnostic_test_oracle_baseline'].median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    logs = root / 'logs'; logs.mkdir(parents=True, exist_ok=True)

    base = load_base(root)
    dfs = base.load_data(root)
    prs = base.prep_prs(dfs['prs'])
    inter, commit = base.interaction_tables(prs, dfs['reviews'], dfs['comments'], dfs['timeline'])
    states = base.build_states(prs, inter, commit)

    all_results, all_cmp, diags = [], [], []

    for seed in SEEDS:
        tr, va, te = base.split_repo_disjoint(states, seed)
        if not base.valid_split(tr, va, te):
            raise RuntimeError(f'Invalid repository-disjoint split for seed {seed}.')
        r, c, d = evaluate_split(base, tr, va, te, 'REPO_DISJOINT', seed)
        all_results.append(r); all_cmp.append(c); diags.append(d)

    tr, va, te, temporal_audit = strict_temporal_split(base, states, prs)
    r, c, d = evaluate_split(base, tr, va, te, 'TEMPORAL_STRICT', 2026)
    all_results.append(r); all_cmp.append(c); diags.append(d)

    held_out_agents = []
    for agent, tr, va, te in loao_splits(base, states, 2026):
        r, c, d = evaluate_split(base, tr, va, te, f'LOAO:{agent}', 2026)
        all_results.append(r); all_cmp.append(c); diags.append(d); held_out_agents.append(agent)

    results = pd.concat(all_results, ignore_index=True)
    cmp = pd.concat(all_cmp, ignore_index=True)
    pd.DataFrame(diags).to_csv(logs / '06_split_diagnostics.csv', index=False)
    results.to_csv(logs / '06_all_policy_results.csv', index=False)
    cmp.to_csv(logs / '06_empirical_comparison.csv', index=False)
    write_json(temporal_audit, logs / '06_temporal_strict_audit.json')

    loao = cmp[cmp['split'].astype(str).str.startswith('LOAO:')].copy()
    if len(loao):
        loao['held_out_agent'] = loao['split'].astype(str).str.replace('LOAO:', '', regex=False)
    loao.to_csv(logs / '06_loao_by_agent.csv', index=False)

    core = cmp[~cmp['split'].astype(str).str.startswith('LOAO:')].copy()
    unseen = cmp[cmp['split'].astype(str).str.startswith('LOAO:')].copy()
    core_s, unseen_s = summarize(core), summarize(unseen)

    summary = {
        'phase': '06_final_empirical_preservation_protocol',
        'status': 'COMPLETE',
        'method_claim': 'Preservation-constrained sequential stopping with thresholds tuned empirically on validation data. No finite-sample or deployment risk guarantee is claimed.',
        'protocol': {
            'repository_disjoint': '3 seeds; model fit on train, all policy parameters and baseline family selected on validation, one-time test evaluation.',
            'temporal': 'Strict temporal 60/20/20 replay; earlier PRs whose terminal outcome crosses the next boundary are excluded.',
            'baseline': 'Primary comparator family selected on validation only; best-on-test comparator is diagnostic only.',
            'uncertainty': f'Paired repository-cluster bootstrap with {N_BOOT} replicates.',
            'loao': 'Per-agent external-validity boundary; not part of the supported deployment claim.',
        },
        'core_repo_plus_strict_temporal': core_s,
        'unseen_agent_boundary': unseen_s,
        'temporal_audit': temporal_audit,
        'held_out_agents': held_out_agents,
        'claim_guard': 'Historical observational replay only. Avoided-event rates are counterfactual replay quantities, not measured causal reviewer-time savings. Validation preservation is an empirical tuning constraint, not a statistical deployment guarantee.',
        'manuscript_decision': 'REVIEW_FINAL_EMPIRICAL_RESULTS',
    }
    write_json(summary, logs / '06_empirical_summary.json')
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
