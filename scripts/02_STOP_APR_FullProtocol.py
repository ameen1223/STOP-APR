from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

SEEDS = [2026, 2027, 2028]
TARGETS = [0.90, 0.95, 0.98]
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def dt(s):
    return pd.to_datetime(s, errors='coerce', utc=True)


def jdump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def load_data(root: Path):
    data = root / 'data'
    paths = {
        'prs': data / 'aidev_pull_request.parquet',
        'reviews': data / 'aidev_pr_reviews.parquet',
        'comments': data / 'aidev_pr_comments.parquet',
        'timeline': data / 'aidev_pr_timeline.parquet',
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    return {k: pd.read_parquet(p) for k, p in paths.items()}


def prep_prs(prs: pd.DataFrame):
    required = {'id', 'created_at', 'closed_at', 'merged_at', 'repo_id', 'agent'}
    miss = required - set(prs.columns)
    if miss:
        raise RuntimeError(f'pull_request missing columns: {sorted(miss)}')
    x = prs.copy()
    x['id'] = pd.to_numeric(x['id'], errors='coerce').astype('Int64')
    x['repo_id'] = pd.to_numeric(x['repo_id'], errors='coerce').astype('Int64')
    x['created_at_dt'] = dt(x['created_at'])
    x['closed_at_dt'] = dt(x['closed_at'])
    x['merged_at_dt'] = dt(x['merged_at'])
    x['terminal_at'] = x['closed_at_dt'].fillna(x['merged_at_dt'])
    x['merged'] = x['merged_at_dt'].notna().astype(int)
    x['agent'] = x['agent'].fillna('UNKNOWN').astype(str)
    x['title_len'] = x.get('title', pd.Series('', index=x.index)).fillna('').astype(str).str.len().clip(0, 10000)
    x['body_len'] = x.get('body', pd.Series('', index=x.index)).fillna('').astype(str).str.len().clip(0, 200000)
    x['created_month'] = x['created_at_dt'].dt.month.fillna(0).astype(int)
    x['created_weekday'] = x['created_at_dt'].dt.weekday.fillna(0).astype(int)
    x = x.dropna(subset=['id', 'repo_id', 'created_at_dt', 'terminal_at']).copy()
    return x


def interaction_tables(prs: pd.DataFrame, reviews: pd.DataFrame, comments: pd.DataFrame, timeline: pd.DataFrame):
    parts = []
    if {'pr_id', 'submitted_at'}.issubset(reviews.columns):
        r = pd.DataFrame({
            'pr_id': pd.to_numeric(reviews['pr_id'], errors='coerce').astype('Int64'),
            'ts': dt(reviews['submitted_at']),
            'kind': 'review',
            'review_state': reviews['state'].fillna('').astype(str) if 'state' in reviews else '',
            'actor': reviews['user'].fillna('').astype(str) if 'user' in reviews else '',
            'body_len_event': reviews['body'].fillna('').astype(str).str.len() if 'body' in reviews else 0,
        })
        parts.append(r)
    if {'pr_id', 'created_at'}.issubset(comments.columns):
        c = pd.DataFrame({
            'pr_id': pd.to_numeric(comments['pr_id'], errors='coerce').astype('Int64'),
            'ts': dt(comments['created_at']),
            'kind': 'comment',
            'review_state': '',
            'actor': comments['user'].fillna('').astype(str) if 'user' in comments else '',
            'body_len_event': comments['body'].fillna('').astype(str).str.len() if 'body' in comments else 0,
        })
        parts.append(c)
    if not parts:
        raise RuntimeError('No review/comment interactions found.')
    rc = pd.concat(parts, ignore_index=True)
    rc = rc.dropna(subset=['pr_id', 'ts'])

    base = prs[['id', 'created_at_dt', 'terminal_at']].rename(columns={'id': 'pr_id'})
    rc = rc.merge(base, on='pr_id', how='inner')
    rc = rc[(rc['ts'] >= rc['created_at_dt']) & (rc['ts'] <= rc['terminal_at'])].copy()
    rc = rc.sort_values(['pr_id', 'ts', 'kind']).reset_index(drop=True)

    commit = pd.DataFrame(columns=['pr_id', 'ts'])
    if {'pr_id', 'created_at', 'event'}.issubset(timeline.columns):
        t = timeline.copy()
        ev = t['event'].fillna('').astype(str).str.lower()
        mask = ev.isin(['committed', 'head_ref_force_pushed', 'synchronize'])
        t = t.loc[mask, ['pr_id', 'created_at']].copy()
        t['pr_id'] = pd.to_numeric(t['pr_id'], errors='coerce').astype('Int64')
        t['ts'] = dt(t['created_at'])
        t = t.dropna(subset=['pr_id', 'ts'])
        t = t.merge(base, on='pr_id', how='inner')
        t = t[(t['ts'] >= t['created_at_dt']) & (t['ts'] <= t['terminal_at'])]
        commit = t[['pr_id', 'ts']].sort_values(['pr_id', 'ts']).reset_index(drop=True)
    return rc, commit


def build_states(prs: pd.DataFrame, rc: pd.DataFrame, commit: pd.DataFrame):
    pr_lookup = prs.set_index('id', drop=False)
    commit_map = {int(pid): g['ts'].astype('int64').to_numpy() for pid, g in commit.groupby('pr_id')} if len(commit) else {}
    rows = []

    for pid, g in rc.groupby('pr_id', sort=False):
        if len(g) < 2:
            continue
        pid_i = int(pid)
        if pid_i not in pr_lookup.index:
            continue
        p = pr_lookup.loc[pid_i]
        g = g.sort_values(['ts', 'kind']).reset_index(drop=True)
        n = len(g)
        review_count = 0
        comment_count = 0
        approve_count = 0
        change_count = 0
        review_comment_count = 0
        cumulative_body = 0
        seen_actors = set()
        prev_ts = p['created_at_dt']
        cts = commit_map.get(pid_i, np.array([], dtype='datetime64[ns]'))

        for i, e in g.iterrows():
            kind = str(e['kind'])
            state = str(e['review_state'] or '').upper()
            if kind == 'review':
                review_count += 1
                if state == 'APPROVED':
                    approve_count += 1
                elif state == 'CHANGES_REQUESTED':
                    change_count += 1
                elif state == 'COMMENTED':
                    review_comment_count += 1
            else:
                comment_count += 1
            cumulative_body += int(e['body_len_event']) if pd.notna(e['body_len_event']) else 0
            actor = str(e['actor'] or '')
            if actor:
                seen_actors.add(actor)

            ts = e['ts']
            elapsed_h = max(0.0, (ts - p['created_at_dt']).total_seconds() / 3600.0)
            gap_h = max(0.0, (ts - prev_ts).total_seconds() / 3600.0)
            prev_ts = ts
            if len(cts):
                # np.searchsorted works on monotone datetime64 array.
                commit_count = int(np.searchsorted(cts, int(ts.value), side='right'))
            else:
                commit_count = 0

            remaining = n - i - 1
            rows.append({
                'id': pid_i,
                'repo_id': int(p['repo_id']),
                'agent': str(p['agent']),
                'created_at_dt': p['created_at_dt'],
                'merged': int(p['merged']),
                'title_len': int(p['title_len']),
                'body_len': int(p['body_len']),
                'created_month': int(p['created_month']),
                'created_weekday': int(p['created_weekday']),
                'decision_index': i + 1,
                'elapsed_hours': elapsed_h,
                'gap_hours': gap_h,
                'current_kind': kind,
                'last_review_state': state if kind == 'review' else '',
                'cum_reviews': review_count,
                'cum_comments': comment_count,
                'cum_approved': approve_count,
                'cum_changes_requested': change_count,
                'cum_review_commented': review_comment_count,
                'cum_commit_events': commit_count,
                'unique_actors': len(seen_actors),
                'current_body_len': int(e['body_len_event']) if pd.notna(e['body_len_event']) else 0,
                'cumulative_body_len': cumulative_body,
                'remaining_events': int(remaining),
                'total_interactions': int(n),
            })
    st = pd.DataFrame(rows)
    if len(st) == 0:
        raise RuntimeError('No sequential state rows built.')
    st['state_rows_per_pr'] = st.groupby('id')['id'].transform('size')
    st['sample_weight'] = 1.0 / st['state_rows_per_pr'].clip(lower=1)
    return st.sort_values(['id', 'decision_index']).reset_index(drop=True)


CREATION_NUM = ['title_len', 'body_len', 'created_month', 'created_weekday']
CREATION_CAT = ['agent']
STATE_NUM = CREATION_NUM + [
    'decision_index', 'elapsed_hours', 'gap_hours', 'cum_reviews', 'cum_comments',
    'cum_approved', 'cum_changes_requested', 'cum_review_commented', 'cum_commit_events',
    'unique_actors', 'current_body_len', 'cumulative_body_len'
]
STATE_CAT = ['agent', 'current_kind', 'last_review_state']


def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), cat_cols),
    ], remainder='drop')


def fit_models(train_states: pd.DataFrame, seed: int):
    first = train_states.sort_values(['id', 'decision_index']).groupby('id', as_index=False).first()

    creation = Pipeline([
        ('prep', make_preprocessor(CREATION_NUM, CREATION_CAT)),
        ('clf', HistGradientBoostingClassifier(
            max_iter=140, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed
        )),
    ])
    creation.fit(first[CREATION_NUM + CREATION_CAT], first['merged'])

    seq = Pipeline([
        ('prep', make_preprocessor(STATE_NUM, STATE_CAT)),
        ('clf', HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0, random_state=seed
        )),
    ])
    seq.fit(
        train_states[STATE_NUM + STATE_CAT], train_states['merged'],
        clf__sample_weight=train_states['sample_weight'].to_numpy()
    )

    cost = Pipeline([
        ('prep', make_preprocessor(STATE_NUM, STATE_CAT)),
        ('reg', HistGradientBoostingRegressor(
            max_iter=160, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0, random_state=seed
        )),
    ])
    cost.fit(
        train_states[STATE_NUM + STATE_CAT], np.log1p(train_states['remaining_events'].to_numpy()),
        reg__sample_weight=train_states['sample_weight'].to_numpy()
    )
    return creation, seq, cost


def add_scores(states: pd.DataFrame, models):
    creation, seq, cost = models
    x = states.copy()
    x['p_creation'] = creation.predict_proba(x[CREATION_NUM + CREATION_CAT])[:, 1]
    x['p_seq'] = seq.predict_proba(x[STATE_NUM + STATE_CAT])[:, 1]
    pred_log_cost = cost.predict(x[STATE_NUM + STATE_CAT])
    x['pred_remaining'] = np.maximum(0.0, np.expm1(pred_log_cost))
    return x


def pr_first_table(states):
    return states.sort_values(['id', 'decision_index']).groupby('id', as_index=False).first()


def simulate_threshold(states: pd.DataFrame, score_col: str, threshold: float, lower_is_stop=True):
    d = states[states['remaining_events'] > 0].sort_values(['id', 'decision_index'])
    if lower_is_stop:
        cand = d[d[score_col] < threshold]
    else:
        cand = d[d[score_col] > threshold]
    if len(cand) == 0:
        return cand[['id', 'merged', 'remaining_events', 'decision_index']].copy()
    return cand.groupby('id', as_index=False).first()[['id', 'merged', 'remaining_events', 'decision_index']]


def simulate_round(states: pd.DataFrame, k: int):
    d = states[(states['remaining_events'] > 0) & (states['decision_index'] >= k)].sort_values(['id', 'decision_index'])
    if len(d) == 0:
        return d[['id', 'merged', 'remaining_events', 'decision_index']].copy()
    return d.groupby('id', as_index=False).first()[['id', 'merged', 'remaining_events', 'decision_index']]


def policy_metrics(states: pd.DataFrame, stopped):
    pr = pr_first_table(states)
    n_merged = int((pr['merged'] == 1).sum())
    total_downstream = int(pr['remaining_events'].sum())
    if stopped is None or len(stopped) == 0:
        stopped_n = 0; stopped_merged = 0; saved = 0; stop_rounds = []
    elif isinstance(stopped, pd.DataFrame):
        stopped_n = int(stopped['id'].nunique())
        stopped_merged = int((stopped['merged'] == 1).sum())
        saved = int(stopped['remaining_events'].sum())
        stop_rounds = stopped['decision_index'].tolist()
    else:
        stopped_n = len({x[0] for x in stopped})
        stopped_merged = sum(1 for x in stopped if x[1] == 1)
        saved = int(sum(x[2] for x in stopped))
        stop_rounds = [x[3] for x in stopped]
    return {
        'n_prs': int(pr['id'].nunique()),
        'n_merged': n_merged,
        'stopped_prs': stopped_n,
        'stop_rate': stopped_n / max(1, int(pr['id'].nunique())),
        'merged_preservation': 1.0 - stopped_merged / max(1, n_merged),
        'merged_loss_rate': stopped_merged / max(1, n_merged),
        'stopped_historical_success_fraction': stopped_merged / max(1, stopped_n),
        'saved_downstream_events': saved,
        'total_downstream_events': total_downstream,
        'avoided_event_rate': saved / max(1, total_downstream),
        'median_stop_round': float(np.median(stop_rounds)) if stop_rounds else None,
    }


def threshold_grid(series: pd.Series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(vals) == 0:
        return [0.5]
    qs = np.linspace(0.02, 0.98, 39)
    return np.unique(np.quantile(vals, qs)).tolist()


def tune_score_policy(val_states, score_col, target):
    best = None
    for t in threshold_grid(val_states[score_col]):
        m = policy_metrics(val_states, simulate_threshold(val_states, score_col, t, True))
        if m['merged_preservation'] + 1e-12 < target:
            continue
        cand = (m['avoided_event_rate'], m['stop_rate'], float(t), m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        # Safest fallback: threshold below minimum -> stop nobody.
        t = float(val_states[score_col].min()) - 1e-12
        m = policy_metrics(val_states, simulate_threshold(val_states, score_col, t, True))
        best = (m['avoided_event_rate'], m['stop_rate'], t, m)
    return best[2], best[3]


def tune_round_policy(val_states, target):
    max_k = int(min(20, max(1, val_states['decision_index'].max())))
    best = None
    for k in range(1, max_k + 1):
        m = policy_metrics(val_states, simulate_round(val_states, k))
        if m['merged_preservation'] + 1e-12 < target:
            continue
        cand = (m['avoided_event_rate'], m['stop_rate'], k, m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        k = max_k + 1
        m = policy_metrics(val_states, [])
        best = (m['avoided_event_rate'], m['stop_rate'], k, m)
    return best[2], best[3]


def add_stopapr_score(df: pd.DataFrame, alpha: float, scale: float):
    x = df.copy()
    denom = 1.0 + alpha * (np.log1p(x['pred_remaining']) / max(scale, 1e-9))
    x['stopapr_score'] = x['p_seq'] / denom
    return x


def tune_stopapr(val_states, target, cost_scale):
    best = None
    for alpha in ALPHAS:
        x = add_stopapr_score(val_states, alpha, cost_scale)
        t, m = tune_score_policy(x, 'stopapr_score', target)
        cand = (m['avoided_event_rate'], m['stop_rate'], alpha, t, m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return {'alpha': best[2], 'threshold': best[3], 'metrics': best[4]}


def evaluate_one(train_states, val_states, test_states, split_name, seed, diagnostics):
    models = fit_models(train_states, seed)
    val = add_scores(val_states, models)
    test = add_scores(test_states, models)
    cost_scale = float(np.median(np.log1p(train_states['remaining_events'])))
    if not np.isfinite(cost_scale) or cost_scale <= 0:
        cost_scale = 1.0

    # AUCs measured at first decision to avoid repeated-state weighting in this diagnostic.
    vf = pr_first_table(val)
    tf = pr_first_table(test)
    auc_creation = float(roc_auc_score(tf['merged'], tf['p_creation'])) if tf['merged'].nunique() > 1 else math.nan
    auc_seq = float(roc_auc_score(tf['merged'], tf['p_seq'])) if tf['merged'].nunique() > 1 else math.nan

    results = []
    for target in TARGETS:
        # Always continue.
        m = policy_metrics(test, [])
        results.append({'split': split_name, 'seed': seed, 'target_preservation': target,
                        'policy': 'AlwaysContinue', 'tuned_param': '', **m})

        # Retrospective oracle: stop closed-unmerged PRs at first decision; never stop merged PRs.
        oracle_stop = test[(test['remaining_events'] > 0) & (test['merged'] == 0)].sort_values(['id', 'decision_index'])
        if len(oracle_stop):
            oracle_stop = oracle_stop.groupby('id', as_index=False).first()[['id', 'merged', 'remaining_events', 'decision_index']]
        else:
            oracle_stop = oracle_stop[['id', 'merged', 'remaining_events', 'decision_index']]
        m = policy_metrics(test, oracle_stop)
        results.append({'split': split_name, 'seed': seed, 'target_preservation': target,
                        'policy': 'RetrospectiveOracle', 'tuned_param': 'outcome_known', **m})

        k, _ = tune_round_policy(val, target)
        m = policy_metrics(test, simulate_round(test, k))
        results.append({'split': split_name, 'seed': seed, 'target_preservation': target,
                        'policy': 'FixedRoundCap', 'tuned_param': f'k={k}', **m})

        t0, _ = tune_score_policy(val, 'p_creation', target)
        m = policy_metrics(test, simulate_threshold(test, 'p_creation', t0, True))
        results.append({'split': split_name, 'seed': seed, 'target_preservation': target,
                        'policy': 'CreationOnlyHGB', 'tuned_param': f'tau={t0:.6g}', **m})

        t1, _ = tune_score_policy(val, 'p_seq', target)
        m = policy_metrics(test, simulate_threshold(test, 'p_seq', t1, True))
        results.append({'split': split_name, 'seed': seed, 'target_preservation': target,
                        'policy': 'SequentialProbHGB', 'tuned_param': f'tau={t1:.6g}', **m})

        tuned = tune_stopapr(val, target, cost_scale)
        xt = add_stopapr_score(test, tuned['alpha'], cost_scale)
        m = policy_metrics(xt, simulate_threshold(xt, 'stopapr_score', tuned['threshold'], True))
        results.append({'split': split_name, 'seed': seed, 'target_preservation': target,
                        'policy': 'STOP_APR',
                        'tuned_param': f"alpha={tuned['alpha']};tau={tuned['threshold']:.6g}", **m})

    diagnostics.update({
        'auc_creation_first_decision': auc_creation,
        'auc_sequential_first_decision': auc_seq,
        'auc_delta': auc_seq - auc_creation if np.isfinite(auc_creation) and np.isfinite(auc_seq) else None,
        'train_state_rows': len(train_states),
        'val_state_rows': len(val_states),
        'test_state_rows': len(test_states),
        'train_prs': int(train_states['id'].nunique()),
        'val_prs': int(val_states['id'].nunique()),
        'test_prs': int(test_states['id'].nunique()),
        'test_repositories': int(test_states['repo_id'].nunique()),
        'test_agents': int(test_states['agent'].nunique()),
    })
    return pd.DataFrame(results), diagnostics


def split_repo_disjoint(states, seed):
    pr = pr_first_table(states)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    tr_idx, temp_idx = next(gss1.split(pr, groups=pr['repo_id']))
    tr_pr = pr.iloc[tr_idx]
    temp_pr = pr.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 101)
    va_idx, te_idx = next(gss2.split(temp_pr, groups=temp_pr['repo_id']))
    va_pr = temp_pr.iloc[va_idx]
    te_pr = temp_pr.iloc[te_idx]
    tr_ids = set(tr_pr['id'].astype(int)); va_ids = set(va_pr['id'].astype(int)); te_ids = set(te_pr['id'].astype(int))
    return states[states['id'].isin(tr_ids)].copy(), states[states['id'].isin(va_ids)].copy(), states[states['id'].isin(te_ids)].copy()


def split_temporal(states):
    pr = pr_first_table(states).sort_values(['created_at_dt', 'id']).reset_index(drop=True)
    n = len(pr)
    a = int(n * 0.60); b = int(n * 0.80)
    tr_ids = set(pr.iloc[:a]['id'].astype(int)); va_ids = set(pr.iloc[a:b]['id'].astype(int)); te_ids = set(pr.iloc[b:]['id'].astype(int))
    return states[states['id'].isin(tr_ids)].copy(), states[states['id'].isin(va_ids)].copy(), states[states['id'].isin(te_ids)].copy()


def valid_split(tr, va, te):
    if min(tr['id'].nunique(), va['id'].nunique(), te['id'].nunique()) < 100:
        return False
    return all(x['merged'].nunique() == 2 for x in [tr, va, te])


def run_loao(states, seed=2026):
    pr = pr_first_table(states)
    results = []
    diags = []
    counts = pr.groupby('agent').agg(n=('id', 'size'), merge_rate=('merged', 'mean')).reset_index()
    eligible = counts[(counts['n'] >= 150) & (counts['merge_rate'].between(0.10, 0.90))]['agent'].tolist()
    for agent in eligible:
        test_ids = set(pr.loc[pr['agent'] == agent, 'id'].astype(int))
        trainval_pr = pr[pr['agent'] != agent].copy()
        if trainval_pr['id'].nunique() < 300:
            continue
        gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=seed)
        tr_i, va_i = next(gss.split(trainval_pr, groups=trainval_pr['repo_id']))
        tr_ids = set(trainval_pr.iloc[tr_i]['id'].astype(int))
        va_ids = set(trainval_pr.iloc[va_i]['id'].astype(int))
        tr = states[states['id'].isin(tr_ids)].copy()
        va = states[states['id'].isin(va_ids)].copy()
        te = states[states['id'].isin(test_ids)].copy()
        if not valid_split(tr, va, te):
            continue
        diag = {'split': f'LOAO:{agent}', 'seed': seed, 'held_out_agent': agent}
        res, diag = evaluate_one(tr, va, te, f'LOAO:{agent}', seed, diag)
        results.append(res); diags.append(diag)
    return results, diags, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    logs = root / 'logs'; logs.mkdir(parents=True, exist_ok=True)

    dfs = load_data(root)
    prs = prep_prs(dfs['prs'])
    rc, commit = interaction_tables(prs, dfs['reviews'], dfs['comments'], dfs['timeline'])
    states = build_states(prs, rc, commit)

    # Frozen protocol audit.
    pr = pr_first_table(states)
    protocol = {
        'phase': '02_frozen_full_protocol',
        'estimand': 'historical downstream review-interaction events avoidable under a sequential stop policy while preserving a target fraction of historically merged PRs',
        'outcome': 'historically merged versus closed-unmerged terminal PR',
        'decision_points': 'ordered review/comment events only',
        'state_evidence': 'creation metadata plus review/comment/commit-like timeline evidence available no later than each decision timestamp',
        'excluded_from_state': ['final CI summary', 'merged_at except as outcome', 'closed_at except terminal censoring', 'future interactions', 'final review totals'],
        'primary_targets_merged_preservation': TARGETS,
        'primary_metric': 'avoided_event_rate at fixed merged-preservation target',
        'secondary_metrics': ['merged_loss_rate', 'stop_rate', 'median_stop_round', 'first-decision AUC'],
        'splits': ['repository-disjoint seeds 2026/2027/2028', 'temporal 60/20/20', 'leave-one-agent-out when n>=150 and both outcomes>=10%'],
        'baselines': ['AlwaysContinue', 'FixedRoundCap', 'CreationOnlyHGB', 'SequentialProbHGB', 'RetrospectiveOracle'],
        'method': 'STOP_APR = sequential success probability discounted by predicted remaining interaction cost; alpha and threshold selected on validation only',
        'causal_scope': 'observational historical replay; no claim of causal reviewer-time savings',
        'state_rows': len(states),
        'decision_ready_prs': int(states['id'].nunique()),
        'repositories': int(states['repo_id'].nunique()),
        'agents': sorted(states['agent'].astype(str).unique().tolist()),
    }
    jdump(protocol, logs / '02_protocol_freeze.json')

    all_results = []
    diagnostics = []

    for seed in SEEDS:
        tr, va, te = split_repo_disjoint(states, seed)
        if not valid_split(tr, va, te):
            diagnostics.append({'split': 'REPO_DISJOINT', 'seed': seed, 'status': 'SKIP_INVALID_SPLIT'})
            continue
        diag = {'split': 'REPO_DISJOINT', 'seed': seed, 'status': 'OK'}
        res, diag = evaluate_one(tr, va, te, 'REPO_DISJOINT', seed, diag)
        all_results.append(res); diagnostics.append(diag)

    tr, va, te = split_temporal(states)
    if valid_split(tr, va, te):
        diag = {'split': 'TEMPORAL', 'seed': 0, 'status': 'OK'}
        res, diag = evaluate_one(tr, va, te, 'TEMPORAL', 2026, diag)
        all_results.append(res); diagnostics.append(diag)
    else:
        diagnostics.append({'split': 'TEMPORAL', 'seed': 0, 'status': 'SKIP_INVALID_SPLIT'})

    loao_res, loao_diag, agent_counts = run_loao(states, 2026)
    all_results.extend(loao_res); diagnostics.extend(loao_diag)

    if not all_results:
        raise RuntimeError('No valid full-protocol splits were produced.')

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(logs / '02_policy_results.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(logs / '02_split_diagnostics.csv', index=False)
    agent_counts.to_csv(logs / '02_agent_counts.csv', index=False)

    # Compact aggregate over non-oracle policies.
    core = results[results['policy'].isin(['CreationOnlyHGB', 'SequentialProbHGB', 'STOP_APR', 'FixedRoundCap'])].copy()
    agg = core.groupby(['split', 'target_preservation', 'policy'], as_index=False).agg(
        runs=('avoided_event_rate', 'size'),
        mean_preservation=('merged_preservation', 'mean'),
        mean_avoided_event_rate=('avoided_event_rate', 'mean'),
        mean_stop_rate=('stop_rate', 'mean'),
        mean_merged_loss_rate=('merged_loss_rate', 'mean'),
    )
    agg.to_csv(logs / '02_policy_aggregate.csv', index=False)

    # Research-oriented summary: STOP-APR delta against strongest non-oracle comparator per run/target.
    deltas = []
    key_cols = ['split', 'seed', 'target_preservation']
    for key, g in results.groupby(key_cols):
        ours = g[g['policy'] == 'STOP_APR']
        base = g[g['policy'].isin(['CreationOnlyHGB', 'SequentialProbHGB', 'FixedRoundCap'])]
        if len(ours) != 1 or len(base) == 0:
            continue
        o = ours.iloc[0]
        eligible = base[base['merged_preservation'] >= key[2] - 1e-12]
        if len(eligible) == 0:
            eligible = base
        b = eligible.sort_values(['avoided_event_rate', 'merged_preservation'], ascending=False).iloc[0]
        deltas.append({
            'split': key[0], 'seed': int(key[1]), 'target_preservation': float(key[2]),
            'stopapr_preservation': float(o['merged_preservation']),
            'stopapr_avoided_event_rate': float(o['avoided_event_rate']),
            'best_baseline': str(b['policy']),
            'baseline_preservation': float(b['merged_preservation']),
            'baseline_avoided_event_rate': float(b['avoided_event_rate']),
            'delta_avoided_event_rate': float(o['avoided_event_rate'] - b['avoided_event_rate']),
        })
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(logs / '02_stopapr_deltas.csv', index=False)

    summary = {
        'phase': '02_frozen_full_protocol',
        'status': 'FULL_PROTOCOL_COMPLETE',
        'data': {
            'state_rows': len(states),
            'decision_ready_prs': int(states['id'].nunique()),
            'repositories': int(states['repo_id'].nunique()),
            'agents': int(states['agent'].nunique()),
            'median_total_interactions': float(pr['total_interactions'].median()),
            'median_downstream_events_from_first_decision': float(pr['remaining_events'].median()),
        },
        'evaluation': {
            'repository_disjoint_runs': int(sum(1 for d in diagnostics if d.get('split') == 'REPO_DISJOINT' and d.get('status') == 'OK')),
            'temporal_runs': int(sum(1 for d in diagnostics if d.get('split') == 'TEMPORAL' and d.get('status') == 'OK')),
            'leave_one_agent_out_runs': int(sum(1 for d in diagnostics if str(d.get('split', '')).startswith('LOAO:'))),
            'targets': TARGETS,
        },
        'stopapr_vs_best_baseline': {
            'comparisons': int(len(delta_df)),
            'positive_delta_count': int((delta_df['delta_avoided_event_rate'] > 0).sum()) if len(delta_df) else 0,
            'nonnegative_delta_count': int((delta_df['delta_avoided_event_rate'] >= 0).sum()) if len(delta_df) else 0,
            'mean_delta_avoided_event_rate': float(delta_df['delta_avoided_event_rate'].mean()) if len(delta_df) else None,
            'median_delta_avoided_event_rate': float(delta_df['delta_avoided_event_rate'].median()) if len(delta_df) else None,
        },
        'interpretation': 'Historical replay only. Avoided-event rates are counterfactual replay quantities, not measured causal reviewer-time savings.',
        'next': 'Upload 02_full_protocol_summary.json, 02_policy_results.csv, and 02_stopapr_deltas.csv for the manuscript decision.'
    }
    jdump(summary, logs / '02_full_protocol_summary.json')
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
