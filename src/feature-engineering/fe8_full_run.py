#!/usr/bin/env python3
"""FE-8 full stability run from CLI (checkpoint resume)."""
from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from purgedcv import WalkForwardSplit
from purgedcv.diagnostics import assert_no_temporal_leakage

FEATURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_DIR.parents[1]
DATA_PATH = PROJECT_ROOT / 'input' / 'creditcard.csv'
FE6_CONFIG_PATH = FEATURE_DIR / 'MODEL_FEATURES_V2_purgedcv_if.json'
OUTPUT_DIR = FEATURE_DIR / 'output' / 'fe8'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_SEEDS = [42, 123, 2026]
MODELS = ['LightGBM', 'XGBoost']
CV_N_SPLITS = 5
CV_EMBARGO = pd.Timedelta(hours=2)
CV_PURGE_HORIZON = pd.Timedelta(0)
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1500
ES_FRAC = 0.25
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
IF_RANDOM_STATE = 42
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 0.5
IF_CONTAMINATION = 'auto'
IF_MAX_NORMAL_SAMPLES = 50_000
BASELINE_LABEL = 'A0_FE6_SELECTED'
CHECKPOINT_PATH = OUTPUT_DIR / 'fe8_stability_checkpoint.csv'
COMBO_CSV_PATH = OUTPUT_DIR / 'fe8_combo_specs.csv'
RAW_CSV_PATH = OUTPUT_DIR / 'fe8_stability_raw.csv'
SUMMARY_CSV_PATH = OUTPUT_DIR / 'fe8_stability_summary.csv'
DUAL_CSV_PATH = OUTPUT_DIR / 'fe8_dual_model_summary.csv'
OUTPUT_PATH = FEATURE_DIR / 'MODEL_FEATURES_V3_contribution_fe8.json'
RESULT_PATH = FEATURE_DIR / 'FE8_CONTRIBUTION_STABILITY_RESULTS.json'
_CV_BOUND_DATA = None


def read_creditcard_csv(path: Path) -> pd.DataFrame:
    for enc in ('utf-8', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError('cannot read csv')


def build_eda_features(data):
    out = data.copy()
    out['log1p_amount'] = np.log1p(out['Amount'])
    out['hours_since_start'] = (out['Time'] // 3600).astype(int)
    out['is_one_euro'] = out['Amount'] == 1.0
    out['is_amount_1_30'] = (out['Amount'] > 1) & (out['Amount'] <= 30)
    out['is_amount_75_110'] = (out['Amount'] >= 75) & (out['Amount'] <= 110)
    return out


def build_one_euro_cross_features(data, v_cols=('V14', 'V4')):
    out = data.copy()
    gate = out['is_one_euro'].astype(float)
    for v in v_cols:
        out[f'one_euro_{v}'] = gate * out[v]
    return out


def build_fe8_contribution_features(data):
    out = data.copy()
    out['v14_x_v10'] = out['V14'] * out['V10']
    out['v14_x_v4'] = out['V14'] * out['V4']
    out['v10_x_v4'] = out['V10'] * out['V4']
    out['abs_v14_minus_v10'] = (out['V14'] - out['V10']).abs()
    out['v14_x_log1p_amount'] = out['V14'] * out['log1p_amount']
    feats = ['v14_x_v10', 'v14_x_v4', 'v10_x_v4', 'abs_v14_minus_v10', 'v14_x_log1p_amount']
    return out, feats


def bind_cv_data(data):
    global _CV_BOUND_DATA
    out = data.sort_values('Time', kind='mergesort').reset_index(drop=True)
    _CV_BOUND_DATA = out
    return out


def iter_purged_cv_folds(n_samples=None, data=None):
    bound = data if data is not None else _CV_BOUND_DATA
    n = n_samples if n_samples is not None else len(bound)
    pred = pd.to_timedelta(bound['Time'].astype(float), unit='s')
    evalu = pred.copy()
    test_size = max(1, n // (CV_N_SPLITS + 1))
    cv = WalkForwardSplit(n_splits=CV_N_SPLITS, test_size=test_size, window='expanding',
                          prediction_times=pred, evaluation_times=evalu,
                          purge_horizon=CV_PURGE_HORIZON, embargo=CV_EMBARGO)
    for tr_idx, va_idx in cv.split(np.arange(n)):
        assert_no_temporal_leakage(tr_idx, va_idx, prediction_times=pred, evaluation_times=evalu, purge_horizon=CV_PURGE_HORIZON)
        yield tr_idx, va_idx


def split_early_stop_set(X_tr, y_tr, random_state=42):
    return train_test_split(X_tr, y_tr, test_size=ES_FRAC, random_state=random_state, stratify=y_tr)


def make_classifier(model_name, y_train, random_state=42):
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    if model_name == 'LightGBM':
        return lgb.LGBMClassifier(n_estimators=MAX_BOOST_ROUNDS, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced', random_state=random_state, verbose=-1, n_jobs=-1)
    return xgb.XGBClassifier(n_estimators=MAX_BOOST_ROUNDS, learning_rate=0.05, max_depth=6,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=random_state,
        eval_metric='logloss', verbosity=0, n_jobs=-1)


def fit_classifier(clf, model_name, X_tr, y_tr, X_es, y_es):
    if model_name == 'LightGBM':
        clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    else:
        clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return clf


def best_f1_threshold(y_true, proba):
    mask = np.isfinite(proba)
    prec, rec, thr = precision_recall_curve(np.asarray(y_true)[mask], np.asarray(proba)[mask])
    if len(thr) == 0:
        return DEFAULT_CLASSIFICATION_THRESHOLD
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.nanargmax(f1))])


def metrics_at_threshold(y_true, proba, threshold):
    mask = np.isfinite(proba)
    pred = np.asarray(proba)[mask] >= threshold
    y_eval = np.asarray(y_true)[mask]
    tn, fp, fn, tp = confusion_matrix(y_eval, pred).ravel()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'F1@best': float(f1), 'FP': int(fp), 'FN': int(fn)}


def cross_val_eval(model_name, data, feature_cols, random_state=42):
    X, y = data[feature_cols], data['Class']
    oof = np.full(len(y), np.nan)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(iter_purged_cv_folds(len(X)), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        X_fit, X_es, y_fit, y_es = split_early_stop_set(X_tr, y_tr, random_state=random_state + fold)
        clf = make_classifier(model_name, y_fit, random_state=random_state + fold)
        fit_classifier(clf, model_name, X_fit, y_fit, X_es, y_es)
        proba_va = clf.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba_va
        fold_scores.append(float(average_precision_score(y_va, proba_va)))
    arr = np.array(fold_scores)
    thr = best_f1_threshold(y, oof)
    cls = metrics_at_threshold(y, oof, thr)
    return {'AUC-PR_mean': float(arr.mean()), 'AUC-PR_std': float(arr.std(ddof=0)), **cls}


def oof_if_anomaly_score(data, feature_cols):
    X = data[feature_cols].values.astype(np.float64)
    y = data['Class'].values
    oof = np.full(len(y), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(iter_purged_cv_folds(data=data), start=1):
        normal_tr = tr_idx[y[tr_idx] == 0]
        if len(normal_tr) > IF_MAX_NORMAL_SAMPLES:
            rng = np.random.default_rng(IF_RANDOM_STATE + fold)
            normal_tr = rng.choice(normal_tr, size=IF_MAX_NORMAL_SAMPLES, replace=False)
        scaler = StandardScaler()
        X_normal = scaler.fit_transform(X[normal_tr])
        X_valid = scaler.transform(X[va_idx])
        iforest = IsolationForest(n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
            contamination=IF_CONTAMINATION, random_state=IF_RANDOM_STATE + fold, n_jobs=-1)
        iforest.fit(X_normal)
        oof[va_idx] = -iforest.score_samples(X_valid)
        print(f'  IF fold {fold}/{CV_N_SPLITS}', flush=True)
    return oof


def _lbl(cols):
    return '+'.join(sorted(cols))


def build_fe8_specs(fe6_extra, fe8_feats):
    records, seen = [], set()
    def add(category, label, fe6_cols, fe8_cols):
        fe6_cols, fe8_cols = list(fe6_cols), list(fe8_cols)
        extra = fe6_cols + [c for c in fe8_cols if c not in fe6_cols]
        key = (tuple(sorted(fe6_cols)), tuple(sorted(fe8_cols)))
        if key in seen:
            return
        seen.add(key)
        records.append({'combo_id': len(records), 'category': category, 'label': label,
            'fe6_cols': fe6_cols, 'fe8_cols': fe8_cols, 'extra_cols': extra,
            'n_fe6': len(fe6_cols), 'n_fe8': len(fe8_cols), 'n_extra': len(extra)})
    add('anchor', BASELINE_LABEL, fe6_extra, [])
    for k in range(1, len(fe8_feats) + 1):
        for s in combinations(fe8_feats, k):
            s = list(s)
            add('fe6_plus_fe8_subset', f'B_FE6+FE8_k{k}_{_lbl(s)}', fe6_extra, s)
    for r in range(1, len(fe6_extra)):
        for s in combinations(fe6_extra, r):
            s = list(s)
            add('fe6_partial_plus_fe8_all', f'C_FE6_n{len(s)}+FE8_ALL_{_lbl(s)}', s, fe8_feats)
    add('anchor', 'D_BASE+FE8_ALL', [], fe8_feats)
    return records


def export_catalog(specs):
    rows = [{'combo_id': r['combo_id'], 'category': r['category'], 'label': r['label'],
             'n_fe6': r['n_fe6'], 'n_fe8': r['n_fe8'], 'n_extra': r['n_extra'],
             'fe6_cols': ' | '.join(r['fe6_cols']) if r['fe6_cols'] else '(none)',
             'fe8_cols': ' | '.join(r['fe8_cols']) if r['fe8_cols'] else '(none)',
             'extra_cols': ' | '.join(r['extra_cols'])} for r in specs]
    pd.DataFrame(rows).to_csv(COMBO_CSV_PATH, index=False, encoding='utf-8-sig')


def eval_once(rec, model, seed, df, base_features):
    cols = base_features + [c for c in rec['extra_cols'] if c not in base_features]
    res = cross_val_eval(model, df, cols, seed)
    return {'combo_id': rec['combo_id'], 'category': rec['category'], '特征组合': rec['label'], '模型': model,
            'seed': seed, 'n_fe6': rec['n_fe6'], 'n_fe8': rec['n_fe8'], 'n_extra': rec['n_extra'],
            'fe6_cols': ' | '.join(rec['fe6_cols']), 'fe8_cols': ' | '.join(rec['fe8_cols']),
            'extra_cols': ' | '.join(rec['extra_cols']), **res}


def run_matrix(specs, df, base_features):
    done = {}
    if CHECKPOINT_PATH.is_file():
        prev = pd.read_csv(CHECKPOINT_PATH)
        if '特征组合' in prev.columns and len(prev) > 0:
            for _, row in prev.iterrows():
                done[(row['特征组合'], row['模型'], int(row['seed']))] = row.to_dict()
            print(f'restored {len(done)} checkpoint rows')
    rows = list(done.values())
    total = len(specs) * len(MODELS) * len(RUN_SEEDS)
    step = len(rows)
    for rec in specs:
        for model in MODELS:
            for seed in RUN_SEEDS:
                key = (rec['label'], model, seed)
                if key in done:
                    continue
                step += 1
                print(f'[{step}/{total}] {model} seed={seed} | {rec["label"]}', flush=True)
                row = eval_once(rec, model, seed, df, base_features)
                rows.append(row)
                done[key] = row
                pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False, encoding='utf-8-sig')
    return pd.DataFrame(rows)


def summarize(raw):
    base = raw[raw['特征组合'] == BASELINE_LABEL].groupby(['模型', 'seed'])['AUC-PR_mean'].mean().reset_index()
    base = base.rename(columns={'AUC-PR_mean': 'BASELINE_AUC_seed'})
    merged = raw.merge(base, on=['模型', 'seed'])
    merged['delta_AUC_vs_FE6_seed'] = merged['AUC-PR_mean'] - merged['BASELINE_AUC_seed']
    summary = merged.groupby(['特征组合', '模型']).agg(
        combo_id=('combo_id', 'first'), category=('category', 'first'),
        n_fe6=('n_fe6', 'first'), n_fe8=('n_fe8', 'first'), extra_cols=('extra_cols', 'first'),
        delta_mean_AUC=('delta_AUC_vs_FE6_seed', 'mean'),
        delta_std_AUC=('delta_AUC_vs_FE6_seed', 'std'),
        positive_seed_ratio=('delta_AUC_vs_FE6_seed', lambda s: float((s > 0).mean())),
    ).reset_index()
    summary['conservative_score'] = summary['delta_mean_AUC'] - summary['delta_std_AUC'].fillna(0)
    return merged, summary


def dual_summary_fn(summary):
    lgb = summary[summary['模型'] == 'LightGBM'].set_index('特征组合')
    xgb = summary[summary['模型'] == 'XGBoost'].set_index('特征组合')
    rows = []
    for label in lgb.index.intersection(xgb.index):
        rows.append({
            'combo_label': label,
            'extra_cols': lgb.loc[label, 'extra_cols'],
            'delta_mean_AUC': float((lgb.loc[label, 'delta_mean_AUC'] + xgb.loc[label, 'delta_mean_AUC']) / 2),
            'LGB_positive_seed_ratio': float(lgb.loc[label, 'positive_seed_ratio']),
            'XGB_positive_seed_ratio': float(xgb.loc[label, 'positive_seed_ratio']),
            'both_models_positive': bool(lgb.loc[label, 'delta_mean_AUC'] > 0 and xgb.loc[label, 'delta_mean_AUC'] > 0),
            'conservative_score': float((lgb.loc[label, 'conservative_score'] + xgb.loc[label, 'conservative_score']) / 2),
        })
    return pd.DataFrame(rows).sort_values(['both_models_positive', 'conservative_score', 'delta_mean_AUC'], ascending=False)


def main():
    print('FE-8 full run start', flush=True)
    df_raw = read_creditcard_csv(DATA_PATH)
    v_cols = [c for c in df_raw.columns if c.startswith('V')]
    base_features = v_cols + ['Amount', 'Time']
    fe6_extra = [c for c in json.loads(FE6_CONFIG_PATH.read_text())['MODEL_FEATURES_V2'] if c not in base_features]

    df = build_eda_features(df_raw)
    df = bind_cv_data(df)
    df = build_one_euro_cross_features(df)
    df, fe8_feats = build_fe8_contribution_features(df)
    print('IF OOF...', flush=True)
    df['if_oof_score'] = oof_if_anomaly_score(df, v_cols)

    specs = build_fe8_specs(fe6_extra, fe8_feats)
    export_catalog(specs)
    print(f'combos={len(specs)} total_runs={len(specs)*len(MODELS)*len(RUN_SEEDS)}', flush=True)

    raw = run_matrix(specs, df, base_features)
    seed_df, summary = summarize(raw)
    dual = dual_summary_fn(summary)
    raw.to_csv(RAW_CSV_PATH, index=False, encoding='utf-8-sig')
    summary.to_csv(SUMMARY_CSV_PATH, index=False, encoding='utf-8-sig')
    dual.to_csv(DUAL_CSV_PATH, index=False, encoding='utf-8-sig')

    eligible = dual[(dual['both_models_positive']) & (dual['LGB_positive_seed_ratio'] >= 2/3) & (dual['XGB_positive_seed_ratio'] >= 2/3) & (dual['combo_label'] != BASELINE_LABEL)]
    if eligible.empty:
        winner, selected = BASELINE_LABEL, fe6_extra
    else:
        winner = eligible.iloc[0]['combo_label']
        selected = next(r['extra_cols'] for r in specs if r['label'] == winner)

    OUTPUT_PATH.write_text(json.dumps({
        'MODEL_FEATURES_V3': base_features + [c for c in selected if c not in base_features],
        'winner_combo': winner,
        'selected_extra': selected,
        'combo_count': len(specs),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE winner=', winner, flush=True)


if __name__ == '__main__':
    main()
