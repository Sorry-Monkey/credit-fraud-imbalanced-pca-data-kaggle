"""Purged CV + tuned-classifier helpers for winner feature ablation tests."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from purgedcv import WalkForwardSplit
from purgedcv.diagnostics import assert_no_temporal_leakage
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

MODELS = ["LightGBM", "XGBoost"]
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42
CV_EMBARGO = pd.Timedelta(hours=2)
CV_PURGE_HORIZON = pd.Timedelta(0)
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1500
ES_FRAC = 0.20
ES_MIN_FRAUD = 5
ES_MAX_FRAC = 0.35
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
F1_BEST_COL = "F1@best"
F1_BASE_COL = "F1@BASE_t*"
F1_COMBO_COL = "F1@combo_t*"
THR_BEST_COL = "t*@combo"
PREC_BEST_COL = "Precision@best"
REC_BEST_COL = "Recall@best"
ABLATION_BASELINE_LABEL = "0. 基线（仅 BASE）"

# 主 notebook 已占用种子（实验须零重叠）
SECTION7_SCREEN_SEEDS = [42, 88, 2025]
SECTION7_CV_SEED = SECTION7_SCREEN_SEEDS[0]  # 兼容旧引用
SECTION8_CONFIRM_SEEDS = [7, 13, 123, 256, 3141]
SECTION10_SCREEN_SEEDS = [42, 77, 2026]
EXPERIMENT_SEEDS_DEFAULT = [17, 29, 101, 271, 503, 2027]
# 主 notebook 已占用；EXPERIMENT_SEEDS_DEFAULT 专供本实验 exp_extra，不得列入
USED_SEEDS = sorted({
    *SECTION7_SCREEN_SEEDS,
    *SECTION8_CONFIRM_SEEDS,
    *SECTION10_SCREEN_SEEDS,
})
OOF_CAL_FRAC = 0.25
OOF_CAL_MIN_FRAUD = 50
OOF_CAL_MAX_FRAC = 0.40
WEIGHT_SCHEMES = {
    "balanced": None, "spw_sqrt": "sqrt", "spw_0.5x": 0.5, "spw_2x": 2.0, "no_weight": 0.0,
}

IF_RANDOM_STATE = 42
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 0.5
IF_CONTAMINATION = "auto"
IF_MAX_NORMAL_SAMPLES = 50_000

FE8_WINNER = ["abs_v14_minus_v12", "v14_x_log1p_amount", "v4_minus_v14"]
EDA_WINNER = ["log1p_amount", "hours_since_start", "is_micro_testing", "is_amount_1_30", "is_amount_75_110"]
EDA_NO_75 = ["log1p_amount", "hours_since_start", "is_micro_testing", "is_amount_1_30"]

_CV_BOUND_DATA = None


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "input" / "creditcard.csv").exists():
            return candidate
    raise FileNotFoundError("cannot find input/creditcard.csv")


def read_creditcard_csv(path: Path) -> pd.DataFrame:
    for kwargs in (
        {"encoding": "utf-8"},
        {"encoding": "utf-8", "encoding_errors": "replace"},
        {"encoding": "latin-1"},
    ):
        try:
            return pd.read_csv(path, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "failed to decode creditcard.csv")


def build_eda_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["log1p_amount"] = np.log1p(out["Amount"])
    out["hours_since_start"] = (out["Time"] // 3600).astype(int)
    out["is_micro_testing"] = out["Amount"] < 1
    out["is_one_euro"] = out["Amount"] == 1.0
    out["is_amount_1_30"] = (out["Amount"] > 1) & (out["Amount"] <= 30)
    out["is_amount_75_110"] = (out["Amount"] >= 75) & (out["Amount"] <= 110)
    return out


def sort_by_time(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values("Time", kind="mergesort").reset_index(drop=True)


def bind_cv_data(data: pd.DataFrame) -> pd.DataFrame:
    global _CV_BOUND_DATA
    out = sort_by_time(data)
    _CV_BOUND_DATA = out
    return out


def build_cv_timestamps(data: pd.DataFrame):
    t = pd.to_timedelta(data["Time"].astype(float), unit="s")
    return t.copy(), t.copy()


def iter_purged_cv_folds(n_samples=None, n_splits=CV_N_SPLITS, data=None):
    bound = data if data is not None else _CV_BOUND_DATA
    if bound is None:
        raise RuntimeError("call bind_cv_data() first")
    n = n_samples if n_samples is not None else len(bound)
    pred, evalu = build_cv_timestamps(bound)
    test_size = max(1, n // (n_splits + 1))
    cv = WalkForwardSplit(
        n_splits=n_splits, test_size=test_size, window="expanding",
        prediction_times=pred, evaluation_times=evalu,
        purge_horizon=CV_PURGE_HORIZON, embargo=CV_EMBARGO,
    )
    for tr_idx, va_idx in cv.split(np.arange(n)):
        assert_no_temporal_leakage(
            tr_idx, va_idx, prediction_times=pred, evaluation_times=evalu,
            purge_horizon=CV_PURGE_HORIZON,
        )
        yield tr_idx, va_idx


def _temporal_es_mask(y, es_frac=ES_FRAC, min_fraud_es=ES_MIN_FRAUD, max_frac=ES_MAX_FRAC):
    """ES 段：各类按时间序**后** es_frac 进早停验证；欺诈不足则逐步扩大 es 上限 max_frac。"""
    y = np.asarray(y).astype(int)
    n = len(y)
    fraud_idx = np.flatnonzero(y == 1)
    norm_idx = np.flatnonzero(y == 0)
    if len(fraud_idx) < 2:
        n_es = max(1, min(n - 1, int(n * es_frac)))
        mask = np.zeros(n, dtype=bool)
        mask[n - n_es:] = True
        return mask, int(n_es)
    frac = es_frac
    while frac <= max_frac + 1e-9:
        n_fraud_es = max(min_fraud_es, max(1, int(len(fraud_idx) * frac)))
        n_fraud_es = min(n_fraud_es, len(fraud_idx) - 1)
        n_norm_es = max(1, int(len(norm_idx) * frac))
        n_norm_es = min(n_norm_es, len(norm_idx) - 1)
        mask = np.zeros(n, dtype=bool)
        mask[fraud_idx[-n_fraud_es:]] = True
        mask[norm_idx[-n_norm_es:]] = True
        if int(y[mask].sum()) >= min(min_fraud_es, len(fraud_idx) - 1):
            return mask, int(mask.sum())
        frac += 0.05
    n_fraud_es = min(max(min_fraud_es, 1), len(fraud_idx) - 1)
    n_norm_es = max(1, min(len(norm_idx) * es_frac))
    n_norm_es = min(n_norm_es, len(norm_idx) - 1)
    mask = np.zeros(n, dtype=bool)
    mask[fraud_idx[-n_fraud_es:]] = True
    mask[norm_idx[-n_norm_es:]] = True
    return mask, int(mask.sum())


def split_early_stop_set(X_tr, y_tr, es_frac=ES_FRAC, random_state=42, min_fraud_es=ES_MIN_FRAUD):
    """严格按 Time 顺序：训练折末尾作早停验证集（random_state 保留 API 兼容，不参与切分）。"""
    del random_state  # 与主 notebook §2 一致：时间序切分，不用随机分层
    y_arr = np.asarray(y_tr)
    es_mask, _ = _temporal_es_mask(y_arr, es_frac=es_frac, min_fraud_es=min_fraud_es)
    fit_mask = ~es_mask
    if not fit_mask.any() or not es_mask.any():
        raise ValueError("早停切分失败：fit 或 ES 段为空，请检查 es_frac / ES_MIN_FRAUD")
    if hasattr(X_tr, "iloc"):
        return X_tr.iloc[fit_mask], X_tr.iloc[es_mask], y_tr.iloc[fit_mask], y_tr.iloc[es_mask]
    return X_tr[fit_mask], X_tr[es_mask], y_arr[fit_mask], y_arr[es_mask]


def apply_weight_scheme(defaults, model_name, spw, weight_scheme="balanced"):
    vk = WEIGHT_SCHEMES[weight_scheme]
    defaults.pop("scale_pos_weight", None)
    defaults.pop("class_weight", None)
    if model_name == "LightGBM":
        if vk is None:
            defaults["class_weight"] = "balanced"
        elif vk == 0.0:
            pass
        elif vk == "sqrt":
            defaults["class_weight"] = {0: 1.0, 1: float(np.sqrt(spw))}
        else:
            defaults["class_weight"] = {0: 1.0, 1: spw * vk}
    else:
        if vk is None:
            defaults["scale_pos_weight"] = spw
        elif vk == 0.0:
            defaults["scale_pos_weight"] = 1.0
        elif vk == "sqrt":
            defaults["scale_pos_weight"] = float(np.sqrt(spw))
        else:
            defaults["scale_pos_weight"] = spw * vk
    return defaults


def make_classifier(model_name, y_train, params=None, random_state=CV_RANDOM_STATE):
    params = dict(params or {})
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    weight_scheme = params.pop("weight_scheme", "balanced")
    if model_name == "LightGBM":
        defaults = dict(
            n_estimators=MAX_BOOST_ROUNDS, learning_rate=0.05, max_depth=6, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=random_state, verbose=-1, n_jobs=-1,
        )
        defaults.update(params)
        defaults = apply_weight_scheme(defaults, model_name, spw, weight_scheme)
        return lgb.LGBMClassifier(**defaults)
    defaults = dict(
        n_estimators=MAX_BOOST_ROUNDS, learning_rate=0.05, max_depth=6,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=random_state, eval_metric="logloss", verbosity=0, n_jobs=-1,
    )
    defaults.update(params)
    defaults["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    defaults = apply_weight_scheme(defaults, model_name, spw, weight_scheme)
    return xgb.XGBClassifier(**defaults)


def fit_classifier(clf, model_name, X_tr, y_tr, X_es=None, y_es=None):
    if X_es is None:
        clf.fit(X_tr, y_tr)
        return clf
    if model_name == "LightGBM":
        clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    else:
        clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return clf


def best_f1_threshold(y_true, proba):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    mask = np.isfinite(proba)
    y_eval, p_eval = y_true[mask], proba[mask]
    thr, f1 = _threshold_candidates(y_eval, p_eval)
    if len(thr) == 0:
        return DEFAULT_CLASSIFICATION_THRESHOLD, 0.0
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def _threshold_candidates(y_eval, p_eval):
    prec, rec, thr = precision_recall_curve(y_eval, p_eval)
    if len(thr) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return thr, f1


# --- F1@combo 稳定性协议（实验 Arm D；主 notebook 默认仍为 combo_argmax） ---
F1_PLATEAU_EPS = 0.02
F1_ANCHOR_BAND = 0.15
F1_CAL_LARGE_FRAC = 0.35
F1_CAL_LARGE_MIN_FRAUD = 80

F1_PROTOCOL_COLUMNS = {
    "combo_argmax": F1_COMBO_COL,
    "base_shared": F1_BASE_COL,
    "combo_plateau": "F1@combo_plateau",
    "combo_anchor_base": "F1@combo_anchor",
    "combo_cal_large": "F1@combo_cal_large",
}

F1_PROTOCOL_DESCRIPTIONS = {
    "combo_argmax": "现状：combo 在 cal 上 argmax F1（易抖）",
    "base_shared": "共用 BASE cal t*（横向可比，最稳）",
    "combo_plateau": f"cal 上 F1≥max−{F1_PLATEAU_EPS} 的 plateau，取最靠近 BASE t* 的阈",
    "combo_anchor_base": f"仅在 BASE t*±{F1_ANCHOR_BAND} 内搜 combo 最优阈",
    "combo_cal_large": f"更大 cal（frac={F1_CAL_LARGE_FRAC}, fraud≥{F1_CAL_LARGE_MIN_FRAUD}）再 argmax",
}


def best_f1_threshold_plateau(y_true, proba, eps=F1_PLATEAU_EPS, prefer=None):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    mask = np.isfinite(proba)
    y_eval, p_eval = y_true[mask], proba[mask]
    thr, f1 = _threshold_candidates(y_eval, p_eval)
    if len(thr) == 0:
        return DEFAULT_CLASSIFICATION_THRESHOLD, 0.0
    f1_max = float(np.nanmax(f1))
    plateau = thr[f1 >= f1_max - eps]
    if len(plateau) == 0:
        i = int(np.nanargmax(f1))
        return float(thr[i]), float(f1[i])
    if prefer is not None:
        j = int(np.argmin(np.abs(plateau - prefer)))
        return float(plateau[j]), f1_max
    return float(np.median(plateau)), f1_max


def best_f1_threshold_in_band(y_true, proba, low, high):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    mask = np.isfinite(proba)
    y_eval, p_eval = y_true[mask], proba[mask]
    thr, f1 = _threshold_candidates(y_eval, p_eval)
    if len(thr) == 0:
        return DEFAULT_CLASSIFICATION_THRESHOLD, 0.0
    band = (thr >= low) & (thr <= high)
    if not band.any():
        mid = (low + high) / 2.0
        j = int(np.argmin(np.abs(thr - mid)))
        return float(thr[j]), float(f1[j])
    idx = np.flatnonzero(band)
    j = int(idx[np.nanargmax(f1[band])])
    return float(thr[j]), float(f1[j])


def metrics_f1_protocol(
    y_true,
    proba,
    protocol: str,
    base_threshold: float | None = None,
    cal_frac: float = OOF_CAL_FRAC,
    min_fraud_cal: int = OOF_CAL_MIN_FRAUD,
):
    """在固定 OOF 上按协议重算 F1（无需重训）。"""
    if protocol == "combo_cal_large":
        cal_frac = F1_CAL_LARGE_FRAC
        min_fraud_cal = F1_CAL_LARGE_MIN_FRAUD
    y_cal, p_cal, y_eval, p_eval, n_cal = split_oof_cal_eval(
        y_true, proba, cal_frac=cal_frac, min_fraud_cal=min_fraud_cal,
    )
    ref = base_threshold if base_threshold is not None else DEFAULT_CLASSIFICATION_THRESHOLD
    if protocol == "base_shared":
        thr = ref
    elif protocol == "combo_argmax":
        thr, _ = best_f1_threshold(y_cal, p_cal)
    elif protocol == "combo_plateau":
        thr, _ = best_f1_threshold_plateau(y_cal, p_cal, prefer=ref)
    elif protocol == "combo_anchor_base":
        thr, _ = best_f1_threshold_in_band(y_cal, p_cal, ref - F1_ANCHOR_BAND, ref + F1_ANCHOR_BAND)
    elif protocol == "combo_cal_large":
        thr, _ = best_f1_threshold(y_cal, p_cal)
    else:
        raise ValueError(f"未知 F1 protocol: {protocol!r}")
    m = metrics_at_threshold(y_eval, p_eval, thr)
    col = F1_PROTOCOL_COLUMNS[protocol]
    return {
        col: m[F1_COMBO_COL],
        f"thr@{protocol}": float(thr),
        "n_cal": int(n_cal),
        "n_fraud_cal": int(y_cal.sum()),
    }


def _compute_confusion_metrics(y_eval, pred):
    tn, fp, fn, tp = confusion_matrix(y_eval, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
    }


def metrics_at_threshold(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    mask = np.isfinite(proba)
    pred = proba[mask] >= threshold
    y_eval = y_true[mask]
    m = _compute_confusion_metrics(y_eval, pred)
    return {
        F1_COMBO_COL: m["f1"], PREC_BEST_COL: m["precision"], REC_BEST_COL: m["recall"],
        "FP": m["fp"], "FN": m["fn"], "n_eval": int(mask.sum()), "n_excluded": int((~mask).sum()),
    }


def _oof_cal_mask(y, cal_frac=OOF_CAL_FRAC, min_fraud_cal=OOF_CAL_MIN_FRAUD, max_frac=OOF_CAL_MAX_FRAC):
    y = np.asarray(y).astype(int)
    n = len(y)
    fraud_idx = np.flatnonzero(y == 1)
    norm_idx = np.flatnonzero(y == 0)
    if len(fraud_idx) < 2:
        n_cal = max(1, min(n - 1, int(n * cal_frac)))
        mask = np.zeros(n, dtype=bool)
        mask[:n_cal] = True
        return mask, int(n_cal)
    frac = cal_frac
    while frac <= max_frac + 1e-9:
        n_fraud_cal = max(min_fraud_cal, max(1, int(len(fraud_idx) * frac)))
        n_fraud_cal = min(n_fraud_cal, len(fraud_idx) - 1)
        n_norm_cal = max(1, int(len(norm_idx) * frac))
        n_norm_cal = min(n_norm_cal, len(norm_idx) - 1)
        mask = np.zeros(n, dtype=bool)
        mask[fraud_idx[:n_fraud_cal]] = True
        mask[norm_idx[:n_norm_cal]] = True
        if int(y[mask].sum()) >= min(min_fraud_cal, len(fraud_idx) - 1):
            return mask, int(mask.sum())
        frac += 0.05
    n_fraud_cal = min(max(min_fraud_cal, 1), len(fraud_idx) - 1)
    n_norm_cal = max(1, min(len(norm_idx) * cal_frac))
    n_norm_cal = min(n_norm_cal, len(norm_idx) - 1)
    mask = np.zeros(n, dtype=bool)
    mask[fraud_idx[:n_fraud_cal]] = True
    mask[norm_idx[:n_norm_cal]] = True
    return mask, int(mask.sum())


def split_oof_cal_eval(y_true, proba, cal_frac=OOF_CAL_FRAC, min_fraud_cal=OOF_CAL_MIN_FRAUD):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    mask = np.isfinite(proba)
    y, p = y_true[mask], proba[mask]
    cal_mask, n_cal = _oof_cal_mask(y, cal_frac=cal_frac, min_fraud_cal=min_fraud_cal)
    return y[cal_mask], p[cal_mask], y[~cal_mask], p[~cal_mask], n_cal


def metrics_at_threshold_honest(
    y_true, proba, cal_frac=OOF_CAL_FRAC, threshold=None, min_fraud_cal=OOF_CAL_MIN_FRAUD,
):
    y_cal, p_cal, y_eval, p_eval, n_cal = split_oof_cal_eval(y_true, proba, cal_frac, min_fraud_cal)
    if threshold is None:
        threshold, _ = best_f1_threshold(y_cal, p_cal)
    m = metrics_at_threshold(y_eval, p_eval, threshold)
    m["best_threshold"] = float(threshold)
    m["n_cal"] = int(n_cal)
    m["n_fraud_cal"] = int(y_cal.sum())
    m["n_fraud_eval"] = int(y_eval.sum())
    m["n_excluded"] = int((~np.isfinite(proba)).sum())
    return m


def cross_val_eval(
    model_name,
    data,
    feature_cols,
    params=None,
    n_splits=CV_N_SPLITS,
    random_state=CV_RANDOM_STATE,
    threshold=None,
    return_oof=False,
):
    X, y = data[feature_cols], data["Class"]
    oof = np.full(len(y), np.nan, dtype=float)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(iter_purged_cv_folds(len(X), n_splits=n_splits), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        X_fit, X_es, y_fit, y_es = split_early_stop_set(X_tr, y_tr, random_state=random_state + fold)
        clf = make_classifier(model_name, y_fit, params=params, random_state=random_state + fold)
        fit_classifier(clf, model_name, X_fit, y_fit, X_es, y_es)
        proba_va = clf.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba_va
        fold_scores.append(float(average_precision_score(y_va, proba_va)))
    arr = np.array(fold_scores, dtype=float)
    cls_combo = metrics_at_threshold_honest(y, oof)
    if threshold is None:
        cls_base = cls_combo
        thr = cls_combo["best_threshold"]
    else:
        cls_base = metrics_at_threshold_honest(y, oof, threshold=threshold)
        thr = threshold
    out = {
        "model": model_name, "n_features": len(feature_cols), "seed": random_state,
        "AUC-PR_mean": float(arr.mean()), "AUC-PR_std": float(arr.std(ddof=0)),
        "AUC-PR_min": float(arr.min()), "best_threshold": thr,
        F1_BASE_COL: cls_base[F1_COMBO_COL],
        F1_COMBO_COL: cls_combo[F1_COMBO_COL],
        THR_BEST_COL: cls_combo["best_threshold"],
        PREC_BEST_COL: cls_base[PREC_BEST_COL],
        REC_BEST_COL: cls_base[REC_BEST_COL],
        "FP": cls_base["FP"], "FN": cls_base["FN"],
        "n_eval": cls_base["n_eval"], "n_excluded": cls_base.get("n_excluded", 0),
    }
    if return_oof:
        out["oof"] = oof
        out["y"] = y.values
    return out


def build_fe8_winner_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["abs_v14_minus_v12"] = (out["V14"] - out["V12"]).abs()
    out["v14_x_log1p_amount"] = out["V14"] * out["log1p_amount"]
    out["v4_minus_v14"] = out["V4"] - out["V14"]
    return out


def oof_if_anomaly_score(data, feature_cols, random_state=IF_RANDOM_STATE):
    X = data[feature_cols].values.astype(np.float64)
    y = data["Class"].values
    oof = np.full(len(y), np.nan, dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(iter_purged_cv_folds(data=data, n_splits=CV_N_SPLITS), start=1):
        normal_tr = tr_idx[y[tr_idx] == 0]
        if len(normal_tr) > IF_MAX_NORMAL_SAMPLES:
            rng = np.random.default_rng(random_state + fold)
            normal_tr = rng.choice(normal_tr, size=IF_MAX_NORMAL_SAMPLES, replace=False)
        scaler = StandardScaler()
        X_normal = scaler.fit_transform(X[normal_tr])
        X_valid = scaler.transform(X[va_idx])
        iforest = IsolationForest(
            n_estimators=IF_N_ESTIMATORS, max_samples=IF_MAX_SAMPLES,
            contamination=IF_CONTAMINATION, random_state=random_state + fold, n_jobs=-1,
        )
        iforest.fit(X_normal)
        oof[va_idx] = -iforest.score_samples(X_valid)
        print(f"  IF fold {fold}/{CV_N_SPLITS} done, normal_train={len(normal_tr):,}")
    return oof


def load_or_compute_if_score(df: pd.DataFrame, v_cols: list[str], cache_path: Path) -> pd.Series:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"load if_oof_score cache -> {cache_path}")
        return pd.Series(np.load(cache_path), index=df.index, name="if_oof_score")
    print("compute if_oof_score (purged OOF IF)...")
    score = oof_if_anomaly_score(df, v_cols, random_state=IF_RANDOM_STATE)
    np.save(cache_path, score)
    print(f"cached -> {cache_path}")
    return pd.Series(score, index=df.index, name="if_oof_score")


def build_winner_dataframe(project_root: Path | None = None, cache_if: bool = True):
    root = project_root or find_project_root()
    data_path = root / "input" / "creditcard.csv"
    cache_path = root / "src" / "output" / "final_report" / "cache" / "if_oof_score.npy"
    df_raw = read_creditcard_csv(data_path)
    v_cols = [c for c in df_raw.columns if c.startswith("V")]
    base_features = v_cols + ["Amount", "Time"]
    df = bind_cv_data(build_eda_features(df_raw))
    if cache_if:
        df["if_oof_score"] = load_or_compute_if_score(df, v_cols, cache_path)
    else:
        df["if_oof_score"] = oof_if_anomaly_score(df, v_cols)
    df["one_euro_V4"] = df["is_one_euro"].astype(float) * df["V4"]
    df["one_euro_V14"] = df["is_one_euro"].astype(float) * df["V14"]
    df = build_fe8_winner_features(df)
    return df, base_features


def winner_feature_cols(base_features: list[str], eda_cols: list[str]) -> list[str]:
    extra = ["if_oof_score", *eda_cols, "one_euro_V4", "one_euro_V14", *FE8_WINNER]
    return base_features + extra


def load_tune_params(project_root: Path | None = None) -> dict:
    root = project_root or find_project_root()
    tune_path = root / "src" / "output" / "final_report" / "12_tune" / "tune_results.json"
    return json.loads(tune_path.read_text(encoding="utf-8-sig"))


def load_confirm_seeds(project_root: Path | None = None) -> list[int]:
    root = project_root or find_project_root()
    p = root / "src" / "output" / "final_report" / "11_confirm" / "confirm_winner.json"
    return json.loads(p.read_text(encoding="utf-8-sig"))["confirm_seeds"]


def run_ablation_matrix(df, base_features, variants, seeds, tune, models=None):
    models = models or MODELS
    rows = []
    total = len(variants) * len(models) * len(seeds)
    step = 0
    for label, eda_cols in variants.items():
        feat_cols = winner_feature_cols(base_features, eda_cols)
        for model in models:
            params = {k: v for k, v in tune[model].items() if k != "feature_group"}
            for seed in seeds:
                step += 1
                print(f"[{step}/{total}] {label} | {model} | seed={seed}")
                res = cross_val_eval(model, df, feat_cols, params=params, random_state=seed)
                rows.append({"variant": label, "n_extra_eda": len(eda_cols), **res})
    return pd.DataFrame(rows)


def summarize_ablation(raw: pd.DataFrame) -> pd.DataFrame:
    agg = raw.groupby(["variant", "model"]).agg(
        AUC_mean=("AUC-PR_mean", "mean"),
        AUC_std=("AUC-PR_mean", "std"),
        F1_mean=(F1_BEST_COL, "mean"),
        F1_std=(F1_BEST_COL, "std"),
        FP_mean=("FP", "mean"),
        FN_mean=("FN", "mean"),
        n_runs=("seed", "count"),
    ).reset_index()
    return agg.sort_values(["variant", "model"]).reset_index(drop=True)


# --- EDA base × FE8 top-N comparison (experiments) ---

EDA_TAG_COLS = {
    "log1p": ["log1p_amount"],
    "hours": ["hours_since_start"],
    "one_euro": ["is_one_euro"],
    "micro": ["is_micro_testing"],
    "bands": ["is_amount_1_30", "is_amount_75_110"],
}

FAMILY_A_COLS = {
    "none": [],
    "a1": ["one_euro_V4"],
    "atop2": ["one_euro_V4", "one_euro_V14"],
}

FE8_CONTRIBUTION_ATOMS = [
    "abs_v14_minus_v10",
    "v14_x_log1p_amount",
    "v14_x_v10",
    "v14_x_v4",
    "abs_v14_minus_v12",
    "v14_x_v12",
    "abs_v14_minus_v4",
]

EDA_BASE_SPECS = {
    "IF+Ed[bands]+A_top2(V14+V4)": {
        "eda_tags": ["bands"],
        "family_a": "atop2",
    },
    "IF+Ed[hours+one_euro+micro]+A1": {
        "eda_tags": ["hours", "one_euro", "micro"],
        "family_a": "a1",
    },
    "IF+Ed[hours+bands]+A1": {
        "eda_tags": ["hours", "bands"],
        "family_a": "a1",
    },
    "IF+Ed[log1p+hours+micro+bands]+A_top2(V14+V4)": {
        "eda_tags": ["log1p", "hours", "micro", "bands"],
        "family_a": "atop2",
    },
    "IF+Ed[log1p+hours+one_euro+micro+bands]+A_top2(V14+V4)": {
        "eda_tags": ["log1p", "hours", "one_euro", "micro", "bands"],
        "family_a": "atop2",
    },
    "IF+Ed[log1p+hours+one_euro+bands]+A_top2(V4+V14)": {
        "eda_tags": ["log1p", "hours", "one_euro", "bands"],
        "family_a": "atop2",
    },
}


def eda_tags_to_cols(tags: list[str]) -> list[str]:
    cols: list[str] = []
    for tag in tags:
        cols.extend(EDA_TAG_COLS[tag])
    return cols


def fe8_subset_label(fe8_cols: list[str]) -> str:
    if not fe8_cols:
        return "no_FE8"
    return "FE8[" + "+".join(fe8_cols) + "]"


def build_fe8_contribution_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["abs_v14_minus_v10"] = (out["V14"] - out["V10"]).abs()
    out["v14_x_log1p_amount"] = out["V14"] * out["log1p_amount"]
    out["v14_x_v10"] = out["V14"] * out["V10"]
    out["v14_x_v4"] = out["V14"] * out["V4"]
    out["abs_v14_minus_v12"] = (out["V14"] - out["V12"]).abs()
    out["v14_x_v12"] = out["V14"] * out["V12"]
    out["abs_v14_minus_v4"] = (out["V14"] - out["V4"]).abs()
    return out


def build_experiment_dataframe(project_root: Path | None = None, cache_if: bool = True):
    root = project_root or find_project_root()
    data_path = root / "input" / "creditcard.csv"
    cache_path = root / "src" / "output" / "final_report" / "cache" / "if_oof_score.npy"
    df_raw = read_creditcard_csv(data_path)
    v_cols = [c for c in df_raw.columns if c.startswith("V")]
    base_features = v_cols + ["Amount", "Time"]
    df = bind_cv_data(build_eda_features(df_raw))
    if cache_if:
        df["if_oof_score"] = load_or_compute_if_score(df, v_cols, cache_path)
    else:
        df["if_oof_score"] = oof_if_anomaly_score(df, v_cols)
    df["one_euro_V4"] = df["is_one_euro"].astype(float) * df["V4"]
    df["one_euro_V14"] = df["is_one_euro"].astype(float) * df["V14"]
    df = build_fe8_contribution_features(df)
    return df, base_features


def combo_feature_cols(
    base_features: list[str],
    eda_cols: list[str],
    family_a: str = "none",
    fe8_cols: list[str] | None = None,
) -> list[str]:
    fe8_cols = fe8_cols or []
    extra = ["if_oof_score", *eda_cols, *FAMILY_A_COLS[family_a], *fe8_cols]
    seen: set[str] = set()
    ordered: list[str] = []
    for col in base_features + extra:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def load_fe8_top_subsets(project_root: Path | None = None, top_n: int = 10) -> list[list[str]]:
    root = project_root or find_project_root()
    path = root / "src" / "output" / "final_report" / "10_fe8_stage1" / "fe8_stage1_top15.json"
    subsets = json.loads(path.read_text(encoding="utf-8-sig"))
    return subsets[:top_n]


def build_eda_fe8_variants(
    eda_bases: dict | None = None,
    fe8_subsets: list[list[str]] | None = None,
    include_no_fe8: bool = True,
) -> dict[str, dict]:
    eda_bases = eda_bases or EDA_BASE_SPECS
    fe8_subsets = list(fe8_subsets or [])
    if include_no_fe8:
        fe8_subsets = [[]] + fe8_subsets
    variants: dict[str, dict] = {}
    for base_label, spec in eda_bases.items():
        eda_cols = eda_tags_to_cols(spec["eda_tags"])
        family_a = spec["family_a"]
        for fe8_cols in fe8_subsets:
            fe8_name = fe8_subset_label(fe8_cols)
            combo_label = base_label if not fe8_cols else f"{base_label}+{fe8_name}"
            variants[combo_label] = {
                "eda_base": base_label,
                "fe8_label": fe8_name,
                "eda_cols": eda_cols,
                "family_a": family_a,
                "fe8_cols": fe8_cols,
            }
    return variants


def run_eda_fe8_matrix(
    df,
    base_features,
    variants: dict[str, dict],
    seeds,
    tune,
    models=None,
):
    models = models or MODELS
    rows = []
    total = len(variants) * len(models) * len(seeds)
    step = 0
    for combo_label, spec in variants.items():
        feat_cols = combo_feature_cols(
            base_features,
            spec["eda_cols"],
            family_a=spec["family_a"],
            fe8_cols=spec["fe8_cols"],
        )
        for model in models:
            params = {k: v for k, v in tune[model].items() if k != "feature_group"}
            for seed in seeds:
                step += 1
                print(f"[{step}/{total}] {combo_label} | {model} | seed={seed}")
                res = cross_val_eval(model, df, feat_cols, params=params, random_state=seed)
                rows.append(
                    {
                        "combo_label": combo_label,
                        "eda_base": spec["eda_base"],
                        "fe8_label": spec["fe8_label"],
                        "n_features": len(feat_cols),
                        "n_fe8": len(spec["fe8_cols"]),
                        **res,
                    }
                )
    return pd.DataFrame(rows)


def summarize_eda_fe8(raw: pd.DataFrame) -> pd.DataFrame:
    agg = raw.groupby(["combo_label", "eda_base", "fe8_label", "model"]).agg(
        AUC_mean=("AUC-PR_mean", "mean"),
        AUC_std=("AUC-PR_mean", "std"),
        F1_combo_mean=(F1_COMBO_COL, "mean"),
        F1_combo_std=(F1_COMBO_COL, "std"),
        F1_mean=(F1_BEST_COL, "mean"),
        F1_std=(F1_BEST_COL, "std"),
        FP_mean=("FP", "mean"),
        FN_mean=("FN", "mean"),
        n_features=("n_features", "first"),
        n_runs=("seed", "count"),
    ).reset_index()
    return agg.sort_values(["eda_base", "fe8_label", "model"]).reset_index(drop=True)


def _model_pivot(summary: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = summary[summary["model"] == model].set_index("combo_label")
    return sub


def _no_fe8_baselines(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    base = summary[summary["fe8_label"] == "no_FE8"]
    auc_lgb = base[base["model"] == "LightGBM"].set_index("eda_base")["AUC_mean"]
    auc_xgb = base[base["model"] == "XGBoost"].set_index("eda_base")["AUC_mean"]
    f1_lgb = base[base["model"] == "LightGBM"].set_index("eda_base")["F1_combo_mean"]
    f1_xgb = base[base["model"] == "XGBoost"].set_index("eda_base")["F1_combo_mean"]
    return auc_lgb, auc_xgb, f1_lgb, f1_xgb


def build_no_fe8_base_dual(summary: pd.DataFrame) -> pd.DataFrame:
    """六个 EDA 基底 no_FE8 横向对比（双模型 + 均值）。"""
    base = summary[summary["fe8_label"] == "no_FE8"].copy()
    lgb = base[base["model"] == "LightGBM"].set_index("eda_base")
    xgb = base[base["model"] == "XGBoost"].set_index("eda_base")
    rows = []
    for eda_base in lgb.index.intersection(xgb.index):
        rows.append({
            "eda_base": eda_base,
            "AUC_LGB": float(lgb.loc[eda_base, "AUC_mean"]),
            "AUC_XGB": float(xgb.loc[eda_base, "AUC_mean"]),
            "AUC_mean": float(np.mean([lgb.loc[eda_base, "AUC_mean"], xgb.loc[eda_base, "AUC_mean"]])),
            F1_COMBO_COL + "_LGB": float(lgb.loc[eda_base, "F1_combo_mean"]),
            F1_COMBO_COL + "_XGB": float(xgb.loc[eda_base, "F1_combo_mean"]),
            "F1_combo_mean": float(np.mean([lgb.loc[eda_base, "F1_combo_mean"], xgb.loc[eda_base, "F1_combo_mean"]])),
            "FP_mean": float(np.mean([lgb.loc[eda_base, "FP_mean"], xgb.loc[eda_base, "FP_mean"]])),
            "FN_mean": float(np.mean([lgb.loc[eda_base, "FN_mean"], xgb.loc[eda_base, "FN_mean"]])),
            "n_features": int(lgb.loc[eda_base, "n_features"]),
        })
    return pd.DataFrame(rows).sort_values("AUC_mean", ascending=False).reset_index(drop=True)


def build_eda_fe8_combo_dual(summary: pd.DataFrame) -> pd.DataFrame:
    """全 combo 双模型合并表（§7 build_combo_dual 样式；Δ 相对同 eda_base 的 no_FE8）。"""
    lgb = _model_pivot(summary, "LightGBM")
    xgb = _model_pivot(summary, "XGBoost")
    auc_lgb, auc_xgb, f1_lgb, f1_xgb = _no_fe8_baselines(summary)
    rows = []
    for combo in lgb.index.intersection(xgb.index):
        eda_base = str(lgb.loc[combo, "eda_base"])
        dl = float(lgb.loc[combo, "AUC_mean"] - auc_lgb.loc[eda_base])
        dx = float(xgb.loc[combo, "AUC_mean"] - auc_xgb.loc[eda_base])
        f1_bl = float(f1_lgb.loc[eda_base])
        f1_bx = float(f1_xgb.loc[eda_base])
        rows.append({
            "combo_label": combo,
            "eda_base": eda_base,
            "fe8_label": lgb.loc[combo, "fe8_label"],
            "AUC_LGB": float(lgb.loc[combo, "AUC_mean"]),
            "AUC_XGB": float(xgb.loc[combo, "AUC_mean"]),
            "F1_combo_LGB": float(lgb.loc[combo, "F1_combo_mean"]),
            "F1_combo_XGB": float(xgb.loc[combo, "F1_combo_mean"]),
            "Δ_LGB_AUC": dl,
            "Δ_XGB_AUC": dx,
            "Δ_mean_AUC": (dl + dx) / 2,
            "Δ_mean_F1_combo": (
                (float(lgb.loc[combo, "F1_combo_mean"]) - f1_bl)
                + (float(xgb.loc[combo, "F1_combo_mean"]) - f1_bx)
            ) / 2,
            "双模型Δ_AUC均为正": dl > 0 and dx > 0,
            "符号翻转": (dl > 0) != (dx > 0),
            "n_features": int(lgb.loc[combo, "n_features"]),
        })
    return pd.DataFrame(rows).sort_values("Δ_mean_AUC", ascending=False).reset_index(drop=True)


def best_fe8_per_base(summary: pd.DataFrame, model: str = "LightGBM") -> pd.DataFrame:
    sub = summary[(summary["model"] == model) & (summary["fe8_label"] != "no_FE8")].copy()
    idx = sub.groupby("eda_base")["AUC_mean"].idxmax()
    return sub.loc[idx].sort_values("AUC_mean", ascending=False).reset_index(drop=True)


def best_fe8_per_base_dual(summary: pd.DataFrame) -> pd.DataFrame:
    """各 eda_base 上 Δ_mean_AUC 最大的 FE8 combo（双模型合并视角）。"""
    dual = build_eda_fe8_combo_dual(summary)
    dual_fe8 = dual[dual["fe8_label"] != "no_FE8"].copy()
    idx = dual_fe8.groupby("eda_base")["Δ_mean_AUC"].idxmax()
    return dual_fe8.loc[idx].sort_values("Δ_mean_AUC", ascending=False).reset_index(drop=True)


# --- §7 Top-N 多种子复验（experiments；与主 notebook §8 同协议） ---

F1_FP_MEAN_AGG = {
    "thr_combo_mean": (THR_BEST_COL, "mean"),
    "F1_BASE_mean": (F1_BASE_COL, "mean"),
    "F1_COMBO_mean": (F1_COMBO_COL, "mean"),
    "FP_mean": ("FP", "mean"),
}


def load_runtime_state(report_dir: Path) -> dict:
    path = report_dir / "runtime_state.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_ablation_dataframe(report_dir: Path):
    """与主 notebook §7 一致：读 df_features.pkl + BASE_FEATURES。"""
    state = load_runtime_state(report_dir)
    df_fe = bind_cv_data(pd.read_pickle(report_dir / "df_features.pkl"))
    return df_fe, list(state["BASE_FEATURES"])


def load_ablation_extra_map(specs_csv: Path) -> dict[str, list[str]]:
    specs = pd.read_csv(specs_csv)
    out: dict[str, list[str]] = {}
    for _, row in specs.iterrows():
        extra = [c.strip() for c in str(row["extra_cols"]).split("|") if c.strip()]
        out[str(row["label"])] = extra
    return out


def load_section7_top_labels(dual_csv: Path, top_n: int = 20) -> list[str]:
    dual = pd.read_csv(dual_csv)
    return [
        lbl for lbl in dual["特征组合"].tolist()
        if not str(lbl).startswith("0.")
    ][:top_n]


def _ensure_dual_f1_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if F1_BASE_COL not in out.columns and F1_COMBO_COL in out.columns:
        out[F1_BASE_COL] = out[F1_COMBO_COL]
    if F1_COMBO_COL not in out.columns and F1_BASE_COL in out.columns:
        out[F1_COMBO_COL] = out[F1_BASE_COL]
    if THR_BEST_COL not in out.columns:
        out[THR_BEST_COL] = np.nan
    return out


def _dual_model_avg(lgb, xgb, label, mean_col):
    return float(np.mean([float(lgb.loc[label, mean_col]), float(xgb.loc[label, mean_col])]))


def _dual_f1_col_avg(lgb, xgb, label, col, fallback="F1_mean"):
    if col in lgb.columns:
        return _dual_model_avg(lgb, xgb, label, col)
    if col == "thr_combo_mean":
        return float("nan")
    if fallback in lgb.columns:
        return _dual_model_avg(lgb, xgb, label, fallback)
    raise KeyError(f"缺少列 {col!r}（且无 fallback {fallback!r}）")


def _dual_f1_fp(lgb, xgb, label):
    return {
        THR_BEST_COL: _dual_f1_col_avg(lgb, xgb, label, "thr_combo_mean"),
        F1_BASE_COL: _dual_f1_col_avg(lgb, xgb, label, "F1_BASE_mean"),
        F1_COMBO_COL: _dual_f1_col_avg(lgb, xgb, label, "F1_COMBO_mean"),
        "FP": _dual_f1_col_avg(lgb, xgb, label, "FP_mean", fallback="FP_mean"),
    }


def _f1_at_summary_row(df_i, label, f1_col):
    mean_map = {
        F1_BASE_COL: "F1_BASE_mean",
        F1_COMBO_COL: "F1_COMBO_mean",
        THR_BEST_COL: "thr_combo_mean",
    }
    for col in (mean_map.get(f1_col), f1_col):
        if col and col in df_i.columns:
            return float(df_i.loc[label, col])
    raise KeyError(f"缺少 F1 列 {f1_col!r}（期望 {mean_map.get(f1_col)!r} 或原名）")


def _dual_f1_deltas_vs_base(lgb_i, xgb_i, base_label: str) -> dict:
    return {
        "base_f1_base_lgb": _f1_at_summary_row(lgb_i, base_label, F1_BASE_COL),
        "base_f1_base_xgb": _f1_at_summary_row(xgb_i, base_label, F1_BASE_COL),
        "base_f1_combo_lgb": _f1_at_summary_row(lgb_i, base_label, F1_COMBO_COL),
        "base_f1_combo_xgb": _f1_at_summary_row(xgb_i, base_label, F1_COMBO_COL),
    }


def _row_delta_f1(lgb_i, xgb_i, label, bases) -> dict:
    return {
        "Δ_mean_F1_BASE": (
            (_f1_at_summary_row(lgb_i, label, F1_BASE_COL) - bases["base_f1_base_lgb"])
            + (_f1_at_summary_row(xgb_i, label, F1_BASE_COL) - bases["base_f1_base_xgb"])
        ) / 2,
        "Δ_mean_F1_combo": (
            (_f1_at_summary_row(lgb_i, label, F1_COMBO_COL) - bases["base_f1_combo_lgb"])
            + (_f1_at_summary_row(xgb_i, label, F1_COMBO_COL) - bases["base_f1_combo_xgb"])
        ) / 2,
    }


def _resolve_ablation_baseline_label(per_model: pd.DataFrame) -> str:
    labels = per_model["特征组合"].astype(str).unique()
    for lbl in labels:
        if lbl.startswith("0."):
            return lbl
    if ABLATION_BASELINE_LABEL in labels:
        return ABLATION_BASELINE_LABEL
    raise ValueError("per_model 中未找到基线行（特征组合应以 0. 开头）")


def _ablation_base_row(model: str, seed: int, base_res: dict, baseline_label: str) -> dict:
    return {
        "特征组合": baseline_label,
        "模型": model,
        "seed": seed,
        "AUC-PR_mean": base_res["AUC-PR_mean"],
        "delta_AUC_vs_BASE": 0.0,
        THR_BEST_COL: base_res.get(THR_BEST_COL, np.nan),
        F1_BASE_COL: base_res[F1_BASE_COL],
        F1_COMBO_COL: base_res[F1_COMBO_COL],
        "FP": base_res["FP"],
    }


def _summarize_ablation_multiseed_raw(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _ensure_dual_f1_columns(raw)
    return (
        raw.groupby(["特征组合", "模型"], as_index=False)
        .agg(
            delta_mean_AUC=("delta_AUC_vs_BASE", "mean"),
            delta_std_AUC=("delta_AUC_vs_BASE", "std"),
            positive_seed_ratio=("delta_AUC_vs_BASE", lambda s: float((s > 0).mean())),
            **F1_FP_MEAN_AGG,
        )
    )


def build_ablation_confirm_dual(
    per_model: pd.DataFrame,
    baseline_label: str | None = None,
) -> pd.DataFrame:
    per_model = _ensure_dual_f1_columns(per_model)
    if baseline_label is None:
        baseline_label = _resolve_ablation_baseline_label(per_model)
    lgb = per_model[per_model["模型"] == "LightGBM"].set_index("特征组合")
    xgb = per_model[per_model["模型"] == "XGBoost"].set_index("特征组合")
    if baseline_label not in lgb.index or baseline_label not in xgb.index:
        raise ValueError(f"基线 {baseline_label!r} 须在 per_model 的 LGB/XGB 两行均存在")
    bases = _dual_f1_deltas_vs_base(lgb, xgb, baseline_label)
    rows = []
    for label in lgb.index.intersection(xgb.index):
        if label == baseline_label:
            continue
        dl, dx = float(lgb.loc[label, "delta_mean_AUC"]), float(xgb.loc[label, "delta_mean_AUC"])
        lgb_std = float(lgb.loc[label, "delta_std_AUC"] or 0.0)
        xgb_std = float(xgb.loc[label, "delta_std_AUC"] or 0.0)
        f1d = _dual_f1_fp(lgb, xgb, label)
        row = {
            "特征组合": label,
            "Δ_LGB_AUC": dl, "Δ_XGB_AUC": dx,
            "delta_mean_AUC": (dl + dx) / 2,
            "LGB_positive_seed_ratio": float(lgb.loc[label, "positive_seed_ratio"]),
            "XGB_positive_seed_ratio": float(xgb.loc[label, "positive_seed_ratio"]),
            "both_models_positive": dl > 0 and dx > 0,
            "conservative_score": ((dl - lgb_std) + (dx - xgb_std)) / 2,
            **f1d,
            "符号翻转": (dl > 0) != (dx > 0),
            **_row_delta_f1(lgb, xgb, label, bases),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("conservative_score", ascending=False).reset_index(drop=True)


def run_ablation_confirm(
    df_fe,
    base_features,
    labels: list[str],
    extra_map: dict[str, list[str]],
    seeds: list[int],
    models=None,
    checkpoint_path: Path | None = None,
    baseline_label: str = ABLATION_BASELINE_LABEL,
    params=None,
):
    """§8 同协议：model×seed 外层；BASE 同 seed 写入 raw；combo 共用 BASE cal 的 t*。"""
    models = list(models if models is not None else MODELS)
    done: dict[tuple, dict] = {}
    if checkpoint_path and checkpoint_path.is_file():
        prev = _ensure_dual_f1_columns(pd.read_csv(checkpoint_path))
        for _, row in prev.iterrows():
            done[(row["特征组合"], row["模型"], int(row["seed"]))] = row.to_dict()
    rows = list(done.values())
    total = len(labels) * len(models) * len(seeds) + len(models) * len(seeds)
    step = len(rows)
    for model in models:
        for seed in seeds:
            base_res = cross_val_eval(
                model, df_fe, base_features, params=params, random_state=seed,
            )
            shared_thr = base_res["best_threshold"]
            base_key = (baseline_label, model, seed)
            if base_key not in done:
                step += 1
                print(f"[{step}/{total}] {model} seed={seed} | {baseline_label}", flush=True)
                base_row = _ablation_base_row(model, seed, base_res, baseline_label)
                rows.append(base_row)
                done[base_key] = base_row
                if checkpoint_path:
                    _ensure_dual_f1_columns(pd.DataFrame(rows)).to_csv(
                        checkpoint_path, index=False, encoding="utf-8-sig",
                    )
            for label in labels:
                extra = extra_map.get(label, [])
                cols = base_features + list(extra)
                key = (label, model, seed)
                if key in done:
                    continue
                step += 1
                print(f"[{step}/{total}] {model} seed={seed} | {label}", flush=True)
                res = cross_val_eval(
                    model, df_fe, cols, params=params, random_state=seed, threshold=shared_thr,
                )
                row = {
                    "特征组合": label, "模型": model, "seed": seed,
                    "AUC-PR_mean": res["AUC-PR_mean"],
                    "delta_AUC_vs_BASE": res["AUC-PR_mean"] - base_res["AUC-PR_mean"],
                    THR_BEST_COL: res.get(THR_BEST_COL, np.nan),
                    F1_BASE_COL: res[F1_BASE_COL],
                    F1_COMBO_COL: res[F1_COMBO_COL],
                    "FP": res["FP"],
                }
                rows.append(row)
                done[key] = row
                if checkpoint_path:
                    _ensure_dual_f1_columns(pd.DataFrame(rows)).to_csv(
                        checkpoint_path, index=False, encoding="utf-8-sig",
                    )
    raw = _ensure_dual_f1_columns(pd.DataFrame(rows))
    per_model = _summarize_ablation_multiseed_raw(raw)
    dual = build_ablation_confirm_dual(per_model, baseline_label=baseline_label)
    return raw, per_model, dual


def ablation_dual_from_raw(
    raw: pd.DataFrame,
    baseline_label: str = ABLATION_BASELINE_LABEL,
) -> pd.DataFrame:
    """从 ablation checkpoint/raw 重建 dual（不重跑 CV）。"""
    per_model = _summarize_ablation_multiseed_raw(raw)
    return build_ablation_confirm_dual(per_model, baseline_label=baseline_label)


def enrich_raw_f1_per_seed(raw: pd.DataFrame, baseline_label: str = ABLATION_BASELINE_LABEL) -> pd.DataFrame:
    """逐 model×seed 行：combo 相对同行 BASE 的 F1@combo_t* 增量（与 Δ_mean_F1_combo 同源）。"""
    raw = _ensure_dual_f1_columns(raw)
    base = raw[raw["特征组合"] == baseline_label][["模型", "seed", F1_COMBO_COL]].rename(
        columns={F1_COMBO_COL: "_base_f1_combo"},
    )
    out = raw.merge(base, on=["模型", "seed"], how="left")
    out["f1_combo_delta_vs_base"] = out[F1_COMBO_COL] - out["_base_f1_combo"]
    out["auc_delta_positive"] = out["delta_AUC_vs_BASE"] > 0
    out["f1_combo_delta_positive"] = out["f1_combo_delta_vs_base"] > 0
    out["auc_up_f1_down"] = out["auc_delta_positive"] & (~out["f1_combo_delta_positive"])
    return out.drop(columns=["_base_f1_combo"])


def summarize_f1_sign_rates(
    raw: pd.DataFrame,
    baseline_label: str = ABLATION_BASELINE_LABEL,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """按 seed_block / 特征组合 等聚合：Δ 为正的比例与均值。"""
    group_cols = group_cols or ["seed_block"]
    enriched = enrich_raw_f1_per_seed(raw, baseline_label=baseline_label)
    combos = enriched[~enriched["特征组合"].astype(str).str.startswith("0.")]
    return (
        combos.groupby(group_cols, as_index=False)
        .agg(
            n_rows=("特征组合", "count"),
            n_combos=("特征组合", "nunique"),
            n_seeds=("seed", "nunique"),
            auc_pos_rate=("auc_delta_positive", "mean"),
            f1_combo_pos_rate=("f1_combo_delta_positive", "mean"),
            auc_up_f1_down_rate=("auc_up_f1_down", "mean"),
            mean_delta_AUC=("delta_AUC_vs_BASE", "mean"),
            mean_f1_combo_delta=("f1_combo_delta_vs_base", "mean"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def dual_f1_combo_sign_summary(dual: pd.DataFrame) -> dict:
    """双模型合并表：Δ_mean_F1_combo 正负计数。"""
    if dual.empty or "Δ_mean_F1_combo" not in dual.columns:
        return {"n": 0, "n_positive": 0, "positive_rate": float("nan")}
    s = dual["Δ_mean_F1_combo"]
    n_pos = int((s > 0).sum())
    return {"n": int(len(s)), "n_positive": n_pos, "positive_rate": float(n_pos / len(s))}


def run_multi_seed_block_confirm(
    df_fe,
    base_features,
    labels: list[str],
    extra_map: dict[str, list[str]],
    seed_blocks: dict[str, list[int]],
    out_dir: Path,
    models=None,
    baseline_label: str = ABLATION_BASELINE_LABEL,
    params=None,
    skip_if_exists: bool = False,
):
    """同一批 combo 在多个 seed 块上复验（Arm A 核心）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_raw, summaries, duals = [], [], []
    for block_name, seeds in seed_blocks.items():
        ckpt = out_dir / f"checkpoint_{block_name}.csv"
        if skip_if_exists and ckpt.is_file():
            raw = _ensure_dual_f1_columns(pd.read_csv(ckpt))
            per_model = _summarize_ablation_multiseed_raw(raw)
            dual = build_ablation_confirm_dual(per_model, baseline_label=baseline_label)
            print(f"[{block_name}] 从 checkpoint 加载")
        else:
            raw, per_model, dual = run_ablation_confirm(
                df_fe, base_features, labels, extra_map,
                seeds=list(seeds), models=models,
                checkpoint_path=ckpt,
                baseline_label=baseline_label,
                params=params,
            )
        raw = raw.copy()
        raw["seed_block"] = block_name
        dual = dual.copy()
        dual["seed_block"] = block_name
        sign = dual_f1_combo_sign_summary(dual)
        summaries.append({
            "seed_block": block_name,
            "seeds": list(seeds),
            **sign,
            "mean_delta_mean_AUC": float(dual["delta_mean_AUC"].mean()) if len(dual) else float("nan"),
            "mean_delta_mean_F1_combo": float(dual["Δ_mean_F1_combo"].mean()) if len(dual) else float("nan"),
        })
        raw.to_csv(out_dir / f"raw_{block_name}.csv", index=False, encoding="utf-8-sig")
        dual.to_csv(out_dir / f"dual_{block_name}.csv", index=False, encoding="utf-8-sig")
        all_raw.append(raw)
        duals.append(dual)
    raw_all = pd.concat(all_raw, ignore_index=True)
    dual_all = pd.concat(duals, ignore_index=True)
    summary = pd.DataFrame(summaries)
    sign_detail = summarize_f1_sign_rates(raw_all, baseline_label=baseline_label)
    per_combo = summarize_f1_sign_rates(
        raw_all, baseline_label=baseline_label, group_cols=["seed_block", "特征组合"],
    )
    raw_all.to_csv(out_dir / "raw_all_blocks.csv", index=False, encoding="utf-8-sig")
    dual_all.to_csv(out_dir / "dual_all_blocks.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "block_sign_summary.csv", index=False, encoding="utf-8-sig")
    sign_detail.to_csv(out_dir / "block_per_seed_sign_rates.csv", index=False, encoding="utf-8-sig")
    per_combo.to_csv(out_dir / "combo_by_block_sign_rates.csv", index=False, encoding="utf-8-sig")
    return raw_all, dual_all, summary, sign_detail, per_combo


def load_section7_bottom_labels(dual_csv: Path, bottom_n: int = 15) -> list[str]:
    dual = pd.read_csv(dual_csv)
    dual = dual[~dual["特征组合"].astype(str).str.startswith("0.")].copy()
    sort_col = "conservative_score" if "conservative_score" in dual.columns else "delta_mean_AUC"
    dual = dual.sort_values(sort_col, ascending=True)
    return dual["特征组合"].tolist()[:bottom_n]


def _oof_cache_path(out_dir: Path, label: str, model: str, seed: int) -> Path:
    safe = label.replace("/", "_").replace("+", "_").replace(" ", "")
    return out_dir / "oof_cache" / f"{safe}__{model}__s{seed}.npz"


def run_ablation_f1_stability_arm(
    df_fe,
    base_features,
    labels: list[str],
    extra_map: dict[str, list[str]],
    seeds: list[int],
    out_dir: Path,
    protocols: list[str] | None = None,
    models=None,
    baseline_label: str = ABLATION_BASELINE_LABEL,
    params=None,
    skip_if_exists: bool = False,
):
    """Arm D：CV 一次，多 F1 协议后处理；比较跨 seed F1 Δ 稳定性。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "oof_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    protocols = list(protocols or F1_PROTOCOL_COLUMNS.keys())
    models = list(models if models is not None else MODELS)
    y_all = df_fe["Class"].values

    rows = []
    for model in models:
        for seed in seeds:
            base_path = _oof_cache_path(out_dir, baseline_label, model, seed)
            if skip_if_exists and base_path.is_file():
                base_npz = np.load(base_path)
                base_oof = base_npz["oof"]
                base_thr = float(base_npz["best_threshold"])
                base_auc = float(base_npz["auc_pr_mean"])
            else:
                base_res = cross_val_eval(
                    model, df_fe, base_features, params=params,
                    random_state=seed, return_oof=True,
                )
                base_oof = base_res["oof"]
                base_thr = float(base_res["best_threshold"])
                base_auc = float(base_res["AUC-PR_mean"])
                np.savez_compressed(
                    base_path, oof=base_oof, best_threshold=base_thr, auc_pr_mean=base_auc,
                )
            base_f1_by_proto = {}
            base_thr_by_proto = {}
            for p in protocols:
                pr = metrics_f1_protocol(y_all, base_oof, p, base_threshold=base_thr)
                base_f1_by_proto[p] = pr[F1_PROTOCOL_COLUMNS[p]]
                base_thr_by_proto[p] = pr[f"thr@{p}"]
            rows.append({
                "特征组合": baseline_label, "模型": model, "seed": seed,
                "AUC-PR_mean": base_auc, "delta_AUC_vs_BASE": 0.0,
                "base_threshold": base_thr, **{
                    f"f1_{p}": base_f1_by_proto[p] for p in protocols
                }, **{
                    f"thr_{p}": base_thr_by_proto[p] for p in protocols
                },
            })

            for label in labels:
                oof_path = _oof_cache_path(out_dir, label, model, seed)
                extra = extra_map.get(label, [])
                cols = base_features + list(extra)
                if skip_if_exists and oof_path.is_file():
                    combo_npz = np.load(oof_path)
                    combo_oof = combo_npz["oof"]
                    combo_auc = float(combo_npz["auc_pr_mean"])
                else:
                    combo_res = cross_val_eval(
                        model, df_fe, cols, params=params,
                        random_state=seed, threshold=base_thr, return_oof=True,
                    )
                    combo_oof = combo_res["oof"]
                    combo_auc = float(combo_res["AUC-PR_mean"])
                    np.savez_compressed(
                        oof_path, oof=combo_oof, auc_pr_mean=combo_auc,
                    )
                f1_map = {}
                thr_map = {}
                for p in protocols:
                    pr = metrics_f1_protocol(y_all, combo_oof, p, base_threshold=base_thr)
                    f1_map[p] = pr[F1_PROTOCOL_COLUMNS[p]]
                    thr_map[p] = pr[f"thr@{p}"]
                rows.append({
                    "特征组合": label, "模型": model, "seed": seed,
                    "AUC-PR_mean": combo_auc,
                    "delta_AUC_vs_BASE": combo_auc - base_auc,
                    "base_threshold": base_thr,
                    **{f"f1_{p}": f1_map[p] for p in protocols},
                    **{f"thr_{p}": thr_map[p] for p in protocols},
                })

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw_f1_protocols.csv", index=False, encoding="utf-8-sig")
    summary = summarize_f1_protocol_stability(raw, protocols, baseline_label=baseline_label)
    summary.to_csv(out_dir / "protocol_stability_summary.csv", index=False, encoding="utf-8-sig")
    dual_by_proto = {
        p: build_dual_from_protocol_raw(raw, p, baseline_label=baseline_label)
        for p in protocols
    }
    for p, dual in dual_by_proto.items():
        dual.to_csv(out_dir / f"dual_{p}.csv", index=False, encoding="utf-8-sig")
    return raw, summary, dual_by_proto


def summarize_f1_protocol_stability(
    raw: pd.DataFrame,
    protocols: list[str],
    baseline_label: str = ABLATION_BASELINE_LABEL,
) -> pd.DataFrame:
    """每协议：跨 seed 行级 F1 Δ 为正比例、thr 标准差、combo 级 Δ_mean 符号率。"""
    combos = raw[~raw["特征组合"].astype(str).str.startswith("0.")].copy()
    base = raw[raw["特征组合"] == baseline_label].set_index(["模型", "seed"])
    rows = []
    for p in protocols:
        col = f"f1_{p}"
        merged = combos.merge(
            base[[col]].rename(columns={col: "_base_f1"}),
            left_on=["模型", "seed"], right_index=True, how="left",
        )
        merged["f1_delta"] = merged[col] - merged["_base_f1"]
        thr_col = f"thr_{p}"
        mean_thr_std = float(
            merged.groupby(["特征组合", "模型"])[thr_col].std().mean()
        ) if thr_col in merged.columns else float("nan")
        dual = build_dual_from_protocol_raw(raw, p, baseline_label=baseline_label)
        rows.append({
            "protocol": p,
            "description": F1_PROTOCOL_DESCRIPTIONS.get(p, ""),
            "f1_col": F1_PROTOCOL_COLUMNS.get(p, col),
            "row_f1_pos_rate": float((merged["f1_delta"] > 0).mean()),
            "mean_f1_delta": float(merged["f1_delta"].mean()),
            "std_f1_delta_across_seeds": float(
                merged.groupby(["特征组合", "模型"])["f1_delta"].std().mean()
            ),
            "mean_thr_std_per_combo": mean_thr_std,
            "combo_dual_pos_rate": float((dual["delta_mean_F1"] > 0).mean()) if len(dual) else float("nan"),
            "mean_combo_dual_f1_delta": float(dual["delta_mean_F1"].mean()) if len(dual) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("std_f1_delta_across_seeds")


def build_dual_from_protocol_raw(
    raw: pd.DataFrame,
    protocol: str,
    baseline_label: str = ABLATION_BASELINE_LABEL,
) -> pd.DataFrame:
    col = f"f1_{protocol}"
    per = (
        raw.groupby(["特征组合", "模型"], as_index=False)
        .agg(f1_mean=(col, "mean"), delta_auc=("delta_AUC_vs_BASE", "mean"))
    )
    lgb = per[per["模型"] == "LightGBM"].set_index("特征组合")
    xgb = per[per["模型"] == "XGBoost"].set_index("特征组合")
    base_lgb = float(lgb.loc[baseline_label, "f1_mean"]) if baseline_label in lgb.index else 0.0
    base_xgb = float(xgb.loc[baseline_label, "f1_mean"]) if baseline_label in xgb.index else 0.0
    rows = []
    for label in lgb.index.intersection(xgb.index):
        if str(label).startswith("0."):
            continue
        dl = float(lgb.loc[label, "f1_mean"]) - base_lgb
        dx = float(xgb.loc[label, "f1_mean"]) - base_xgb
        rows.append({
            "特征组合": label,
            "protocol": protocol,
            "Δ_LGB_F1": dl, "Δ_XGB_F1": dx,
            "delta_mean_F1": (dl + dx) / 2,
            "delta_mean_AUC": float(np.mean([
                lgb.loc[label, "delta_auc"], xgb.loc[label, "delta_auc"],
            ])),
        })
    return pd.DataFrame(rows).sort_values("delta_mean_F1", ascending=False).reset_index(drop=True)


def merge_section7_rank_compare(
    section7_dual: pd.DataFrame,
    confirm_dual: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """§7 Top-N 排名 vs 本实验多种子复验（按 conservative_score）。"""
    s7 = section7_dual[~section7_dual["特征组合"].astype(str).str.startswith("0.")].copy()
    s7 = s7.head(top_n).reset_index(drop=True)
    s7["§7排名"] = s7.index + 1
    s7_key = s7[["特征组合", "§7排名", "Δ_mean_AUC"]].rename(
        columns={"Δ_mean_AUC": "§7_Δ_mean_AUC"},
    )
    exp = confirm_dual.copy().reset_index(drop=True)
    exp["实验排名"] = exp.index + 1
    merged = s7_key.merge(
        exp,
        on="特征组合",
        how="left",
    )
    merged["排名差(实验-§7)"] = merged["实验排名"] - merged["§7排名"]
    return merged.sort_values("§7排名").reset_index(drop=True)
