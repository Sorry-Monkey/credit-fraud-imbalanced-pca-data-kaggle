# --- 2. purged walk-forward CV 与模型评估 ---
_CV_BOUND_DATA = None


def sort_by_time(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values('Time', kind='mergesort').reset_index(drop=True)


def bind_cv_data(data: pd.DataFrame) -> pd.DataFrame:
    global _CV_BOUND_DATA
    out = sort_by_time(data)
    _CV_BOUND_DATA = out
    return out


def build_cv_timestamps(data: pd.DataFrame):
    t = pd.to_timedelta(data['Time'].astype(float), unit='s')
    return t.copy(), t.copy()


def iter_purged_cv_folds(n_samples=None, n_splits=CV_N_SPLITS, data=None):
    bound = data if data is not None else _CV_BOUND_DATA
    if bound is None:
        raise RuntimeError('请先调用 bind_cv_data()')
    n = n_samples if n_samples is not None else len(bound)
    pred, evalu = build_cv_timestamps(bound)
    test_size = max(1, n // (n_splits + 1))
    cv = WalkForwardSplit(
        n_splits=n_splits, test_size=test_size, window='expanding',
        prediction_times=pred, evaluation_times=evalu,
        purge_horizon=CV_PURGE_HORIZON, embargo=CV_EMBARGO,
    )
    for tr_idx, va_idx in cv.split(np.arange(n)):
        assert_no_temporal_leakage(tr_idx, va_idx, prediction_times=pred, evaluation_times=evalu, purge_horizon=CV_PURGE_HORIZON)
        yield tr_idx, va_idx


def split_early_stop_set(X_tr, y_tr, es_frac=ES_FRAC, random_state=42):
    return train_test_split(X_tr, y_tr, test_size=es_frac, random_state=random_state, stratify=y_tr)


def make_classifier(model_name: str, y_train: pd.Series, random_state: int = 42):
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    if model_name == 'LightGBM':
        return lgb.LGBMClassifier(
            n_estimators=MAX_BOOST_ROUNDS, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
            random_state=random_state, verbose=-1, n_jobs=-1,
        )
    return xgb.XGBClassifier(
        n_estimators=MAX_BOOST_ROUNDS, learning_rate=0.05, max_depth=6,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=random_state,
        eval_metric='logloss', verbosity=0, n_jobs=-1,
    )


def fit_classifier(clf, model_name: str, X_tr, y_tr, X_es=None, y_es=None):
    if X_es is None or y_es is None:
        clf.fit(X_tr, y_tr)
        return clf
    if model_name == 'LightGBM':
        clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    else:
        clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return clf


def best_f1_threshold(y_true, proba):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    mask = np.isfinite(proba)
    y_eval, p_eval = y_true[mask], proba[mask]
    prec, rec, thr = precision_recall_curve(y_eval, p_eval)
    if len(thr) == 0:
        return DEFAULT_CLASSIFICATION_THRESHOLD, 0.0
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def metrics_at_threshold(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    mask = np.isfinite(proba)
    pred = proba[mask] >= threshold
    y_eval = y_true[mask]
    tn, fp, fn, tp = confusion_matrix(y_eval, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'F1@best': float(f1), 'Precision@best': float(precision), 'Recall@best': float(recall),
        'FP': int(fp), 'FN': int(fn), 'n_eval': int(mask.sum()), 'n_excluded': int((~mask).sum()),
    }


def cross_val_eval(model_name: str, data: pd.DataFrame, feature_cols: list, random_state: int = 42):
    X, y = data[feature_cols], data['Class']
    oof = np.full(len(y), np.nan, dtype=float)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(iter_purged_cv_folds(len(X), n_splits=CV_N_SPLITS), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        X_fit, X_es, y_fit, y_es = split_early_stop_set(X_tr, y_tr, random_state=random_state + fold)
        clf = make_classifier(model_name, y_fit, random_state=random_state + fold)
        fit_classifier(clf, model_name, X_fit, y_fit, X_es, y_es)
        proba_va = clf.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba_va
        fold_scores.append(float(average_precision_score(y_va, proba_va)))
    arr = np.array(fold_scores, dtype=float)
    threshold, _ = best_f1_threshold(y, oof)
    cls = metrics_at_threshold(y, oof, threshold)
    return {
        '模型': model_name, '特征数': len(feature_cols),
        'AUC-PR_mean': float(arr.mean()), 'AUC-PR_std': float(arr.std(ddof=0)),
        'AUC-PR_min': float(arr.min()), 'fold_AUC-PR': fold_scores,
        'best_threshold': threshold, **cls,
    }