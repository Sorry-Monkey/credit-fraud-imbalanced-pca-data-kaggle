# --- 3. OOF IF 与完整特征表 ---
def oof_if_anomaly_score(data, feature_cols=None, random_state=IF_RANDOM_STATE):
    feature_cols = feature_cols or V_COLS
    X = data[feature_cols].values.astype(np.float64)
    y = data['Class'].values
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
        print(f'  IF fold {fold}/{CV_N_SPLITS} 完成，正常训练样本={len(normal_tr):,}')
    return oof


df_fe8 = build_eda_features(df_raw)
df_fe8 = bind_cv_data(df_fe8)

TOP_V = pick_top_v_features(df_fe8, BASE_FEATURES, k=TOP_V_K, model_name='LightGBM')
df_fe8, CROSS_FAMILY_A = build_cross_features(df_fe8, TOP_V, gate_col='is_one_euro', prefix='one_euro')
df_fe8, FE8_NEW_FEATURES = build_fe8_contribution_features(df_fe8)

print('开始计算 if_oof_score（purged OOF IF）...')
df_fe8['if_oof_score'] = oof_if_anomaly_score(df_fe8, V_COLS, random_state=IF_RANDOM_STATE)

A1_COL = CROSS_FAMILY_A[0]
A2_COL = CROSS_FAMILY_A[1]
A_TOP2 = list(CROSS_FAMILY_A[:2])

with open(MT4_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _mt4 = json.load(f)
_mt4_extra = [c for c in _mt4['MODEL_FEATURES'] if c not in BASE_FEATURES]
assert MT4_FINAL_EXTRA == _mt4_extra, f'MT4 配置与 notebook 不一致: {_mt4_extra}'

print('A1:', A1_COL, '| A2:', A2_COL, '| A_top2:', A_TOP2)
print('FE-8 新特征:', FE8_NEW_FEATURES)