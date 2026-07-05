# --- 5. 跑稳定性矩阵 + 汇总（Δ 相对 BASE）---
def summarize_stability(raw: pd.DataFrame, baseline_label=BASELINE_LABEL):
    base = (
        raw[raw['特征组合'] == baseline_label]
        .groupby(['模型', 'seed'], as_index=False)['AUC-PR_mean']
        .mean()
        .rename(columns={'AUC-PR_mean': 'BASELINE_AUC_seed'})
    )
    if base.empty:
        raise ValueError(f'基线 {baseline_label} 无结果，请先跑完基线组合')

    merged = raw.merge(base, on=['模型', 'seed'], how='left')
    merged['delta_AUC_vs_BASE'] = merged['AUC-PR_mean'] - merged['BASELINE_AUC_seed']

    summary = (
        merged.groupby(['特征组合', '模型'], as_index=False)
        .agg(
            combo_id=('combo_id', 'first'),
            category=('category', 'first'),
            n_handcrafted=('n_handcrafted', 'first'),
            n_fe8=('n_fe8', 'first'),
            n_extra=('n_extra', 'first'),
            handcrafted_cols=('handcrafted_cols', 'first'),
            fe8_cols=('fe8_cols', 'first'),
            extra_cols=('extra_cols', 'first'),
            AUC_mean=('AUC-PR_mean', 'mean'),
            AUC_std_across_seed=('AUC-PR_mean', 'std'),
            delta_mean_AUC=('delta_AUC_vs_BASE', 'mean'),
            delta_std_AUC=('delta_AUC_vs_BASE', 'std'),
            positive_seed_ratio=('delta_AUC_vs_BASE', lambda s: float((s > 0).mean())),
            F1_mean=('F1@best', 'mean'),
            FP_mean=('FP', 'mean'),
            FN_mean=('FN', 'mean'),
        )
    )
    summary['conservative_score'] = summary['delta_mean_AUC'] - summary['delta_std_AUC'].fillna(0.0)
    return merged, summary.sort_values(['模型', 'conservative_score'], ascending=[True, False]).reset_index(drop=True)


def build_dual_model_summary(stability_summary: pd.DataFrame):
    lgb_s = stability_summary[stability_summary['模型'] == 'LightGBM'].set_index('特征组合')
    xgb_s = stability_summary[stability_summary['模型'] == 'XGBoost'].set_index('特征组合')
    rows = []
    for label in lgb_s.index.intersection(xgb_s.index):
        rows.append({
            'combo_id': int(lgb_s.loc[label, 'combo_id']),
            'category': lgb_s.loc[label, 'category'],
            'combo_label': label,
            'n_handcrafted': int(lgb_s.loc[label, 'n_handcrafted']),
            'n_fe8': int(lgb_s.loc[label, 'n_fe8']),
            'n_extra': int(lgb_s.loc[label, 'n_extra']),
            'handcrafted_cols': lgb_s.loc[label, 'handcrafted_cols'],
            'fe8_cols': lgb_s.loc[label, 'fe8_cols'],
            'extra_cols': lgb_s.loc[label, 'extra_cols'],
            'delta_LGB_AUC': float(lgb_s.loc[label, 'delta_mean_AUC']),
            'delta_XGB_AUC': float(xgb_s.loc[label, 'delta_mean_AUC']),
            'delta_mean_AUC': float(np.mean([lgb_s.loc[label, 'delta_mean_AUC'], xgb_s.loc[label, 'delta_mean_AUC']])),
            'LGB_positive_seed_ratio': float(lgb_s.loc[label, 'positive_seed_ratio']),
            'XGB_positive_seed_ratio': float(xgb_s.loc[label, 'positive_seed_ratio']),
            'both_models_positive': bool(lgb_s.loc[label, 'delta_mean_AUC'] > 0 and xgb_s.loc[label, 'delta_mean_AUC'] > 0),
            'conservative_score': float(np.mean([lgb_s.loc[label, 'conservative_score'], xgb_s.loc[label, 'conservative_score']])),
        })
    return pd.DataFrame(rows).sort_values(
        ['both_models_positive', 'conservative_score', 'delta_mean_AUC'], ascending=False
    ).reset_index(drop=True)


stability_raw = run_stability_matrix(SPEC_RECORDS_FE8)
stability_seed, stability_summary = summarize_stability(stability_raw)
dual_summary = build_dual_model_summary(stability_summary)

stability_raw.to_csv(RAW_CSV_PATH, index=False, encoding='utf-8-sig')
stability_summary.to_csv(SUMMARY_CSV_PATH, index=False, encoding='utf-8-sig')
dual_summary.to_csv(DUAL_CSV_PATH, index=False, encoding='utf-8-sig')

print('=== 稳定性汇总 vs BASE（Top 20）===')
display(stability_summary[
    ['特征组合', '模型', 'n_handcrafted', 'n_fe8', 'delta_mean_AUC', 'positive_seed_ratio', 'conservative_score']
].head(20).round(5))

print('=== 双模型汇总（Top 20）===')
display(dual_summary[
    ['combo_id', 'category', 'combo_label', 'n_handcrafted', 'n_fe8', 'delta_mean_AUC', 'conservative_score', 'both_models_positive']
].head(20).round(5))

print('完整结果:')
print(' -', RAW_CSV_PATH)
print(' -', SUMMARY_CSV_PATH)
print(' -', DUAL_CSV_PATH)