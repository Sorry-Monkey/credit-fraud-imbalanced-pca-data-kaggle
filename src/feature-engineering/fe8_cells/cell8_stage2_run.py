# --- 7. 阶段2：跑矩阵 + 汇总 ---
stage2_raw = run_stability_matrix(STAGE2_RECORDS, STAGE2_CHECKPOINT)
stage2_seed, stage2_summary = summarize_stability(stage2_raw)
stage2_dual = build_dual_model_summary(stage2_summary)

stage2_raw.to_csv(STAGE2_RAW_CSV, index=False, encoding='utf-8-sig')
stage2_summary.to_csv(STAGE2_SUMMARY_CSV, index=False, encoding='utf-8-sig')
stage2_dual.to_csv(STAGE2_DUAL_CSV, index=False, encoding='utf-8-sig')

print('=== 阶段2 Top（Δ vs BASE）===')
display(stage2_dual[
    ['combo_label', 'category', 'n_fe8', 'delta_mean_AUC', 'conservative_score', 'both_models_positive']
].head(20).round(5))
print('完整结果:', STAGE2_DUAL_CSV)