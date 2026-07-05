# --- 5. 阶段1：shortlist（可跳过已完成的阶段1跑批）---
if MANUAL_FE8_SHORTLIST is not None and SKIP_STAGE1_RUN:
    SELECTED_FE8_LISTS = [list(cols) for cols in MANUAL_FE8_SHORTLIST]
    print('跳过阶段1跑批，使用硬编码 FE8 shortlist (n=%d):' % len(SELECTED_FE8_LISTS))
    for i, cols in enumerate(SELECTED_FE8_LISTS):
        print(f'  [{i}] {_subset_label(cols)}')
    if STAGE1_DUAL_CSV.is_file():
        stage1_dual = pd.read_csv(STAGE1_DUAL_CSV)
        print('（阶段1历史结果）')
        display(stage1_dual[
            ['combo_label', 'n_fe8', 'fe8_cols', 'delta_mean_AUC', 'conservative_score', 'both_models_positive']
        ].head(5).round(5))
    else:
        stage1_dual = pd.DataFrame()
else:
    stage1_raw = run_stability_matrix(STAGE1_RECORDS, STAGE1_CHECKPOINT)
    _, stage1_summary = summarize_stability(stage1_raw)
    stage1_dual = build_dual_model_summary(stage1_summary)
    stage1_raw.to_csv(STAGE1_RAW_CSV, index=False, encoding='utf-8-sig')
    stage1_dual.to_csv(STAGE1_DUAL_CSV, index=False, encoding='utf-8-sig')
    SELECTED_FE8_LISTS = select_fe8_shortlist(stage1_dual, top_n=3)
    print('=== 阶段1 Top（Δ vs BASE）===')
    display(stage1_dual[
        ['combo_label', 'n_fe8', 'fe8_cols', 'delta_mean_AUC', 'conservative_score', 'both_models_positive']
    ].head(15).round(5))

shortlist_payload = {
    'selected_fe8_lists': SELECTED_FE8_LISTS,
    'n_selected': len(SELECTED_FE8_LISTS),
    'manual_override': MANUAL_FE8_SHORTLIST is not None,
    'skip_stage1_run': SKIP_STAGE1_RUN,
}
with open(STAGE1_SHORTLIST_JSON, 'w', encoding='utf-8') as f:
    json.dump(shortlist_payload, f, ensure_ascii=False, indent=2)
print('shortlist 已写入:', STAGE1_SHORTLIST_JSON)
