# --- 6. 阶段2：构建组合目录 ---
STAGE2_RECORDS = build_stage2_specs(SELECTED_FE8_LISTS)
stage2_catalog_df = export_combo_catalog(
    STAGE2_RECORDS, STAGE2_COMBO_CSV, STAGE2_COMBO_MD, 'FE-8 阶段2：IF×EDA×入选FE8×A'
)
n_main = sum(1 for r in STAGE2_RECORDS if r['category'] == 'stage2_main')
n_ref = len(STAGE2_RECORDS) - n_main
print(f'阶段2 组合数: {len(STAGE2_RECORDS)} (主网格 {n_main} + 参照 {n_ref})')
print(f'  shortlist n={len(SELECTED_FE8_LISTS)} × EDA(32) × A档(2: A1/A_top2) = {len(SELECTED_FE8_LISTS)*32*2} 主网格')
print(f'  运行轮次: {len(STAGE2_RECORDS) * len(MODELS) * len(RUN_SEEDS)}')
print('目录:', STAGE2_COMBO_CSV)
display(stage2_catalog_df.groupby('category').size().rename('count'))