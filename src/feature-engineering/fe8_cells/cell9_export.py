# --- 8. 定稿决策 + 导出 + 图 ---
def get_extra_by_label(spec_records, label):
    for rec in spec_records:
        if rec['label'] == label:
            return list(rec['extra_cols'])
    raise KeyError(label)


eligible = stage2_dual[
    stage2_dual['both_models_positive']
    & (stage2_dual['LGB_positive_seed_ratio'] >= 2 / 3)
    & (stage2_dual['XGB_positive_seed_ratio'] >= 2 / 3)
    & (stage2_dual['combo_label'] != BASELINE_LABEL)
    & (stage2_dual['category'] == 'stage2_main')
].copy()

if eligible.empty:
    WINNER_LABEL = MT4_LABEL
    SELECTED_EXTRA = list(MT4_FINAL_EXTRA)
    decision = '阶段2 无稳定优胜组合；保留 MT-4 定稿（IF+hours+log1p+A_top2）。'
else:
    winner = eligible.sort_values(['conservative_score', 'delta_mean_AUC'], ascending=False).iloc[0]
    WINNER_LABEL = winner['combo_label']
    SELECTED_EXTRA = get_extra_by_label(STAGE2_RECORDS, WINNER_LABEL)
    decision = (
        f'优胜: {WINNER_LABEL} | Δ vs BASE={winner["delta_mean_AUC"]:.5f} | '
        f'conservative={winner["conservative_score"]:.5f}'
    )

MODEL_FEATURES_V3 = BASE_FEATURES + [c for c in SELECTED_EXTRA if c not in BASE_FEATURES]
mt4_row = stage2_dual[stage2_dual['combo_label'] == MT4_LABEL]
mt4_delta = float(mt4_row['delta_mean_AUC'].iloc[0]) if not mt4_row.empty else None

print('=== FE-8 定稿 ===')
print(decision)
print('组合:', WINNER_LABEL)
print('MT-4 Δ vs BASE:', mt4_delta)
print('增量列:', [c for c in SELECTED_EXTRA if c not in BASE_FEATURES])

export_payload = {
    'MODEL_FEATURES_V3': MODEL_FEATURES_V3,
    'winner_combo': WINNER_LABEL,
    'decision': decision,
    'selected_extra': [c for c in MODEL_FEATURES_V3 if c not in BASE_FEATURES],
    'stage1_shortlist': SELECTED_FE8_LISTS,
    'stage1_n_combos': len(STAGE1_RECORDS),
    'stage2_n_combos': len(STAGE2_RECORDS),
    'mt4_reference_extra': MT4_FINAL_EXTRA,
    'mt4_reference_label': MT4_LABEL,
    'mt4_delta_vs_base': mt4_delta,
    'fe8_new_features': FE8_NEW_FEATURES,
    'baseline_label': BASELINE_LABEL,
    'stage2_dual_summary_csv': str(STAGE2_DUAL_CSV),
}

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(export_payload, f, ensure_ascii=False, indent=2)

with open(RESULT_PATH, 'w', encoding='utf-8') as f:
    json.dump({
        'stage1_dual_summary': stage1_dual.round(6).to_dict(orient='records'),
        'stage1_shortlist': SELECTED_FE8_LISTS,
        'stage2_dual_summary': stage2_dual.round(6).to_dict(orient='records'),
        'stage2_catalog': stage2_catalog_df.to_dict(orient='records'),
    }, f, ensure_ascii=False, indent=2)

plot_df = stage2_dual.head(20).iloc[::-1].copy()
fig, ax = plt.subplots(figsize=(11, max(6, 0.38 * len(plot_df))))
colors = ['#E45756' if r['combo_label'] == MT4_LABEL else ('#4C78A8' if r['both_models_positive'] else '#BAB0AC') for _, r in plot_df.iterrows()]
ax.barh(plot_df['combo_label'], plot_df['delta_mean_AUC'], color=colors)
ax.axvline(0, color='black', linewidth=1)
ax.set_title('阶段2 Top20：Δ AUC-PR vs BASE（红=MT-4）')
ax.set_xlabel('双模型平均 Δ AUC-PR')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fe8_stage2_delta_top20.png', dpi=150, bbox_inches='tight')
plt.show()
print('已导出:', OUTPUT_PATH, RESULT_PATH)
print('图表:', FIG_DIR / 'fe8_stage2_delta_top20.png')