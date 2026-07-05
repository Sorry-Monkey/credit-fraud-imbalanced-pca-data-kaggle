# --- 6. 定稿决策 + 导出 ---
def get_extra_by_label(spec_records, label):
    for rec in spec_records:
        if rec['label'] == label:
            return list(rec['extra_cols'])
    raise KeyError(label)


# 在 BASE 上稳定优于 BASE 的 FE8 组合（排除纯 BASE 本身）
eligible = dual_summary[
    dual_summary['both_models_positive']
    & (dual_summary['LGB_positive_seed_ratio'] >= 2 / 3)
    & (dual_summary['XGB_positive_seed_ratio'] >= 2 / 3)
    & (dual_summary['combo_label'] != BASELINE_LABEL)
    & (dual_summary['n_fe8'] > 0)
].copy()

if eligible.empty:
    WINNER_LABEL = MT4_LABEL
    SELECTED_EXTRA = list(MT4_FINAL_EXTRA)
    decision = '未发现相对 BASE 稳定增益的 FE8 组合；保持 MT-4 定稿（IF+hours+log1p+A_top2）。'
else:
    winner = eligible.sort_values(['conservative_score', 'delta_mean_AUC'], ascending=False).iloc[0]
    WINNER_LABEL = winner['combo_label']
    SELECTED_EXTRA = get_extra_by_label(SPEC_RECORDS_FE8, WINNER_LABEL)
    decision = (
        f'优胜组合: {WINNER_LABEL}；相对 BASE 的 delta_mean_AUC={winner["delta_mean_AUC"]:.5f}，'
        f'conservative_score={winner["conservative_score"]:.5f}'
    )

MODEL_FEATURES_V3 = BASE_FEATURES + [c for c in SELECTED_EXTRA if c not in BASE_FEATURES]

# MT-4 参照行的 Δ（便于对比定稿是否被 FE8 超越）
mt4_row = dual_summary[dual_summary['combo_label'] == MT4_LABEL]
mt4_delta = float(mt4_row['delta_mean_AUC'].iloc[0]) if not mt4_row.empty else None

print('=== FE-8 定稿决策 ===')
print(decision)
print('优胜/保留组合:', WINNER_LABEL)
print('MT-4 参照 Δ vs BASE:', mt4_delta)
print('手工特征:', [c for c in SELECTED_EXTRA if c not in FE8_NEW_FEATURES])
print('FE8 特征:', [c for c in SELECTED_EXTRA if c in FE8_NEW_FEATURES])

export_payload = {
    'MODEL_FEATURES_V3': MODEL_FEATURES_V3,
    'winner_combo': WINNER_LABEL,
    'decision': decision,
    'selected_extra': [c for c in MODEL_FEATURES_V3 if c not in BASE_FEATURES],
    'mt4_reference_extra': MT4_FINAL_EXTRA,
    'mt4_reference_label': MT4_LABEL,
    'mt4_delta_vs_base': mt4_delta,
    'fe8_new_features': FE8_NEW_FEATURES,
    'baseline_label': BASELINE_LABEL,
    'combo_catalog_csv': str(COMBO_CSV_PATH),
    'dual_summary_csv': str(DUAL_CSV_PATH),
}

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(export_payload, f, ensure_ascii=False, indent=2)

with open(RESULT_PATH, 'w', encoding='utf-8') as f:
    json.dump({
        'seed_level_results': stability_seed.round(6).to_dict(orient='records'),
        'dual_summary': dual_summary.round(6).to_dict(orient='records'),
        'combo_catalog': combo_catalog_df.to_dict(orient='records'),
    }, f, ensure_ascii=False, indent=2)

print('已导出:', OUTPUT_PATH)
print('已导出:', RESULT_PATH)

# --- 7. 可视化 ---
plot_df = dual_summary.head(20).iloc[::-1].copy()
fig, ax = plt.subplots(figsize=(11, max(6, 0.38 * len(plot_df))))
colors = []
for _, row in plot_df.iterrows():
    if row['combo_label'] == MT4_LABEL:
        colors.append('#E45756')
    elif row['both_models_positive']:
        colors.append('#4C78A8')
    else:
        colors.append('#BAB0AC')
ax.barh(plot_df['combo_label'], plot_df['delta_mean_AUC'], color=colors)
ax.axvline(0, color='black', linewidth=1)
ax.set_title('FE-8 Top 20：相对 BASE 的 Δ AUC-PR（红=MT-4 参照）')
ax.set_xlabel('双模型平均 Δ AUC-PR')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fe8_delta_auc_top20.png', dpi=150, bbox_inches='tight')
plt.show()
print('图表已保存:', FIG_DIR / 'fe8_delta_auc_top20.png')