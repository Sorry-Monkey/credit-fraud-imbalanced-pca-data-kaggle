# --- 9. Top-5 扩展 seeds 复验 ---
def _parse_pipe_cols(s):
    if not isinstance(s, str) or not s.strip() or s == '(none)':
        return []
    return [c.strip() for c in s.split('|')]


def _record_from_row(row, rank_stage2):
    extra = _parse_pipe_cols(row['extra_cols'])
    fe8 = _parse_pipe_cols(row['fe8_cols'])
    hc = _parse_pipe_cols(row['handcrafted_cols'])
    return {
        'combo_id': int(row['combo_id']),
        'stage': 3,
        'category': 'top5_rerun',
        'label': row['combo_label'],
        'handcrafted_cols': hc,
        'fe8_cols': fe8,
        'extra_cols': extra,
        'n_handcrafted': int(row['n_handcrafted']),
        'n_fe8': int(row['n_fe8']),
        'n_extra': int(row['n_extra']),
        'rank_stage2': rank_stage2,
    }


def build_top5_validation_specs(stage2_dual_df):
    refs = stage2_dual_df[stage2_dual_df['combo_label'].isin([BASELINE_LABEL, MT4_LABEL])]
    main = stage2_dual_df[stage2_dual_df['category'] == 'stage2_main'].copy()
    main = main.sort_values(['conservative_score', 'delta_mean_AUC'], ascending=False).head(5)
    records = []
    for _, row in refs.iterrows():
        records.append(_record_from_row(row, 0))
    for i, (_, row) in enumerate(main.iterrows(), start=1):
        records.append(_record_from_row(row, i))
    return records, main


def summarize_top5_validation(raw: pd.DataFrame):
    base = (
        raw[raw['特征组合'] == BASELINE_LABEL]
        .groupby(['模型', 'seed'], as_index=False)['AUC-PR_mean']
        .mean()
        .rename(columns={'AUC-PR_mean': 'BASE_AUC_seed'})
    )
    mt4 = (
        raw[raw['特征组合'] == MT4_LABEL]
        .groupby(['模型', 'seed'], as_index=False)['AUC-PR_mean']
        .mean()
        .rename(columns={'AUC-PR_mean': 'MT4_AUC_seed'})
    )
    merged = raw.merge(base, on=['模型', 'seed'], how='left')
    merged['delta_vs_BASE'] = merged['AUC-PR_mean'] - merged['BASE_AUC_seed']
    merged = merged.merge(mt4, on=['模型', 'seed'], how='left')
    merged['delta_vs_MT4'] = merged['AUC-PR_mean'] - merged['MT4_AUC_seed']

    dual_rows = []
    for label in merged['特征组合'].unique():
        if label in (BASELINE_LABEL, MT4_LABEL):
            continue
        sub = merged[merged['特征组合'] == label]
        lgb = sub[sub['模型'] == 'LightGBM']
        xgb = sub[sub['模型'] == 'XGBoost']
        rank_val = int(sub['rank_stage2'].iloc[0]) if 'rank_stage2' in sub.columns else 0
        dual_rows.append({
            'combo_label': label,
            'rank_stage2': rank_val,
            'delta_LGB_vs_BASE': float(lgb['delta_vs_BASE'].mean()),
            'delta_XGB_vs_BASE': float(xgb['delta_vs_BASE'].mean()),
            'delta_mean_vs_BASE': float(sub['delta_vs_BASE'].mean()),
            'delta_mean_vs_MT4': float(sub['delta_vs_MT4'].mean()),
            'positive_vs_BASE_ratio': float((sub['delta_vs_BASE'] > 0).mean()),
            'conservative_vs_BASE': float(
                sub.groupby('模型')['delta_vs_BASE'].mean().mean()
                - sub.groupby('模型')['delta_vs_BASE'].std().fillna(0).mean()
            ),
        })
    dual = pd.DataFrame(dual_rows)
    top5_rerun = dual[dual['rank_stage2'] > 0].sort_values(
        'conservative_vs_BASE', ascending=False
    )
    return merged, dual, top5_rerun


if 'stage2_dual' not in globals() or stage2_dual is None or stage2_dual.empty:
    if not STAGE2_DUAL_CSV.is_file():
        raise FileNotFoundError(f'未找到阶段2结果: {STAGE2_DUAL_CSV}，请先跑 cell 8')
    _stage2_dual = pd.read_csv(STAGE2_DUAL_CSV)
else:
    _stage2_dual = stage2_dual.copy()

TOP5_SPECS, TOP5_STAGE2_TABLE = build_top5_validation_specs(_stage2_dual)

print('=== 阶段2 Top-5（扩展 seeds 复验对象）===')
display(TOP5_STAGE2_TABLE[['combo_label', 'delta_mean_AUC', 'conservative_score', 'extra_cols']].round(5))

_rank1 = TOP5_STAGE2_TABLE.iloc[0]
_mt4 = _stage2_dual[_stage2_dual['combo_label'] == MT4_LABEL].iloc[0]
print('\n=== 当前排名第1（阶段2: seeds 42, 2026）===')
print('组合:', _rank1['combo_label'])
print(f"超 BASE: Δ={_rank1['delta_mean_AUC']:.5f}")
print(
    f"超 MT-4: Δ={_rank1['delta_mean_AUC'] - _mt4['delta_mean_AUC']:.5f} "
    f"(第1 {_rank1['delta_mean_AUC']:.5f} vs MT4 {_mt4['delta_mean_AUC']:.5f})"
)

_expected_runs = len(TOP5_SPECS) * len(MODELS) * len(TOP5_VALIDATION_SEEDS)
_checkpoint_done = 0
if TOP5_CHECKPOINT.is_file():
    _checkpoint_done = len(pd.read_csv(TOP5_CHECKPOINT))

if SKIP_TOP5_RUN and TOP5_JSON.is_file() and _checkpoint_done >= _expected_runs:
    print(f'\n跳过 Top-5 跑批（SKIP_TOP5_RUN=True，已有 {_checkpoint_done}/{_expected_runs} 轮）')
    with open(TOP5_JSON, encoding='utf-8') as f:
        top5_result = json.load(f)
    top5_rerun_rank = pd.DataFrame(top5_result['top5_rerun_ranking'])
    if TOP5_DUAL_CSV.is_file():
        top5_dual = pd.read_csv(TOP5_DUAL_CSV)
    else:
        top5_dual = top5_rerun_rank.copy()
    print('\n=== 扩展 seeds 复验 Top-5 排名（自 fe8_top5_validation_result.json）===')
    display(top5_rerun_rank.round(5))
    print('复验优胜:', top5_result.get('rerun_winner_label'))
    print('已保存:', TOP5_JSON)
else:
    print(f'\n扩展 seeds: {TOP5_VALIDATION_SEEDS}')
    print(f'轮次: {len(TOP5_SPECS)}组 × 2模型 × {len(TOP5_VALIDATION_SEEDS)}seeds = {_expected_runs}')

    _rank_map = {r['label']: r['rank_stage2'] for r in TOP5_SPECS}
    top5_raw = run_stability_matrix(TOP5_SPECS, TOP5_CHECKPOINT, seeds=TOP5_VALIDATION_SEEDS)
    top5_raw['rank_stage2'] = top5_raw['特征组合'].map(_rank_map)
    top5_seed_df, top5_dual, top5_rerun_rank = summarize_top5_validation(top5_raw)
    top5_raw.to_csv(TOP5_RAW_CSV, index=False, encoding='utf-8-sig')
    top5_dual.to_csv(TOP5_DUAL_CSV, index=False, encoding='utf-8-sig')

    winner = top5_rerun_rank.iloc[0]
    top5_result = {
        'validation_seeds': TOP5_VALIDATION_SEEDS,
        'stage2_rank1_label': _rank1['combo_label'],
        'stage2_rank1_delta_vs_BASE': float(_rank1['delta_mean_AUC']),
        'stage2_rank1_delta_vs_MT4': float(_rank1['delta_mean_AUC'] - _mt4['delta_mean_AUC']),
        'stage2_mt4_delta_vs_BASE': float(_mt4['delta_mean_AUC']),
        'rerun_winner_label': winner['combo_label'],
        'rerun_winner_delta_vs_BASE': float(winner['delta_mean_vs_BASE']),
        'rerun_winner_delta_vs_MT4': float(winner['delta_mean_vs_MT4']),
        'top5_rerun_ranking': top5_rerun_rank.round(6).to_dict(orient='records'),
    }
    with open(TOP5_JSON, 'w', encoding='utf-8') as f:
        json.dump(top5_result, f, ensure_ascii=False, indent=2)

    print('\n=== 扩展 seeds 复验 Top-5 排名 ===')
    display(top5_rerun_rank.round(5))
    print('复验优胜:', winner['combo_label'])
    print('已保存:', TOP5_JSON)
