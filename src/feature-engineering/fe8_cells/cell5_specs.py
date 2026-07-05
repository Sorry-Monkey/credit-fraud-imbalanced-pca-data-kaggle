# --- 4. 组合规格：共享工具 + 阶段1矩阵 ---
from itertools import combinations


def _dedupe(cols):
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _subset_label(cols):
    return '+'.join(sorted(cols)) if cols else 'none'


def _fe8_subsets():
    return [list(s) for k in range(1, len(FE8_NEW_FEATURES) + 1) for s in combinations(FE8_NEW_FEATURES, k)]


# EDA 原子：bands 成对；含 is_micro_testing
EDA_ATOMS = [
    ('log1p', ['log1p_amount']),
    ('hours', ['hours_since_start']),
    ('one_euro', ['is_one_euro']),
    ('micro', ['is_micro_testing']),
    ('bands', list(AMOUNT_BAND_FEATURES)),
]


def iter_eda_subsets():
    for mask in range(2 ** len(EDA_ATOMS)):
        cols, tags = [], []
        for i, (tag, atom_cols) in enumerate(EDA_ATOMS):
            if mask & (1 << i):
                cols.extend(atom_cols)
                tags.append(tag)
        yield _dedupe(cols), '+'.join(tags) if tags else 'none'


def _make_record(records, seen, category, label, extra, stage):
    extra = _dedupe(extra)
    fe8_cols = [c for c in extra if c in FE8_NEW_FEATURES]
    hc_cols = [c for c in extra if c not in FE8_NEW_FEATURES]
    key = (stage, tuple(sorted(extra)))
    if key in seen:
        return
    seen.add(key)
    records.append({
        'combo_id': len(records), 'stage': stage, 'category': category, 'label': label,
        'handcrafted_cols': hc_cols, 'fe8_cols': fe8_cols, 'extra_cols': extra,
        'n_handcrafted': len(hc_cols), 'n_fe8': len(fe8_cols), 'n_extra': len(extra),
    })


def build_stage1_specs():
    records, seen = [], set()
    _make_record(records, seen, 'baseline', BASELINE_LABEL, [], stage=1)
    for fe8_subset in _fe8_subsets():
        k = len(fe8_subset)
        label = f'S1_FE8_k{k}_{_subset_label(fe8_subset)}'
        _make_record(records, seen, 'stage1_fe8_on_base', label, list(fe8_subset), stage=1)
    return records


def build_stage2_specs(selected_fe8_lists):
    records, seen = [], set()
    fe_if = list(FE_IF)
    atop2_short = '+'.join(c.replace('one_euro_', '') for c in A_TOP2)

    _make_record(records, seen, 'baseline', BASELINE_LABEL, [], stage=2)
    _make_record(records, seen, 'mt4_ref', MT4_LABEL, MT4_FINAL_EXTRA, stage=2)

    a_variants = [
        ('+A1', [A1_COL]),
        (f'+A_top2({atop2_short})', A_TOP2),
    ]

    for fe8_subset in selected_fe8_lists:
        fe8_tag = _subset_label(fe8_subset)
        for eda_cols, eda_tag in iter_eda_subsets():
            for a_suffix, a_cols in a_variants:
                label = f'IF+Ed[{eda_tag}]+FE8[{fe8_tag}]{a_suffix}'
                extra = fe_if + eda_cols + list(fe8_subset) + a_cols
                _make_record(records, seen, 'stage2_main', label, extra, stage=2)

    return records


def export_combo_catalog(spec_records, csv_path, md_path, title):
    rows = [{
        'combo_id': r['combo_id'], 'stage': r['stage'], 'category': r['category'], 'label': r['label'],
        'n_handcrafted': r['n_handcrafted'], 'n_fe8': r['n_fe8'], 'n_extra': r['n_extra'],
        'handcrafted_cols': ' | '.join(r['handcrafted_cols']) if r['handcrafted_cols'] else '(none)',
        'fe8_cols': ' | '.join(r['fe8_cols']) if r['fe8_cols'] else '(none)',
        'extra_cols': ' | '.join(r['extra_cols']),
    } for r in spec_records]
    catalog_df = pd.DataFrame(rows)
    catalog_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    lines = [f'# {title}（共 {len(spec_records)} 组）\n\n', f'- 对比基线：{BASELINE_LABEL}\n\n']
    for cat in catalog_df['category'].unique():
        sub = catalog_df[catalog_df['category'] == cat]
        lines.append(f'## {cat}（{len(sub)} 组）\n\n')
        for _, row in sub.iterrows():
            lines.append(f"### [{row['combo_id']}] {row['label']}\n")
            lines.append(f"- 手工：{row['handcrafted_cols']}\n")
            lines.append(f"- FE8：{row['fe8_cols']}\n")
            lines.append(f"- 全部：{row['extra_cols']}\n\n")
    md_path.write_text(''.join(lines), encoding='utf-8')
    return catalog_df


def eval_spec_once(rec, model_name, seed):
    cols = BASE_FEATURES + [c for c in rec['extra_cols'] if c not in BASE_FEATURES]
    missing = [c for c in cols if c not in df_fe8.columns]
    if missing:
        raise KeyError(f"{rec['label']} 缺失列: {missing}")
    res = cross_val_eval(model_name, df_fe8, cols, random_state=seed)
    return {
        'combo_id': rec['combo_id'], 'stage': rec['stage'], 'category': rec['category'],
        '特征组合': rec['label'], '模型': model_name, 'seed': seed,
        'n_handcrafted': rec['n_handcrafted'], 'n_fe8': rec['n_fe8'], 'n_extra': rec['n_extra'],
        'handcrafted_cols': ' | '.join(rec['handcrafted_cols']),
        'fe8_cols': ' | '.join(rec['fe8_cols']),
        'extra_cols': ' | '.join(rec['extra_cols']),
        **res,
    }


def run_stability_matrix(spec_records, checkpoint_path, models=MODELS, seeds=RUN_SEEDS):
    done = {}
    if checkpoint_path.is_file():
        prev = pd.read_csv(checkpoint_path)
        if '特征组合' in prev.columns:
            for _, row in prev.iterrows():
                done[(row['特征组合'], row['模型'], int(row['seed']))] = row.to_dict()
            print(f'checkpoint 恢复 {len(done)} 行 | {checkpoint_path}')

    rows = list(done.values())
    total = len(spec_records) * len(models) * len(seeds)
    step = len(rows)

    for rec in spec_records:
        for model_name in models:
            for seed in seeds:
                key = (rec['label'], model_name, seed)
                if key in done:
                    continue
                step += 1
                print(f'[{step}/{total}] {model_name} seed={seed} | {rec["label"]}', flush=True)
                row = eval_spec_once(rec, model_name, seed)
                rows.append(row)
                done[key] = row
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False, encoding='utf-8-sig')

    return pd.DataFrame(rows)


def summarize_stability(raw: pd.DataFrame, baseline_label=BASELINE_LABEL):
    base = (
        raw[raw['特征组合'] == baseline_label]
        .groupby(['模型', 'seed'], as_index=False)['AUC-PR_mean']
        .mean()
        .rename(columns={'AUC-PR_mean': 'BASELINE_AUC_seed'})
    )
    if base.empty:
        raise ValueError(f'基线 {baseline_label} 无结果')

    merged = raw.merge(base, on=['模型', 'seed'], how='left')
    merged['delta_AUC_vs_BASE'] = merged['AUC-PR_mean'] - merged['BASELINE_AUC_seed']

    summary = (
        merged.groupby(['特征组合', '模型'], as_index=False)
        .agg(
            combo_id=('combo_id', 'first'), stage=('stage', 'first'), category=('category', 'first'),
            n_handcrafted=('n_handcrafted', 'first'), n_fe8=('n_fe8', 'first'), n_extra=('n_extra', 'first'),
            handcrafted_cols=('handcrafted_cols', 'first'), fe8_cols=('fe8_cols', 'first'),
            extra_cols=('extra_cols', 'first'),
            AUC_mean=('AUC-PR_mean', 'mean'), AUC_std_across_seed=('AUC-PR_mean', 'std'),
            delta_mean_AUC=('delta_AUC_vs_BASE', 'mean'), delta_std_AUC=('delta_AUC_vs_BASE', 'std'),
            positive_seed_ratio=('delta_AUC_vs_BASE', lambda s: float((s > 0).mean())),
            F1_mean=('F1@best', 'mean'), FP_mean=('FP', 'mean'), FN_mean=('FN', 'mean'),
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
            'stage': lgb_s.loc[label, 'stage'], 'category': lgb_s.loc[label, 'category'],
            'combo_label': label,
            'n_handcrafted': int(lgb_s.loc[label, 'n_handcrafted']),
            'n_fe8': int(lgb_s.loc[label, 'n_fe8']), 'n_extra': int(lgb_s.loc[label, 'n_extra']),
            'handcrafted_cols': lgb_s.loc[label, 'handcrafted_cols'],
            'fe8_cols': lgb_s.loc[label, 'fe8_cols'], 'extra_cols': lgb_s.loc[label, 'extra_cols'],
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


def parse_fe8_cols(fe8_cols_str):
    if not isinstance(fe8_cols_str, str) or not fe8_cols_str.strip() or fe8_cols_str == '(none)':
        return []
    return [c.strip() for c in fe8_cols_str.split('|')]


def select_fe8_shortlist(dual_summary, top_n=FE8_TOP_N_MAX):
    if MANUAL_FE8_SHORTLIST is not None:
        out = []
        for item in MANUAL_FE8_SHORTLIST:
            cols = list(item) if isinstance(item, (list, tuple)) else [item]
            out.append(_dedupe(cols))
        print('使用 MANUAL_FE8_SHORTLIST:', out)
        return out

    pool = dual_summary[
        (dual_summary['combo_label'] != BASELINE_LABEL) & (dual_summary['n_fe8'] > 0)
    ].copy()

    eligible = pool[
        pool['both_models_positive']
        & (pool['LGB_positive_seed_ratio'] >= 2 / 3)
        & (pool['XGB_positive_seed_ratio'] >= 2 / 3)
    ].sort_values(['conservative_score', 'delta_mean_AUC'], ascending=False)

    if eligible.empty:
        print('阶段1：无满足稳定性门槛的 FE8 组合，回退为 delta_mean_AUC Top-3')
        eligible = pool.sort_values('delta_mean_AUC', ascending=False).head(3)

    shortlisted, seen = [], set()
    for _, row in eligible.iterrows():
        key = tuple(sorted(parse_fe8_cols(row['fe8_cols'])))
        if not key or key in seen:
            continue
        seen.add(key)
        shortlisted.append(list(key))
        if len(shortlisted) >= top_n:
            break

    if not shortlisted:
        shortlisted = [list(FE8_NEW_FEATURES)]
        print('阶段1：shortlist 为空，回退 FE8_ALL')
    return shortlisted


if __name__ == '__main__' or 'get_ipython' in dir():
    STAGE1_RECORDS = build_stage1_specs()
    stage1_catalog_df = export_combo_catalog(
        STAGE1_RECORDS, STAGE1_COMBO_CSV, STAGE1_COMBO_MD, 'FE-8 阶段1：BASE + FE8 子集'
    )
    print(f'阶段1 组合数: {len(STAGE1_RECORDS)} | 运行轮次: {len(STAGE1_RECORDS) * len(MODELS) * len(RUN_SEEDS)}')
    display(stage1_catalog_df)