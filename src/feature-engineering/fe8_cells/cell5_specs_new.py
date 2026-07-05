# --- 4. combo matrix baseline=BASE ---
from itertools import combinations

def _dedupe(cols):
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def _subset_label(cols):
    return '+'.join(sorted(cols)) if cols else 'none'

def _fe8_subsets():
    out = []
    for k in range(1, len(FE8_NEW_FEATURES) + 1):
        for s in combinations(FE8_NEW_FEATURES, k):
            out.append(list(s))
    return out

def build_handcrafted_specs(family_a_cols):
    specs = []
    eda_cols = list(FE_EDA)
    fe_if = list(FE_IF)
    fe_if_gate = list(FE_IF_GATE)
    eda_minus_bands = [c for c in eda_cols if c not in AMOUNT_BAND_FEATURES]
    eda_minus_no_hours = [c for c in eda_minus_bands if c != 'hours_since_start']
    eda_minus_no_hours_log = [c for c in eda_minus_no_hours if c != 'log1p_amount']
    def add(cat, label, extra):
        specs.append((cat, label, _dedupe(extra)))
    add('baseline', BASELINE_LABEL, [])
    add('mt4_ref', MT4_LABEL, MT4_FINAL_EXTRA)
    for i, col in enumerate(eda_cols, start=1):
        add('eda_single', f'Ed{i}. BASE+{col}', [col])
    add('eda_group', 'Ed_all. BASE+all_EDA', eda_cols)
    add('eda_group', 'Ed_noBands', eda_minus_bands)
    add('eda_group', 'Ed_bands', list(AMOUNT_BAND_FEATURES))
    add('eda_group', 'Ed_noHours', eda_minus_no_hours)
    add('eda_group', 'Ed_noHoursLog', eda_minus_no_hours_log)
    add('eda_group', 'Ed_cur', EDA_CURATED)
    add('if', 'IF', fe_if)
    add('if', 'IF+gate', fe_if + fe_if_gate)
    add('eda_if', 'Ed_all+IF', eda_cols + fe_if)
    add('eda_if', 'Ed_cur+IF', EDA_CURATED + fe_if)
    add('eda_if', 'Ed_noHoursLog+IF', eda_minus_no_hours_log + fe_if)
    for col in eda_cols:
        add('eda_if', f'IF+{col}', fe_if + [col])
    for i, col in enumerate(family_a_cols, start=1):
        short = col.replace('one_euro_', '')
        add('family_a_single', f'A{i}. BASE+{col}', [col])
        add('family_a_if', f'A{i}+IF', fe_if + [col])
    for k in (2, 3):
        if k < len(family_a_cols):
            subset = family_a_cols[:k]
            v_short = '+'.join(c.replace('one_euro_', '') for c in subset)
            add('family_a_topk', f'A_top{k}({v_short})', subset)
            add('family_a_topk', f'A_top{k}+IF', fe_if + subset)
    add('family_a_all', 'A_all', family_a_cols)
    add('family_a_all', 'A_all+IF', fe_if + family_a_cols)
    for k in (2, 3):
        if k < len(family_a_cols):
            subset = family_a_cols[:k]
            add('eda_family_a', f'Ed_all+A_top{k}', eda_cols + subset)
            add('eda_family_a', f'Ed_cur+A_top{k}', EDA_CURATED + subset)
            add('eda_if_family_a', f'Ed_all+IF+A_top{k}', eda_cols + fe_if + subset)
            add('eda_if_family_a', f'Ed_cur+IF+A_top{k}', EDA_CURATED + fe_if + subset)
    all_hand = _dedupe(eda_cols + fe_if + family_a_cols + fe_if_gate)
    add('full_handcrafted', 'FULL', all_hand)
    bands = list(AMOUNT_BAND_FEATURES)
    for k in (2, 3):
        if k < len(family_a_cols):
            subset = family_a_cols[:k]
            add('eda_family_a', f'Ed_bands+A_top{k}', bands + subset)
            add('eda_if_family_a', f'Ed_bands+IF+A_top{k}', bands + fe_if + subset)
    return specs

def build_fe8_specs(family_a_cols):
    records, seen = [], set()
    def add(category, label, extra, fe8_cols=None):
        extra = _dedupe(extra)
        if fe8_cols is None:
            fe8_cols = [c for c in extra if c in FE8_NEW_FEATURES]
        hc_cols = [c for c in extra if c not in FE8_NEW_FEATURES]
        key = tuple(sorted(extra))
        if key in seen: return
        seen.add(key)
        records.append({'combo_id': len(records), 'category': category, 'label': label,
            'handcrafted_cols': hc_cols, 'fe8_cols': fe8_cols, 'extra_cols': extra,
            'n_handcrafted': len(hc_cols), 'n_fe8': len(fe8_cols), 'n_extra': len(extra)})
    for cat, label, extra in build_handcrafted_specs(family_a_cols):
        add(cat, label, extra, fe8_cols=[])
    for fe8_subset in _fe8_subsets():
        k = len(fe8_subset)
        add('fe8_on_base', f'FE8_k{k}_BASE+{_subset_label(fe8_subset)}', list(fe8_subset))
    for fe8_subset in _fe8_subsets():
        k = len(fe8_subset)
        add('fe8_on_mt4', f'FE8_k{k}_MT4+{_subset_label(fe8_subset)}', MT4_FINAL_EXTRA + fe8_subset)
    for fe8_subset in _fe8_subsets():
        k = len(fe8_subset)
        add('fe8_on_atop2', f'FE8_k{k}_A_top2+{_subset_label(fe8_subset)}', A_TOP2 + fe8_subset)
    for fe8_subset in _fe8_subsets():
        k = len(fe8_subset)
        add('fe8_on_if_hours_log', f'FE8_k{k}_IFhrslog+{_subset_label(fe8_subset)}', IF_HOURS_LOG + fe8_subset)
    add('fe8_interaction', 'MT4+FE8_ALL', MT4_FINAL_EXTRA + FE8_NEW_FEATURES)
    for drop_col in MT4_FINAL_EXTRA:
        remain = [c for c in MT4_FINAL_EXTRA if c != drop_col]
        add('fe8_interaction', f'MT4-DROP_{drop_col}+FE8_ALL', remain + FE8_NEW_FEATURES)
    add('fe8_interaction', 'A_top2+FE8_ALL', A_TOP2 + FE8_NEW_FEATURES)
    for drop_col in A_TOP2:
        remain = [c for c in A_TOP2 if c != drop_col]
        add('fe8_interaction', f'A_top2-DROP_{drop_col}+FE8_ALL', remain + FE8_NEW_FEATURES)
    for f in FE8_NEW_FEATURES:
        add('fe8_sparse', f'IF+{f}', FE_IF + [f])
        add('fe8_sparse', f'Ed_all+{f}', FE_EDA + [f])
        add('fe8_sparse', f'A_top2+{f}', A_TOP2 + [f])
    return records

def export_combo_catalog(spec_records):
    rows = [{'combo_id': r['combo_id'], 'category': r['category'], 'label': r['label'],
             'n_handcrafted': r['n_handcrafted'], 'n_fe8': r['n_fe8'], 'n_extra': r['n_extra'],
             'handcrafted_cols': ' | '.join(r['handcrafted_cols']) if r['handcrafted_cols'] else '(none)',
             'fe8_cols': ' | '.join(r['fe8_cols']) if r['fe8_cols'] else '(none)',
             'extra_cols': ' | '.join(r['extra_cols'])} for r in spec_records]
    catalog_df = pd.DataFrame(rows)
    catalog_df.to_csv(COMBO_CSV_PATH, index=False, encoding='utf-8-sig')
    lines = [f'# FE-8 catalog ({len(spec_records)} groups)\n\n', f'- baseline: {BASELINE_LABEL}\n', f'- MT4 ref: {MT4_LABEL}\n\n']
    for cat in catalog_df['category'].unique():
        sub = catalog_df[catalog_df['category']==cat]
        lines.append(f'## {cat} ({len(sub)})\n\n')
        for _, row in sub.iterrows():
            lines.append(f"### [{row['combo_id']}] {row['label']}\n- hc: {row['handcrafted_cols']}\n- fe8: {row['fe8_cols']}\n- all: {row['extra_cols']}\n\n")
    COMBO_MD_PATH.write_text(''.join(lines), encoding='utf-8')
    return catalog_df

def eval_spec_once(rec, model_name, seed):
    cols = BASE_FEATURES + [c for c in rec['extra_cols'] if c not in BASE_FEATURES]
    missing = [c for c in cols if c not in df_fe8.columns]
    if missing: raise KeyError(f"{rec['label']} missing {missing}")
    res = cross_val_eval(model_name, df_fe8, cols, random_state=seed)
    return {'combo_id': rec['combo_id'], 'category': rec['category'], '特征组合': rec['label'], '模型': model_name,
            'seed': seed, 'n_handcrafted': rec['n_handcrafted'], 'n_fe8': rec['n_fe8'], 'n_extra': rec['n_extra'],
            'handcrafted_cols': ' | '.join(rec['handcrafted_cols']), 'fe8_cols': ' | '.join(rec['fe8_cols']),
            'extra_cols': ' | '.join(rec['extra_cols']), **res}

def run_stability_matrix(spec_records, models=MODELS, seeds=RUN_SEEDS, checkpoint_path=CHECKPOINT_PATH):
    done = {}
    if checkpoint_path.is_file():
        prev = pd.read_csv(checkpoint_path)
        if '特征组合' in prev.columns:
            for _, row in prev.iterrows():
                done[(row['特征组合'], row['模型'], int(row['seed']))] = row.to_dict()
            print(f'checkpoint restored {len(done)} rows; delete if matrix changed: {checkpoint_path}')
    rows = list(done.values()); total = len(spec_records)*len(models)*len(seeds); step = len(rows)
    for rec in spec_records:
        for model_name in models:
            for seed in seeds:
                key = (rec['label'], model_name, seed)
                if key in done: continue
                step += 1
                print(f'[{step}/{total}] {model_name} seed={seed} | {rec["label"]}', flush=True)
                row = eval_spec_once(rec, model_name, seed)
                rows.append(row); done[key] = row
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
    return pd.DataFrame(rows)

SPEC_RECORDS_FE8 = build_fe8_specs(CROSS_FAMILY_A)
combo_catalog_df = export_combo_catalog(SPEC_RECORDS_FE8)
print(f'combos={len(SPEC_RECORDS_FE8)} runs={len(SPEC_RECORDS_FE8)*len(MODELS)*len(RUN_SEEDS)}')
print(COMBO_CSV_PATH, COMBO_MD_PATH)
display(combo_catalog_df.groupby('category').size().rename('count'))
