# --- 1. 数据读取与特征块定义 ---
def read_creditcard_csv(path: Path) -> pd.DataFrame:
    for kwargs in (
        {'encoding': 'utf-8'},
        {'encoding': 'utf-8', 'encoding_errors': 'replace'},
        {'encoding': 'latin-1'},
    ):
        try:
            return pd.read_csv(path, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError('utf-8', b'', 0, 1, 'failed to decode creditcard.csv')


def build_eda_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out['log1p_amount'] = np.log1p(out['Amount'])
    out['hours_since_start'] = (out['Time'] // 3600).astype(int)
    out['is_micro_testing'] = out['Amount'] < 1
    out['is_one_euro'] = out['Amount'] == 1.0
    out['is_amount_1_30'] = (out['Amount'] > 1) & (out['Amount'] <= 30)
    out['is_amount_75_110'] = (out['Amount'] >= 75) & (out['Amount'] <= 110)
    return out


def build_cross_features(data: pd.DataFrame, top_v: list, gate_col='is_one_euro', prefix='one_euro'):
    out = data.copy()
    gate = out[gate_col].astype(float)
    new_cols = []
    for v in top_v:
        name = f'{prefix}_{v}'
        out[name] = gate * out[v]
        new_cols.append(name)
    return out, new_cols


def build_fe8_contribution_features(data: pd.DataFrame):
    out = data.copy()
    out['v14_x_v10'] = out['V14'] * out['V10']
    out['v14_x_v4'] = out['V14'] * out['V4']
    out['v10_x_v4'] = out['V10'] * out['V4']
    out['abs_v14_minus_v10'] = (out['V14'] - out['V10']).abs()
    out['v14_x_log1p_amount'] = out['V14'] * out['log1p_amount']
    feats = ['v14_x_v10', 'v14_x_v4', 'v10_x_v4', 'abs_v14_minus_v10', 'v14_x_log1p_amount']
    return out, feats


def pick_top_v_features(data, feature_cols, k=TOP_V_K, model_name='LightGBM', random_state=42):
    data = sort_by_time(data)
    n = len(data)
    test_size = max(1, n // (CV_N_SPLITS + 1))
    tr_idx = np.arange(0, n - test_size)
    X_tr, y_tr = data.iloc[tr_idx][feature_cols], data.iloc[tr_idx]['Class']
    X_fit, X_es, y_fit, y_es = split_early_stop_set(X_tr, y_tr, random_state=random_state)
    clf = make_classifier(model_name, y_fit, random_state=random_state)
    fit_classifier(clf, model_name, X_fit, y_fit, X_es, y_es)
    imp = pd.Series(clf.feature_importances_, index=feature_cols)
    top_v = list(imp[[c for c in feature_cols if c.startswith('V')]].sort_values(ascending=False).head(k).index)
    print(f'Top-{k} V ({model_name} gain): {top_v}')
    return top_v


df_raw = read_creditcard_csv(DATA_PATH)
V_COLS = [c for c in df_raw.columns if c.startswith('V')]
BASE_FEATURES = V_COLS + ['Amount', 'Time']

FE_EDA = ['log1p_amount', 'hours_since_start', 'is_micro_testing', 'is_one_euro', 'is_amount_1_30', 'is_amount_75_110']
AMOUNT_BAND_FEATURES = ['is_amount_1_30', 'is_amount_75_110']
EDA_CURATED = ['hours_since_start', 'is_one_euro']
FE_IF = ['if_oof_score']
FE_IF_GATE = ['if_oof_score_x_one_euro']
A_TOP2 = ['one_euro_V14', 'one_euro_V4']
MT4_FINAL_EXTRA = ['if_oof_score', 'hours_since_start', 'log1p_amount'] + A_TOP2
IF_HOURS_LOG = ['if_oof_score', 'hours_since_start', 'log1p_amount']

with open(MT4_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _mt4 = json.load(f)
_mt4_extra = [c for c in _mt4['MODEL_FEATURES'] if c not in BASE_FEATURES]
assert MT4_FINAL_EXTRA == _mt4_extra, f'MT4 mismatch: {_mt4_extra}'

print(f'行数: {len(df_raw):,} | 欺诈: {int(df_raw["Class"].sum())} | 欺诈率: {df_raw["Class"].mean():.4f}')
print('MT-4 定稿增量:', MT4_FINAL_EXTRA)