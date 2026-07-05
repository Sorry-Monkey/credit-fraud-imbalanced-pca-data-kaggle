# --- 0. 环境、路径与实验开关 ---
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from purgedcv import WalkForwardSplit
from purgedcv.diagnostics import assert_no_temporal_leakage
from IPython.display import display

pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 160)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_project_root(start=None):
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / 'input' / 'creditcard.csv').exists():
            return candidate
    raise FileNotFoundError('无法找到 input/creditcard.csv，请确认 notebook 在项目内运行')


PROJECT_ROOT = find_project_root()
FEATURE_DIR = PROJECT_ROOT / 'src' / 'feature-engineering'
DATA_PATH = PROJECT_ROOT / 'input' / 'creditcard.csv'
MT4_CONFIG_PATH = PROJECT_ROOT / 'src' / 'model-training' / 'model_training_if_log1p_atop2_purgedcv_mt4.json'
OUTPUT_PATH = FEATURE_DIR / 'MODEL_FEATURES_V3_contribution_fe8.json'
RESULT_PATH = FEATURE_DIR / 'FE8_CONTRIBUTION_STABILITY_RESULTS.json'

RUN_SEEDS = [42, 2026]
MODELS = ['LightGBM', 'XGBoost']
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42
CV_EMBARGO = pd.Timedelta(hours=2)
CV_PURGE_HORIZON = pd.Timedelta(0)

EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1500
ES_FRAC = 0.25
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
TOP_V_K = 2  # Family A 只保留 A1、A2（A_top2 = 前两列）

IF_RANDOM_STATE = 42
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 0.5
IF_CONTAMINATION = 'auto'
IF_MAX_NORMAL_SAMPLES = 50_000

BASELINE_LABEL = '0_BASE'
MT4_LABEL = 'MT4_IF+hours+log1p+A_top2'

# 阶段 1 已完成：硬编码 Top-3 FE8（按 fe8_stage1_dual_summary 稳定性排序）
# 设为 None 且 SKIP_STAGE1_RUN=False 可重新跑阶段1
MANUAL_FE8_SHORTLIST = [
    ['abs_v14_minus_v10', 'v14_x_log1p_amount', 'v14_x_v10'],
    ['abs_v14_minus_v10', 'v10_x_v4', 'v14_x_log1p_amount', 'v14_x_v10'],
    ['abs_v14_minus_v10', 'v10_x_v4', 'v14_x_log1p_amount'],
]
SKIP_STAGE1_RUN = True
# Top-5 扩展 seeds 复验：已有 fe8_top5_validation_result.json 时跳过跑批
SKIP_TOP5_RUN = True
FE8_TOP_N_MAX = 3

OUTPUT_DIR = FEATURE_DIR / 'output' / 'fe8'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 阶段 1
STAGE1_COMBO_CSV = OUTPUT_DIR / 'fe8_stage1_combo_specs.csv'
STAGE1_COMBO_MD = OUTPUT_DIR / 'fe8_stage1_combo_specs.md'
STAGE1_CHECKPOINT = OUTPUT_DIR / 'fe8_stage1_checkpoint.csv'
STAGE1_RAW_CSV = OUTPUT_DIR / 'fe8_stage1_raw.csv'
STAGE1_DUAL_CSV = OUTPUT_DIR / 'fe8_stage1_dual_summary.csv'
STAGE1_SHORTLIST_JSON = OUTPUT_DIR / 'fe8_stage1_shortlist.json'

# 阶段 2
STAGE2_COMBO_CSV = OUTPUT_DIR / 'fe8_stage2_combo_specs.csv'
STAGE2_COMBO_MD = OUTPUT_DIR / 'fe8_stage2_combo_specs.md'
STAGE2_CHECKPOINT = OUTPUT_DIR / 'fe8_stage2_checkpoint.csv'
STAGE2_RAW_CSV = OUTPUT_DIR / 'fe8_stage2_raw.csv'
STAGE2_SUMMARY_CSV = OUTPUT_DIR / 'fe8_stage2_summary.csv'
STAGE2_DUAL_CSV = OUTPUT_DIR / 'fe8_stage2_dual_summary.csv'

# Top-5 扩展 seeds 复验
TOP5_VALIDATION_SEEDS = [7, 13, 42, 77, 123, 256, 2026, 3141]
TOP5_CHECKPOINT = OUTPUT_DIR / 'fe8_top5_validation_checkpoint.csv'
TOP5_RAW_CSV = OUTPUT_DIR / 'fe8_top5_validation_raw.csv'
TOP5_DUAL_CSV = OUTPUT_DIR / 'fe8_top5_validation_dual_summary.csv'
TOP5_JSON = OUTPUT_DIR / 'fe8_top5_validation_result.json'

print('项目根目录:', PROJECT_ROOT)
print('对比基线:', BASELINE_LABEL)
print('定稿参照:', MT4_LABEL)
print('阶段1: FE8 k=1..5 on BASE | SKIP_STAGE1_RUN:', SKIP_STAGE1_RUN)
print('输出目录:', OUTPUT_DIR)