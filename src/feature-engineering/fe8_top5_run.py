#!/usr/bin/env python3
"""Run Top-5 extended seed validation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS = ROOT / 'fe8_cells'
ORDER = [
    'cell1_setup.py', 'cell2_data.py', 'cell3_cv.py', 'cell4_df.py',
    'cell5_specs.py', 'cell10_top5_validate.py',
]

def main():
    g = {'__name__': '__main__', 'display': print}
    try:
        from IPython.display import display as ipy_display
        g['display'] = ipy_display
    except ImportError:
        pass
    for name in ORDER:
        print(f'--- exec {name} ---', flush=True)
        code = (CELLS / name).read_text(encoding='utf-8')
        exec(compile(code, str(CELLS / name), 'exec'), g)

if __name__ == '__main__':
    main()
