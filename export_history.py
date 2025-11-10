import os
import pandas as pd
from utils import ensure_dirs, DATA_DIR, EXPORT_DIR, load_json, today_str

ensure_dirs()
HIST_PATH = os.path.join(DATA_DIR, 'history.json')

def export():
    history = load_json(HIST_PATH, [])
    if not history:
        print('Không có lịch sử để xuất.')
        return
    df = pd.DataFrame(history)
    out_path = os.path.join(EXPORT_DIR, f'history_{today_str()}.xlsx')
    df.to_excel(out_path, index=False)
    print('Đã xuất file:', out_path)

if __name__ == '__main__':
    export()
