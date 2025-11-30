# tap_detector

タップ音のオンセット検出ツールキット（Fujii 10% 法）

WAV ファイルからタップ（打鍵・タッピング）のオンセット（開始時刻）とピーク時刻を検出するための Python ツールキットです。

## 特徴

- **Fujii 10% 法** によるオンセット検出
  - Butterworth ハイパスフィルタ（ゼロ位相）
  - Hilbert 変換による包絡線抽出
  - 包絡線ピークから逆方向に 10% 閾値を超える点を探索
- **2段階フィルタリング**
  - 包絡線振幅によるフィルタリング（`threshold_ratio`）
  - 波形振幅によるフィルタリング（`amp_threshold_ratio`）
- **インタラクティブ GUI**
  - 複数 WAV ファイルの一括処理
  - リアルタイムパラメータ調整（HPF カットオフ、閾値、スムージング）
  - ズーム・パン対応のプロット表示
  - Shift+クリックでタップを手動削除
  - CSV エクスポート

## インストール

```bash
pip install -e .
```

または依存パッケージのみインストール:

```bash
pip install -r requirements.txt
```

## 使い方

### GUI モード

```bash
python -m tap_detector
```

ファイル選択ダイアログから WAV ファイルを選択すると、タップ検出とインタラクティブプロットが表示されます。

### Python API

```python
from tap_detector import detect_taps_from_wav

# WAV ファイルからタップを検出
taps = detect_taps_from_wav(
    "path/to/audio.wav",
    hp_cutoff=300.0,           # ハイパスフィルタ カットオフ [Hz]
    threshold_ratio=0.1,        # 包絡線閾値（最大値の 10%）
    amp_threshold_ratio=0.03,   # 波形振幅閾値（最大値の 3%）
    min_distance_ms=100.0,      # ピーク間最小距離 [ms]
    smooth_ms=0.3,              # 包絡線スムージング窓 [ms]
)

# 結果を表示
for tap in taps:
    print(f"onset: {tap['tap_start']:.4f}s, peak: {tap['tap_peak']:.4f}s")
```

### 配列から直接検出

```python
import numpy as np
import soundfile as sf
from tap_detector import detect_tap_onsets_and_peaks

# 音声データを読み込み
y, sr = sf.read("path/to/audio.wav")
if y.ndim > 1:
    y = np.mean(y, axis=1)  # ステレオをモノラルに変換

# タップを検出
taps = detect_tap_onsets_and_peaks(
    y, sr,
    hp_cutoff=300.0,
    threshold_ratio=0.1,
)
```

## API リファレンス

### `detect_taps_from_wav(wav_path, **kwargs) -> List[Dict]`

WAV ファイルからタップを検出します。

**引数:**
- `wav_path`: WAV ファイルパス
- `hp_cutoff`: ハイパスフィルタ カットオフ周波数 [Hz]（デフォルト: 300.0）
- `threshold_ratio`: 包絡線閾値比率（デフォルト: 0.1）
- `amp_threshold_ratio`: 波形振幅閾値比率（デフォルト: 0.03）
- `min_distance_ms`: ピーク間最小距離 [ms]（デフォルト: 100.0）
- `smooth_ms`: 包絡線スムージング窓 [ms]（デフォルト: 0.3）

**戻り値:** 検出されたタップの辞書リスト

### `detect_tap_onsets_and_peaks(y, sr, **kwargs) -> List[Dict]`

配列データからタップを検出します。引数は `detect_taps_from_wav` と同様（`wav_path` の代わりに `y` と `sr`）。

## ファイル構成

```
.
├── __init__.py       # パッケージ初期化、API エクスポート
├── __main__.py       # CLI エントリポイント
├── tap_core.py       # コア検出ロジック（Fujii 10% 法）
├── tap_utils.py      # DSP ユーティリティ（HPF, Hilbert包絡線）
├── tap_plot.py       # インタラクティブプロット
├── tap_export.py     # CSV エクスポート
├── tap_gui.py        # Tkinter GUI
├── requirements.txt  # 依存パッケージ
├── pyproject.toml    # パッケージ設定
└── README.md         # このファイル
```

## 依存パッケージ

- numpy
- scipy
- soundfile
- matplotlib

## ライセンス

MIT License
