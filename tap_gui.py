"""
tap_gui.py
Tkinter GUI for tap onset detection (Fujii 10% method).

- 複数 WAV ファイルの一括選択
- 各ファイルごとに:
    - tap_start / tap_peak の検出 (tap_core.detect_taps_from_wav)
    - 結果をテキストエリアに表示
    - tap_plot.plot_tap_detection_interactive でインタラクティブ表示
- Export はプロット側の「Export」ボタンで実行
"""

from __future__ import annotations

from typing import List
import os
import tkinter as tk
from tkinter import filedialog, ttk, TclError

from .tap_core import detect_taps_from_wav
from .tap_plot import plot_tap_detection_interactive


# デフォルトパラメータ（研究用途として固定）
DEFAULT_HPF = 100.0           # Hz
DEFAULT_THRESHOLD_RATIO = 0.1  # 10%
DEFAULT_AMP_THRESHOLD_RATIO = 0.03  # 3%
DEFAULT_PEAK_AMP_RATIO = 0.0  # 0% (無効)
DEFAULT_MIN_DISTANCE_MS = 100.0  # ms


class TapDetectorGUI:
    """Tap onset detection GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tap Onset Detection (Fujii 10% method)")
        self.root.geometry("800x600")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # タイトル
        title_label = ttk.Label(
            main_frame,
            text="Tap Onset Detection (Fujii 10% method)",
            font=("Helvetica", 16, "bold"),
        )
        title_label.grid(row=0, column=0, pady=20)

        # 説明
        peak_amp_desc = f"{int(DEFAULT_PEAK_AMP_RATIO * 100)}%" if DEFAULT_PEAK_AMP_RATIO > 0 else "OFF"
        desc = (
            "Select one or more WAV files.\n"
            "For each file, tap onsets are detected using:\n"
            f"  - HPF = {DEFAULT_HPF:.0f} Hz (Butterworth + filtfilt, zero-phase)\n"
            f"  - Threshold = {int(DEFAULT_THRESHOLD_RATIO * 100)}% of envelope peak\n"
            f"  - Amp Threshold = {int(DEFAULT_AMP_THRESHOLD_RATIO * 100)}% of max amplitude (lower limit)\n"
            f"  - Peak Amp = {peak_amp_desc} of max amplitude (upper limit)\n"
            f"  - Min distance between peaks = {DEFAULT_MIN_DISTANCE_MS:.0f} ms\n\n"
            "Results are shown below and an interactive plot window opens.\n"
            "Zoom / pan / re-detect / export are controlled in the plot window."
        )
        desc_label = ttk.Label(
            main_frame,
            text=desc,
            wraplength=740,
            justify="left",
        )
        desc_label.grid(row=1, column=0, pady=10)

        # ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=10)

        self.detect_button = ttk.Button(
            button_frame,
            text="Select WAV file(s) and detect",
            command=self.detect_taps_for_files,
            width=40,
        )
        self.detect_button.grid(row=0, column=0, padx=10, pady=5)

        # ステータス
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            font=("Helvetica", 10),
            foreground="green",
        )
        self.status_label.grid(row=3, column=0, pady=10, sticky=tk.W)

        # 結果表示エリア
        result_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.result_text = tk.Text(result_frame, height=15, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(
            result_frame,
            orient=tk.VERTICAL,
            command=self.result_text.yview,
        )
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text["yscrollcommand"] = scrollbar.set

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------
    def update_status(self, message: str, color: str = "black") -> None:
        """ステータスラベル更新。ウィンドウ破棄後は何もしない。"""
        try:
            self.status_label.config(text=message, foreground=color)
            self.root.update_idletasks()
        except TclError:
            # ウィンドウが既に閉じている場合は無視
            pass

    def append_result(self, text: str) -> None:
        """結果テキストを追記。ウィンドウ破棄後は何もしない。"""
        try:
            self.result_text.insert(tk.END, text + "\n")
            self.result_text.see(tk.END)
            self.root.update_idletasks()
        except TclError:
            pass

    def clear_results(self) -> None:
        """結果テキストをクリア。ウィンドウ破棄後は何もしない。"""
        try:
            self.result_text.delete(1.0, tk.END)
        except TclError:
            pass

    # ------------------------------------------------------------------
    # メイン処理
    # ------------------------------------------------------------------
    def detect_taps_for_files(self) -> None:
        """ファイル選択 → 各ファイルごとに検出 & プロット。"""
        self.clear_results()
        self.update_status("Select WAV file(s) for tap detection...", "blue")

        wav_paths = filedialog.askopenfilenames(
            title="Select WAV file(s) for tap detection",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )

        if not wav_paths:
            self.update_status("Selection cancelled.", "orange")
            return

        wav_paths = list(wav_paths)
        n_files = len(wav_paths)
        self.append_result(f"{n_files} file(s) selected.")
        self.append_result("")

        for idx, path in enumerate(wav_paths, start=1):
            basename = os.path.basename(path)
            self.clear_results()
            self.append_result(
                f"[{idx}/{n_files}] Processing file: {basename}"
            )
            self.append_result("=" * 60)
            self.update_status(f"Processing {idx}/{n_files}...", "blue")

            try:
                taps = detect_taps_from_wav(
                    path,
                    hp_cutoff=DEFAULT_HPF,
                    threshold_ratio=DEFAULT_THRESHOLD_RATIO,
                    min_distance_ms=DEFAULT_MIN_DISTANCE_MS,
                )

                if not taps:
                    self.append_result("No taps detected.")
                else:
                    self.append_result(
                        f"Detected {len(taps)} taps (tap_start / tap_peak in seconds):"
                    )
                    self.append_result("")
                    for j, t in enumerate(taps, start=1):
                        self.append_result(
                            f"  {j:3d}. tap_start={t['tap_start']:.4f} s, "
                            f"tap_peak={t['tap_peak']:.4f} s"
                        )

                self.append_result("")
                self.append_result("Opening interactive plot window...")
                self.append_result(
                    "  - Use sliders to adjust HPF and threshold\n"
                    "  - Re-detect keeps current zoom\n"
                    "  - Shift+Click near a marker deletes that tap\n"
                    "  - Export button in the plot window writes CSV"
                )

                # インタラクティブプロット（この呼び出し中はブロッキング）
                plot_tap_detection_interactive(
                    path,
                    initial_hp_cutoff_hz=DEFAULT_HPF,
                    threshold_ratio=DEFAULT_THRESHOLD_RATIO,
                    amp_threshold_ratio=DEFAULT_AMP_THRESHOLD_RATIO,
                    peak_amp_ratio=DEFAULT_PEAK_AMP_RATIO,
                    min_distance_ms=DEFAULT_MIN_DISTANCE_MS,
                    title=(
                        f"Tap Onset Detection (Fujii) - {basename} "
                        f"({idx}/{n_files})"
                    ),
                    on_next_callback=None,
                    enable_export=True,
                )

                self.update_status(
                    f"Finished {idx}/{n_files}: {basename}", "green"
                )
                self.append_result("")
                self.append_result("Plot window closed.")
                self.append_result("=" * 60)

            except Exception as e:
                msg = f"Error during tap detection: {e}"
                self.append_result("")
                self.append_result(msg)
                self.update_status("Error occurred.", "red")

        self.update_status("All files processed.", "green")


def main() -> None:
    """GUI エントリポイント。"""
    root = tk.Tk()
    app = TapDetectorGUI(root)
    root.mainloop()
