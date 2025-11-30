"""
tap_plot.py
Interactive visualization for tap detection (no spectrogram).

- 上段: HPF 後波形 + tap_start / tap_peak
- 下段: Hilbert envelope + tap_start / tap_peak
- HPF / Threshold / Smooth(ms) のスライダ
- Re-detect ボタン（ズーム維持）
- Export / Next ボタン
- Shift+Click で最近傍の tap (start/peak ペア) を削除
"""

from __future__ import annotations
from typing import Callable, List, Dict, Tuple

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from .tap_utils import butter_highpass_zero_phase, hilbert_envelope
from .tap_export import export_taps_to_csv
from .tap_core import detect_tap_onsets_and_peaks


def plot_tap_detection_interactive(
    wav_path: str,
    *,
    initial_hp_cutoff_hz: float = 300.0,
    threshold_ratio: float = 0.1,
    amp_threshold_ratio: float = 0.03,
    peak_amp_ratio: float = 0.0,
    min_distance_ms: float = 100.0,
    smooth_ms: float = 0.3,
    title: str | None = None,
    on_next_callback: Callable[[], None] | None = None,
    enable_export: bool = True,
) -> None:
    """
    タップ検出結果をインタラクティブに表示する。

    Args:
        wav_path: WAV ファイルパス
        initial_hp_cutoff_hz: HPF カットオフ初期値 [Hz]
        threshold_ratio: envelope threshold ratio (env >= ratio * env_max)
        amp_threshold_ratio: waveform amplitude filter (ratio of max_abs)
        peak_amp_ratio: only detect taps with local max >= this ratio of global max
        min_distance_ms: peak 間最小距離 [ms]
        smooth_ms: Hilbert smoothing window [ms]
        title: グラフタイトル
        on_next_callback: Next ボタン押下時に呼ばれるコールバック
        enable_export: Export ボタンを使うかどうか
    """

    # ========= 内部ヘルパ関数 ==============================================

    def run_detection(
        y: np.ndarray,
        sr: int,
        hp: float,
        thr: float,
        amp_thr: float,
        peak_amp: float,
        smooth: float,
    ) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
        """指定パラメータで tap detection を実行し、taps / y_filt / env を返す。"""
        taps_local = detect_tap_onsets_and_peaks(
            y,
            sr,
            hp_cutoff=hp,
            threshold_ratio=thr,
            amp_threshold_ratio=amp_thr,
            peak_amp_ratio=peak_amp,
            min_distance_ms=min_distance_ms,
            smooth_ms=smooth,
        )
        y_f = butter_highpass_zero_phase(y, sr, hp)
        env_f = hilbert_envelope(y_f, sr, smooth_ms=smooth)
        return taps_local, y_f, env_f

    def update_plot(
        y_new: np.ndarray,
        env_new: np.ndarray,
        taps_new: List[Dict[str, float]],
        *,
        preserve_xlim: bool = True,
    ) -> None:
        """波形・包絡・マーカーを更新。必要ならズーム維持。"""
        nonlocal onset_lines_ax1, peak_lines_ax1, onset_lines_ax2, peak_lines_ax2
        nonlocal peak_amp_line_pos, peak_amp_line_neg

        if preserve_xlim:
            xlim1 = ax1.get_xlim()
            xlim2 = ax2.get_xlim()
            ylim1 = ax1.get_ylim()
            ylim2 = ax2.get_ylim()

        waveform_line.set_ydata(y_new)
        envelope_line.set_ydata(env_new)

        # 閾値ラインを更新
        new_max_amp = float(np.max(np.abs(y_new)))
        new_amp_thr_val = state["amp_thr"] * new_max_amp
        amp_thr_line_pos.set_ydata([new_amp_thr_val, new_amp_thr_val])
        amp_thr_line_neg.set_ydata([-new_amp_thr_val, -new_amp_thr_val])

        # Peak Amp 閾値ラインを更新
        new_peak_amp_val = state["peak_amp"] * new_max_amp
        if state["peak_amp"] > 0:
            if peak_amp_line_pos is None:
                # 新規作成
                peak_amp_line_pos = ax1.axhline(
                    y=new_peak_amp_val, color="lime", linestyle="--", alpha=0.7, linewidth=1.5,
                    label=f"Peak Amp ({state['peak_amp']*100:.0f}%)"
                )
                peak_amp_line_neg = ax1.axhline(
                    y=-new_peak_amp_val, color="lime", linestyle="--", alpha=0.7, linewidth=1.5
                )
            else:
                peak_amp_line_pos.set_ydata([new_peak_amp_val, new_peak_amp_val])
                peak_amp_line_neg.set_ydata([-new_peak_amp_val, -new_peak_amp_val])
                peak_amp_line_pos.set_label(f"Peak Amp ({state['peak_amp']*100:.0f}%)")
        else:
            # 0% の場合は非表示
            if peak_amp_line_pos is not None:
                peak_amp_line_pos.remove()
                peak_amp_line_neg.remove()
                peak_amp_line_pos = None
                peak_amp_line_neg = None

        new_env_max = float(np.max(env_new))
        new_env_thr_val = state["thr"] * new_env_max
        env_thr_line.set_ydata([new_env_thr_val, new_env_thr_val])

        # マーカー削除
        for line in onset_lines_ax1 + peak_lines_ax1 + onset_lines_ax2 + peak_lines_ax2:
            line.remove()
        onset_lines_ax1.clear()
        peak_lines_ax1.clear()
        onset_lines_ax2.clear()
        peak_lines_ax2.clear()

        # 新しいマーカー描画
        for t in taps_new:
            ts = t["tap_start"]
            tp = t["tap_peak"]

            onset_lines_ax1.append(
                ax1.axvline(
                    x=ts, color="green", linestyle="--", alpha=0.7, linewidth=1.0
                )
            )
            peak_lines_ax1.append(
                ax1.axvline(
                    x=tp, color="red", linestyle=":", alpha=0.7, linewidth=1.0
                )
            )
            onset_lines_ax2.append(
                ax2.axvline(
                    x=ts, color="green", linestyle="--", alpha=0.7, linewidth=1.0
                )
            )
            peak_lines_ax2.append(
                ax2.axvline(
                    x=tp, color="red", linestyle=":", alpha=0.7, linewidth=1.0
                )
            )

        onset_count_text.set_text(f"Onsets detected: {len(taps_new)}")

        # ラベル更新
        env_label_new = (
            f"Envelope (HPF={state['hp']:.0f}Hz, smooth={state['smooth_ms']:.1f}ms)"
        )
        legend.texts[0].set_text(env_label_new)

        # 閾値ラインのラベルも更新
        amp_thr_line_pos.set_label(f"Amp Thr ({state['amp_thr']*100:.1f}%)")
        env_thr_line.set_label(f"Env Thr ({state['thr']*100:.0f}%)")
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")

        if preserve_xlim:
            ax1.set_xlim(xlim1)
            ax2.set_xlim(xlim2)
            ax1.set_ylim(ylim1)
            ax2.set_ylim(ylim2)
        else:
            ax1.relim()
            ax1.autoscale_view(scalex=False, scaley=True)
            ax2.relim()
            ax2.autoscale_view(scalex=False, scaley=True)

        fig.canvas.draw_idle()

    def is_ui_axis(ax) -> bool:
        """スライダやボタンなど UI 用 axes かどうか判定。"""
        if ax is None:
            return False
        return ax in ui_axes

    # ========= WAV 読み込み & 初回検出 =====================================

    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    state: Dict[str, float | List[Dict[str, float]]] = {
        "hp": float(initial_hp_cutoff_hz),
        "thr": float(threshold_ratio),
        "amp_thr": float(amp_threshold_ratio),
        "peak_amp": float(peak_amp_ratio),
        "smooth_ms": float(smooth_ms),
        "taps": [],
    }

    taps, y_filt, env = run_detection(
        y, sr, state["hp"], state["thr"], state["amp_thr"], state["peak_amp"], state["smooth_ms"]
    )
    state["taps"] = taps

    time_axis = np.arange(len(y_filt)) / sr

    # ========= Figure 準備 ====================================================

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, sharex=ax1)

    # 上段: 波形
    waveform_line, = ax1.plot(
        time_axis,
        y_filt,
        alpha=0.9,
        linewidth=1.0,
        color="blue",
        label="Waveform (HPF)",
    )
    ax1.set_ylabel("Amplitude")
    ax1.set_title(
        title
        or f"Tap Onset Detection (Fujii) - {os.path.basename(wav_path)}"
    )
    ax1.grid(True, alpha=0.3)

    # 振幅閾値ライン（正負両方）
    max_amp = float(np.max(np.abs(y_filt)))
    amp_thr_val = state["amp_thr"] * max_amp
    amp_thr_line_pos = ax1.axhline(
        y=amp_thr_val, color="magenta", linestyle="-.", alpha=0.6, linewidth=1.0,
        label=f"Amp Thr ({state['amp_thr']*100:.1f}%)"
    )
    amp_thr_line_neg = ax1.axhline(
        y=-amp_thr_val, color="magenta", linestyle="-.", alpha=0.6, linewidth=1.0
    )

    # Peak Amp 閾値ライン（上限フィルタ）
    peak_amp_val = state["peak_amp"] * max_amp
    if state["peak_amp"] > 0:
        peak_amp_line_pos = ax1.axhline(
            y=peak_amp_val, color="lime", linestyle="--", alpha=0.7, linewidth=1.5,
            label=f"Peak Amp ({state['peak_amp']*100:.0f}%)"
        )
        peak_amp_line_neg = ax1.axhline(
            y=-peak_amp_val, color="lime", linestyle="--", alpha=0.7, linewidth=1.5
        )
    else:
        peak_amp_line_pos = None
        peak_amp_line_neg = None

    onset_lines_ax1: List[plt.Line2D] = []
    peak_lines_ax1: List[plt.Line2D] = []
    for i, t in enumerate(taps):
        ts = t["tap_start"]
        tp = t["tap_peak"]
        onset_lines_ax1.append(
            ax1.axvline(
                x=ts,
                color="green",
                linestyle="--",
                alpha=0.7,
                linewidth=1.0,
                label="tap_start" if i == 0 else None,
            )
        )
        peak_lines_ax1.append(
            ax1.axvline(
                x=tp,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1.0,
                label="tap_peak" if i == 0 else None,
            )
        )
    ax1.legend(loc="upper right")

    # 下段: Envelope
    env_label = (
        f"Envelope (HPF={state['hp']:.0f}Hz, smooth={state['smooth_ms']:.1f}ms)"
    )
    envelope_line, = ax2.plot(
        time_axis,
        env,
        label=env_label,
        linewidth=1.5,
        color="orange",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Envelope")
    ax2.grid(True, alpha=0.3)

    # エンベロープ閾値ライン
    env_max = float(np.max(env))
    env_thr_val = state["thr"] * env_max
    env_thr_line = ax2.axhline(
        y=env_thr_val, color="cyan", linestyle="-.", alpha=0.6, linewidth=1.0,
        label=f"Env Thr ({state['thr']*100:.0f}%)"
    )

    legend = ax2.legend(loc="upper right")

    onset_lines_ax2: List[plt.Line2D] = []
    peak_lines_ax2: List[plt.Line2D] = []
    for t in taps:
        ts = t["tap_start"]
        tp = t["tap_peak"]
        onset_lines_ax2.append(
            ax2.axvline(x=ts, color="green", linestyle="--", alpha=0.7, linewidth=1.0)
        )
        peak_lines_ax2.append(
            ax2.axvline(x=tp, color="red", linestyle=":", alpha=0.7, linewidth=1.0)
        )

    onset_count_text = ax1.text(
        0.02,
        0.98,
        f"Onsets detected: {len(taps)}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # ========= スライダ & ボタン配置 ========================================

    plt.subplots_adjust(bottom=0.42)

    # スライダ: 5 段（中央寄せ、幅 0.70）
    ax_slider_hpf = plt.axes([0.15, 0.34, 0.70, 0.025])
    slider_hpf = Slider(
        ax_slider_hpf,
        "HPF (Hz)",
        100.0,
        2000.0,
        valinit=state["hp"],
        valstep=50.0,
    )

    ax_slider_threshold = plt.axes([0.15, 0.29, 0.70, 0.025])
    slider_threshold = Slider(
        ax_slider_threshold,
        "Threshold (%)",
        1.0,
        50.0,
        valinit=state["thr"] * 100.0,
        valstep=1.0,
    )

    ax_slider_amp_threshold = plt.axes([0.15, 0.24, 0.70, 0.025])
    slider_amp_threshold = Slider(
        ax_slider_amp_threshold,
        "Amp Thr (%)",
        1.0,
        20.0,
        valinit=state["amp_thr"] * 100.0,
        valstep=0.5,
    )

    ax_slider_peak_amp = plt.axes([0.15, 0.19, 0.70, 0.025])
    slider_peak_amp = Slider(
        ax_slider_peak_amp,
        "Peak Amp (%)",
        0.0,
        100.0,
        valinit=state["peak_amp"] * 100.0,
        valstep=1.0,
    )

    ax_slider_smooth = plt.axes([0.15, 0.14, 0.70, 0.025])
    slider_smooth = Slider(
        ax_slider_smooth,
        "Smooth (ms)",
        0.3,
        2.0,
        valinit=state["smooth_ms"],
        valstep=0.1,
    )

    # Slider tooltips (English)
    slider_tooltips = {
        ax_slider_hpf: "HPF (Hz): High-pass filter cutoff frequency. Removes low-frequency noise.",
        ax_slider_threshold: "Threshold (%): Envelope threshold. Only detects peaks above this ratio of max envelope.",
        ax_slider_amp_threshold: "Amp Thr (%): Lower amplitude threshold. Excludes noise below this ratio of max amplitude.",
        ax_slider_peak_amp: "Peak Amp (%): Upper amplitude threshold. Only detects taps with amplitude above this ratio. 0% disables.",
        ax_slider_smooth: "Smooth (ms): Hilbert envelope smoothing window. Larger values produce smoother envelopes.",
    }

    # ツールチップ表示用のテキスト
    tooltip_text = fig.text(
        0.5, 0.005, "", ha="center", va="bottom",
        fontsize=9, color="gray", style="italic"
    )

    # ボタン列
    button_width = 0.10
    button_height = 0.04
    button_y = 0.08

    ax_redetect = plt.axes([0.15, button_y, button_width, button_height])
    button_redetect = Button(ax_redetect, "Re-detect")

    current_pos = 0.30

    if enable_export:
        ax_export = plt.axes([current_pos, button_y, button_width, button_height])
        button_export = Button(ax_export, "Export")
        current_pos += button_width + 0.02
    else:
        ax_export = None
        button_export = None

    if on_next_callback is not None:
        ax_next = plt.axes([current_pos, button_y, button_width, button_height])
        button_next = Button(ax_next, "Next")
    else:
        ax_next = None
        button_next = None

    # UI axes 一覧（ズーム・パンの対象から外すため）
    ui_axes = [
        ax_slider_hpf,
        ax_slider_threshold,
        ax_slider_amp_threshold,
        ax_slider_peak_amp,
        ax_slider_smooth,
        ax_redetect,
    ]
    if ax_export is not None:
        ui_axes.append(ax_export)
    if ax_next is not None:
        ui_axes.append(ax_next)

    # ========= ボタンコールバック ===========================================

    def do_redetect(preserve_xlim: bool = True) -> None:
        """スライダ値を読んで再検出し、プロット更新。"""
        state["hp"] = float(slider_hpf.val)
        state["thr"] = float(slider_threshold.val) / 100.0
        state["amp_thr"] = float(slider_amp_threshold.val) / 100.0
        state["peak_amp"] = float(slider_peak_amp.val) / 100.0
        state["smooth_ms"] = float(slider_smooth.val)

        taps_new, y_new, env_new = run_detection(
            y,
            sr,
            state["hp"],
            state["thr"],
            state["amp_thr"],
            state["peak_amp"],
            state["smooth_ms"],
        )
        state["taps"] = taps_new

        update_plot(y_new, env_new, taps_new, preserve_xlim=preserve_xlim)

    def on_redetect(event) -> None:
        do_redetect(preserve_xlim=True)

    button_redetect.on_clicked(on_redetect)

    def on_export(event) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        default_name = os.path.splitext(os.path.basename(wav_path))[0] + "_taps.csv"

        while True:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            file_path = filedialog.asksaveasfilename(
                title="Export tap detection results",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )

            if not file_path:
                # User cancelled
                root.destroy()
                break

            try:
                export_taps_to_csv(file_path, state["taps"])
                print(f"Exported to: {file_path}")
                root.destroy()
                break
            except PermissionError:
                messagebox.showerror(
                    "Export Error",
                    f"Permission denied: Cannot write to\n{file_path}\n\n"
                    "Please choose a different location.",
                    parent=root
                )
                root.destroy()
                # Continue loop to let user choose another location
            except OSError as e:
                messagebox.showerror(
                    "Export Error",
                    f"Failed to write file:\n{file_path}\n\n"
                    f"Error: {e}\n\n"
                    "Please choose a different location.",
                    parent=root
                )
                root.destroy()
                # Continue loop to let user choose another location

    if enable_export and button_export is not None:
        button_export.on_clicked(on_export)

    def on_next(event) -> None:
        plt.close(fig)
        if on_next_callback is not None:
            on_next_callback()

    if button_next is not None:
        button_next.on_clicked(on_next)

    # ========= ズーム / パン / ダブルクリック / 範囲選択削除 ================

    pan_data = {
        "pressed": False,
        "x0": None,
        "y0": None,
        "xlim0": None,
        "ylim0": None,
        "ax": None,
    }

    # Shift+ドラッグによる範囲選択用データ
    select_data = {
        "active": False,
        "x0": None,
        "ax": None,
        "rect1": None,  # ax1用の選択矩形
        "rect2": None,  # ax2用の選択矩形
    }

    def on_scroll(event) -> None:
        if event.inaxes is None or is_ui_axis(event.inaxes):
            return

        ax = event.inaxes
        key = (event.key or "").lower()

        if key in ("control", "ctrl", "cmd", "super"):
            cur_ylim = ax.get_ylim()
            ydata = event.ydata
            if ydata is None:
                return
            zoom = 1.2
            scale = 1 / zoom if event.button == "up" else zoom
            new_h = (cur_ylim[1] - cur_ylim[0]) * scale
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            new_ylim = [ydata - new_h * (1 - rely), ydata + new_h * rely]
            ax.set_ylim(new_ylim)
            fig.canvas.draw_idle()
        else:
            cur_xlim = ax.get_xlim()
            xdata = event.xdata
            if xdata is None:
                return
            zoom = 1.2
            scale = 1 / zoom if event.button == "up" else zoom
            new_w = (cur_xlim[1] - cur_xlim[0]) * scale
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            new_xlim = [xdata - new_w * (1 - relx), xdata + new_w * relx]
            ax.set_xlim(new_xlim)
            fig.canvas.draw_idle()

    def _has_shift(event) -> bool:
        """Shiftキーが押されているか判定"""
        key = (event.key or "").lower()
        has_shift = "shift" in key
        if hasattr(event, "modifiers"):
            mods = str(event.modifiers).lower()
            has_shift = has_shift or "shift" in mods
        return has_shift

    def on_press(event) -> None:
        if event.inaxes is None or is_ui_axis(event.inaxes):
            return
        if event.button != 1:
            return

        # Shift+ドラッグ: 範囲選択モード
        if _has_shift(event) and event.inaxes in (ax1, ax2):
            select_data["active"] = True
            select_data["x0"] = event.xdata
            select_data["ax"] = event.inaxes
            # 選択矩形を作成（両方の軸に表示）
            ylim1 = ax1.get_ylim()
            ylim2 = ax2.get_ylim()
            select_data["rect1"] = ax1.axvspan(
                event.xdata, event.xdata, alpha=0.3, color="red"
            )
            select_data["rect2"] = ax2.axvspan(
                event.xdata, event.xdata, alpha=0.3, color="red"
            )
            fig.canvas.draw_idle()
            return

        # 通常のパン操作
        pan_data["pressed"] = True
        pan_data["x0"] = event.xdata
        pan_data["y0"] = event.ydata
        pan_data["ax"] = event.inaxes
        pan_data["xlim0"] = event.inaxes.get_xlim()
        pan_data["ylim0"] = event.inaxes.get_ylim()

    def on_release(event) -> None:
        # 範囲選択モードの終了処理
        if select_data["active"]:
            select_data["active"] = False

            # 選択矩形を削除
            if select_data["rect1"] is not None:
                select_data["rect1"].remove()
                select_data["rect1"] = None
            if select_data["rect2"] is not None:
                select_data["rect2"].remove()
                select_data["rect2"] = None

            # 範囲内のタップを削除
            if event.xdata is not None and select_data["x0"] is not None:
                x_start = min(select_data["x0"], event.xdata)
                x_end = max(select_data["x0"], event.xdata)
                range_width = x_end - x_start

                deleted_any = False

                # 範囲が十分大きい場合: 範囲内一括削除
                if range_width > 0.05:  # 50ms以上の範囲
                    to_delete = []
                    for i, t in enumerate(state["taps"]):
                        # tap_start または tap_peak が範囲内にあれば削除対象
                        if x_start <= t["tap_start"] <= x_end or x_start <= t["tap_peak"] <= x_end:
                            to_delete.append(i)

                    # 逆順で削除（インデックスがずれないように）
                    if to_delete:
                        for i in reversed(to_delete):
                            deleted = state["taps"].pop(i)
                            print(
                                f"Deleted tap at ~{deleted['tap_start']:.3f}s / {deleted['tap_peak']:.3f}s"
                            )
                        print(f"Total deleted: {len(to_delete)} taps")
                        deleted_any = True

                # 範囲が小さい場合（クリックに近い）: 最近傍の単一タップ削除
                else:
                    click_t = select_data["x0"]
                    thr_sec = 0.05  # 50ms以内なら同一タップとみなす
                    idx = None
                    best_dt = float("inf")

                    for i, t in enumerate(state["taps"]):
                        dt = min(
                            abs(t["tap_start"] - click_t),
                            abs(t["tap_peak"] - click_t),
                        )
                        if dt < best_dt:
                            best_dt = dt
                            idx = i

                    if idx is not None and best_dt <= thr_sec:
                        deleted = state["taps"].pop(idx)
                        print(
                            f"Deleted tap at ~{deleted['tap_start']:.3f}s / {deleted['tap_peak']:.3f}s"
                        )
                        deleted_any = True

                if deleted_any:
                    # 再描画（ズーム維持）
                    y_new = butter_highpass_zero_phase(y, sr, state["hp"])
                    env_new = hilbert_envelope(y_new, sr, smooth_ms=state["smooth_ms"])
                    update_plot(y_new, env_new, state["taps"], preserve_xlim=True)
                    return

            fig.canvas.draw_idle()
            return

        pan_data["pressed"] = False

    def on_motion(event) -> None:
        # 範囲選択モード中の矩形更新
        if select_data["active"]:
            if event.xdata is not None and select_data["x0"] is not None:
                x0 = select_data["x0"]
                x1 = event.xdata
                x_min, x_max = min(x0, x1), max(x0, x1)

                # 選択矩形を更新
                if select_data["rect1"] is not None:
                    xy = select_data["rect1"].get_xy()
                    xy[:, 0] = [x_min, x_min, x_max, x_max, x_min]
                    select_data["rect1"].set_xy(xy)
                if select_data["rect2"] is not None:
                    xy = select_data["rect2"].get_xy()
                    xy[:, 0] = [x_min, x_min, x_max, x_max, x_min]
                    select_data["rect2"].set_xy(xy)

                fig.canvas.draw_idle()
            return

        # 通常のパン操作
        if not pan_data["pressed"]:
            return
        if event.inaxes != pan_data["ax"]:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - pan_data["x0"]
        dy = event.ydata - pan_data["y0"]
        xlim = pan_data["xlim0"]
        ylim = pan_data["ylim0"]
        pan_data["ax"].set_xlim([xlim[0] - dx, xlim[1] - dx])
        pan_data["ax"].set_ylim([ylim[0] - dy, ylim[1] - dy])
        fig.canvas.draw_idle()

    def on_double_click(event) -> None:
        if event.inaxes is None or is_ui_axis(event.inaxes):
            return
        if not event.dblclick:
            return

        ax = event.inaxes
        xlim = ax.get_xlim()

        if ax == ax1:
            visible = waveform_line.get_ydata()
        elif ax == ax2:
            visible = envelope_line.get_ydata()
        else:
            return

        mask = (time_axis >= xlim[0]) & (time_axis <= xlim[1])
        visible = visible[mask]
        if len(visible) == 0:
            return

        max_val = np.max(np.abs(visible))
        if max_val == 0:
            return

        target_max = max_val / 0.85
        ax.set_ylim([-target_max, target_max])
        fig.canvas.draw_idle()

    # ツールチップ表示用のマウス移動ハンドラ
    def on_tooltip_motion(event) -> None:
        """スライダ上にマウスがあればツールチップを表示"""
        if event.inaxes is None:
            tooltip_text.set_text("")
            fig.canvas.draw_idle()
            return

        # スライダの軸にマウスがあるかチェック
        tooltip = slider_tooltips.get(event.inaxes, "")
        if tooltip != tooltip_text.get_text():
            tooltip_text.set_text(tooltip)
            fig.canvas.draw_idle()

    # イベント登録
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("motion_notify_event", on_tooltip_motion)
    fig.canvas.mpl_connect("button_press_event", on_double_click)

    plt.show()
