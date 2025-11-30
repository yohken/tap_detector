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
        smooth: float,
    ) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
        """指定パラメータで tap detection を実行し、taps / y_filt / env を返す。"""
        taps_local = detect_tap_onsets_and_peaks(
            y,
            sr,
            hp_cutoff=hp,
            threshold_ratio=thr,
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

        if preserve_xlim:
            xlim1 = ax1.get_xlim()
            xlim2 = ax2.get_xlim()
            ylim1 = ax1.get_ylim()
            ylim2 = ax2.get_ylim()

        waveform_line.set_ydata(y_new)
        envelope_line.set_ydata(env_new)

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
        "smooth_ms": float(smooth_ms),
        "taps": [],
    }

    taps, y_filt, env = run_detection(
        y, sr, state["hp"], state["thr"], state["smooth_ms"]
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

    plt.subplots_adjust(bottom=0.34)

    # スライダ: 3 段（中央寄せ、幅 0.70）
    ax_slider_hpf = plt.axes([0.15, 0.26, 0.70, 0.03])
    slider_hpf = Slider(
        ax_slider_hpf,
        "HPF (Hz)",
        100.0,
        2000.0,
        valinit=state["hp"],
        valstep=50.0,
    )

    ax_slider_threshold = plt.axes([0.15, 0.20, 0.70, 0.03])
    slider_threshold = Slider(
        ax_slider_threshold,
        "Threshold (%)",
        1.0,
        50.0,
        valinit=state["thr"] * 100.0,
        valstep=1.0,
    )

    ax_slider_smooth = plt.axes([0.15, 0.14, 0.70, 0.03])
    slider_smooth = Slider(
        ax_slider_smooth,
        "Smooth (ms)",
        0.3,
        2.0,
        valinit=state["smooth_ms"],
        valstep=0.1,
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
        state["smooth_ms"] = float(slider_smooth.val)

        taps_new, y_new, env_new = run_detection(
            y,
            sr,
            state["hp"],
            state["thr"],
            state["smooth_ms"],
        )
        state["taps"] = taps_new

        update_plot(y_new, env_new, taps_new, preserve_xlim=preserve_xlim)

    def on_redetect(event) -> None:
        do_redetect(preserve_xlim=True)

    button_redetect.on_clicked(on_redetect)

    def on_export(event) -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        default_name = os.path.splitext(os.path.basename(wav_path))[0] + "_taps.csv"

        file_path = filedialog.asksaveasfilename(
            title="Export tap detection results",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        root.destroy()

        if file_path:
            export_taps_to_csv(file_path, state["taps"])
            print(f"Exported to: {file_path}")

    if enable_export and button_export is not None:
        button_export.on_clicked(on_export)

    def on_next(event) -> None:
        plt.close(fig)
        if on_next_callback is not None:
            on_next_callback()

    if button_next is not None:
        button_next.on_clicked(on_next)

    # ========= ズーム / パン / ダブルクリック / マーカー削除 ================

    pan_data = {
        "pressed": False,
        "x0": None,
        "y0": None,
        "xlim0": None,
        "ylim0": None,
        "ax": None,
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

    def on_press(event) -> None:
        if event.inaxes is None or is_ui_axis(event.inaxes):
            return
        if event.button != 1:
            return
        pan_data["pressed"] = True
        pan_data["x0"] = event.xdata
        pan_data["y0"] = event.ydata
        pan_data["ax"] = event.inaxes
        pan_data["xlim0"] = event.inaxes.get_xlim()
        pan_data["ylim0"] = event.inaxes.get_ylim()

    def on_release(event) -> None:
        pan_data["pressed"] = False

    def on_motion(event) -> None:
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

    # Shift+Click で最近傍の tap を削除
    def on_marker_click(event) -> None:
        if event.inaxes not in (ax1, ax2):
            return
        if event.xdata is None:
            return

        key = (event.key or "").lower()
        has_shift = "shift" in key
        if hasattr(event, "modifiers"):
            mods = str(event.modifiers).lower()
            has_shift = has_shift or "shift" in mods
        if not has_shift:
            return

        click_t = event.xdata
        thr_sec = 0.02  # 20ms 以内なら同一タップとみなす

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

        if idx is None or best_dt > thr_sec:
            return

        deleted = state["taps"].pop(idx)
        print(
            f"Deleted tap at ~{deleted['tap_start']:.3f}s / {deleted['tap_peak']:.3f}s"
        )

        # 現在のパラメータで再描画（ズーム維持）
        y_new = butter_highpass_zero_phase(y, sr, state["hp"])
        env_new = hilbert_envelope(y_new, sr, smooth_ms=state["smooth_ms"])
        update_plot(y_new, env_new, state["taps"], preserve_xlim=True)

    # イベント登録
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_double_click)
    fig.canvas.mpl_connect("button_press_event", on_marker_click)

    plt.show()
