"""
tap_export.py
CSV export utilities for tap detection results.

列:
    file_name, tap_start, tap_peak, hp_cutoff, threshold
"""

from __future__ import annotations
from typing import List, Dict
import os
import csv


COLUMNS = ["file_name", "tap_start", "tap_peak", "hp_cutoff", "threshold"]


def export_taps_to_csv(path: str, taps: List[Dict[str, float]]) -> None:
    """
    タップ検出結果を CSV に保存する。

    Args:
        path: 出力 CSV パス
        taps: detect_taps_from_wav の戻り値（dict の list）
    """
    # file_name を basename のみにするかどうかは好みだが、
    # ひとまずそのまま（フルパス）にしておく。
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for t in taps:
            row = {col: t.get(col, "") for col in COLUMNS}
            writer.writerow(row)
