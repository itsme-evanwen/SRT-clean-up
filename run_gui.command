#!/bin/bash
# 雙擊啟動 SRT 字幕整理工具（macOS）
cd "$(dirname "$0")"
python3 srt_gui.py "$@"
