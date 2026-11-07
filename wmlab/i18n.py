"""Ngôn ngữ thông báo: đặt biến môi trường WMLAB_LANG=en (hoặc --lang en) để in tiếng Anh."""
import os


def lang():
    return os.environ.get("WMLAB_LANG", "vi").lower()


def tr(vi, en):
    return en if lang().startswith("en") else vi
