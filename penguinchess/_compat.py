"""
Compatibility helpers for cross-platform support.
"""

import sys


def ensure_utf8_stdout() -> None:
    """确保 stdout 使用 UTF-8 编码，避免 Windows 控制台中文乱码。"""
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass
