from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

from tqdm.auto import tqdm as _tqdm


class TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        for stream in self.streams:
            encoding = getattr(stream, "encoding", None)
            if encoding:
                return encoding
        return "utf-8"

    @property
    def buffer(self):
        for stream in self.streams:
            buffer = getattr(stream, "buffer", None)
            if buffer is not None:
                return buffer
        raise AttributeError("buffer")

    def fileno(self) -> int:
        for stream in self.streams:
            fileno = getattr(stream, "fileno", None)
            if callable(fileno):
                try:
                    return fileno()
                except Exception:
                    continue
        raise OSError("No fileno available")


class NullTerminalStream:
    def write(self, data: str) -> int:
        return len(data)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return "utf-8"


def _get_original_stream(error: bool = False) -> TextIO:
    if error:
        return getattr(sys, "_aih_original_stderr", sys.__stderr__)
    return getattr(sys, "_aih_original_stdout", sys.__stdout__)


def _write_to_log(text: str) -> None:
    log_file = getattr(sys, "_aih_log_file", None)
    if log_file is None:
        return
    log_file.write(text)
    log_file.flush()


def emit_console(message: Any = "", *, error: bool = False) -> None:
    text = str(message)
    if not text.endswith("\n"):
        text += "\n"

    stream = _get_original_stream(error=error)
    stream.write(text)
    stream.flush()
    _write_to_log(text)


def emit_log(message: Any = "") -> None:
    text = str(message)
    if not text.endswith("\n"):
        text += "\n"
    _write_to_log(text)


def get_current_log_path() -> Optional[Path]:
    path = getattr(sys, "_aih_log_path", None)
    if not path:
        return None
    return Path(path)


def emit_exception(exc: BaseException) -> None:
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    if detail:
        emit_console(detail, error=True)


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def step_tqdm(*args, **kwargs):
    kwargs.setdefault("disable", True)
    kwargs.setdefault("leave", False)
    return _tqdm(*args, **kwargs)


class StepReporter:
    def __init__(self, step_name: str, total: Optional[int] = None, unit: str = "file"):
        self.step_name = step_name
        self.total = total
        self.unit = unit
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.started_at = time.time()

    def _progress_text(self) -> str:
        if self.total is None:
            return f"{self.completed}"
        return f"{self.completed}/{self.total}"

    def start(
        self,
        *,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        extra: Optional[str] = None,
    ) -> None:
        parts = [self.step_name, "开始"]
        if self.total is not None:
            parts.append(f"{self.unit}={self.total}")
        if input_dir is not None:
            parts.append(f"输入={Path(input_dir)}")
        if output_dir is not None:
            parts.append(f"输出={Path(output_dir)}")
        if extra:
            parts.append(extra)
        emit_console(" | ".join(parts))

    def info(self, message: str) -> None:
        emit_console(f"{self.step_name} | {message}")

    def item_ok(self, name: str, detail: Optional[str] = None) -> None:
        self.completed += 1
        self.success += 1
        parts = [self.step_name, self._progress_text(), "OK", name]
        if detail:
            parts.append(detail)
        emit_console(" | ".join(parts))

    def item_skip(self, name: str, detail: Optional[str] = None) -> None:
        self.completed += 1
        parts = [self.step_name, self._progress_text(), "SKIP", name]
        if detail:
            parts.append(detail)
        emit_console(" | ".join(parts))

    def item_fail(self, name: str, exc: BaseException, detail: Optional[str] = None) -> None:
        self.completed += 1
        self.failed += 1
        parts = [self.step_name, self._progress_text(), "FAIL", name, f"{type(exc).__name__}: {exc}"]
        if detail:
            parts.append(detail)
        emit_console(" | ".join(parts))
        emit_exception(exc)

    def finish(self, *, output_dir: Optional[Path] = None, extra: Optional[str] = None) -> None:
        parts = [
            self.step_name,
            "结束",
            f"成功={self.success}",
            f"失败={self.failed}",
            f"耗时={format_duration(time.time() - self.started_at)}",
        ]
        if output_dir is not None:
            parts.append(f"输出={Path(output_dir)}")
        if extra:
            parts.append(extra)
        emit_console(" | ".join(parts))


def setup_step_logging(run_root: Path, step_name: str) -> Path:
    run_root = Path(run_root)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{step_name}_{timestamp}.log"

    base_stdout = getattr(sys, "_aih_original_stdout", sys.stdout)
    base_stderr = getattr(sys, "_aih_original_stderr", sys.stderr)
    old_file = getattr(sys, "_aih_log_file", None)
    if old_file is not None:
        try:
            old_file.flush()
            old_file.close()
        except Exception:
            pass

    log_file = log_path.open("a", encoding="utf-8", buffering=1)

    sys._aih_original_stdout = base_stdout
    sys._aih_original_stderr = base_stderr
    sys.stdout = TeeStream(NullTerminalStream(), log_file)
    sys.stderr = TeeStream(base_stderr, log_file)
    sys._aih_log_path = str(log_path)
    sys._aih_log_step_name = step_name
    sys._aih_log_file = log_file

    emit_console(f"[log] writing to {log_path}")
    emit_console(f"[log] cwd: {Path.cwd().resolve()}")
    emit_console(f"[log] argv: {' '.join(sys.argv)}")

    return log_path
